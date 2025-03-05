from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": 0,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            # "risk_reward": -0.5,
            "risk_reward": 0,
            "comfort_reward": 0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,
            "usempc_controller": True,

        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action, delta_a) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """

        rewards = self._rewards(action, delta_a)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                               [self.config["risk_reward"],
                                 self.config["high_speed_reward"]],
                                [-1, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action, delta_a) -> Dict[Text, float]:
        # if self._is_truncated():
        #     print("success")
        r_comfort = 0.2 * abs(delta_a[0] / 5) + 0.8 * abs(delta_a[1] * 10)
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # if 20 <= self.vehicle.speed <= 30:
        #     scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [-1, 1])
        # else:
        #     scaled_speed = -2
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [-1, 1])
        risk = 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road) + self._cost()
        return {
            # "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, -1, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "comfort_reward": np.clip(r_comfort, 0, 1),
            "risk_reward": risk,
        }

    def _cost(self):
        Asta = 1
        Adyn = 1
        Ab = 1  # Field intensity coefficient for the road boundary
        Al = 0.1  # Field intensity coefficient for the lane marking
        σb = 1  # Risk distribution range for the road boundary
        σl = 1  # Risk distribution range for the lane marking
        kx = 1
        ky = 0.8
        kv = 2
        alpha = 0.9
        beta = 2
        L_obs = 5
        W_obs = 2
        positions = [vehicle.position for vehicle in self.road.vehicles]
        speeds = [vehicle.speed for vehicle in self.road.vehicles]
        headings = [vehicle.heading for vehicle in self.road.vehicles]
        vehicles_obs = np.column_stack((np.array(positions), np.array(speeds), np.array(headings)))
        x_ego, y_ego, v_ego, _ = vehicles_obs[0]
        # Road boundary positions
        y_boundary = [-2, 14]
        # Lane marking positions
        y_lane = [2, 6, 10]
        # Potential field for road boundaries
        Eb = np.zeros_like(x_ego)
        for boundary in y_boundary:
            Eb += Ab * np.exp(-((y_ego - boundary) ** 2) / (1 * σb ** 2))
        # Potential field for lane markings
        El = np.zeros_like(x_ego)
        for lane in y_lane:
            El += Al * np.exp(-((y_ego - lane) ** 2) / (1 * σl ** 2))
        Ustas = []
        Udyns = []
        for vehicle in vehicles_obs[1:]:
            x_obs, y_obs, v_obs, h_obs = vehicle
            σx = kx * L_obs
            σy = ky * W_obs
            σv = kv * np.abs(v_obs - v_ego)
            # 计算相对速度方向
            if v_obs >= v_ego:
                relv = 1
            else:
                relv = -1

            # 修正坐标旋转公式
            h_obs = -h_obs  # 这一步非常重要，不然有航向角的障碍物的势场就反了
            x_rel = (x_ego - x_obs) * np.cos(h_obs) - (y_ego - y_obs) * np.sin(h_obs) + x_obs
            y_rel = (x_ego - x_obs) * np.sin(h_obs) + (y_ego - y_obs) * np.cos(h_obs) + y_obs

            Usta = Asta * np.exp(
                -((x_rel - x_obs) ** 2 / σx ** 2) ** beta - ((y_rel - y_obs) ** 2 / σy ** 2) ** beta)
            Udyn = Adyn * (np.exp(-((x_rel - x_obs) ** 2 / σv ** 2 + (y_rel - y_obs) ** 2 / σy ** 2)) /
                           (1 + np.exp(-relv * (x_rel - x_obs - alpha * L_obs * relv))))
            Ustas.append(Usta)
            Udyns.append(Udyn)
        Ustas = np.array(Ustas).sum()
        Udyns = np.array(Udyns).sum()
        Ebs = Eb.sum()
        Els = El.sum()
        U = Ustas + Udyns + Ebs + Els
        # U = Ustas + Udyns
        return U

    # def _cost(self):
    #     A = 1
    #     kx = 2
    #     ky = 1
    #     kv = 2
    #     alpha = 0.9
    #     beta = 2
    #     L_obs = 5
    #     W_obs = 2
    #     positions = [vehicle.position for vehicle in self.road.vehicles]
    #     speeds = [vehicle.speed for vehicle in self.road.vehicles]
    #     vehicles_obs = np.column_stack((np.array(positions), np.array(speeds)))
    #     x_ego, y_ego, v_ego = vehicles_obs[0]
    #     Ustas = []
    #     Udyns = []
    #     for vehicle in vehicles_obs[1:]:
    #         x_obs, y_obs, v_obs = vehicle
    #         σx = kx * L_obs
    #         σy = ky * W_obs
    #         σv = kv * np.abs(v_obs - v_ego)
    #         # 计算相对速度方向
    #         if v_obs >= v_ego:
    #             relv = 1
    #         else:
    #             relv = -1
    #
    #         Usta = A * np.exp(-((x_ego - x_obs) ** 2 / σx ** 2) ** beta - ((y_ego - y_obs) ** 2 / σy ** 2) ** beta)
    #         Ustas.append(Usta)
    #         # 计算动态风险场
    #         Udyn = (A * np.exp(-((x_ego - x_obs) ** 2 / σv ** 2 + (y_ego - y_obs) ** 2 / σy ** 2)) /
    #                 (1 + np.exp(-relv * (x_ego - x_obs - alpha * L_obs * relv))))
    #         Udyns.append(Udyn)
    #     Ustas = np.array(Ustas).sum()
    #     Udyns = np.array(Udyns).sum()
    #     U = Ustas + Udyns
    #     return U


    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        # if self.time >= self.config["duration"]:
        #     print("成功")
        return self.time >= self.config["duration"]


class SafeHighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-4, 4],
                "steering_range": [-0.1, 0.1]
            },
            "simulation_frequency": 25,
            "policy_frequency": 10,  # [Hz]
            "lanes_count": 4,
            "vehicles_count": 20,
            "duration": 20,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
