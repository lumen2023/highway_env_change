from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.utils import near_split
from highway_env.vehicle.kinematics import Vehicle

class SafeMergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
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
            # "action": {
            #     "type": "NoAction",
            # },
            "simulation_frequency": 30,
            "policy_frequency": 10,  # [Hz] #控制步长0.1s
            "screen_width": 1800,  # [px]
            "screen_height": 600,  # [px]
            "main_lanes_count": 1,
            "target_lane": 0,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "normalize_reward": False,
            "offroad_terminal": True,
            "collision_reward": -5,
            "right_lane_reward": 0,
            "high_speed_reward": 1,
            "merging_speed_reward": 0,
            "goal_reward": 1,
            "Headway_reward": 1,
            "acc_lane_reward": 1,
            "speed_difference_reward": 2,
            "center_reward": 1,
            "left_lane_reward": 0.05,
            "reward_speed_range": [15, 25],
            "speed_difference_range": [15, 0],
            "cost_speed_range": [20, 15],
            "duration": 5,
            # "usempc_controller": True,
            "usempc_controller": False,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """

        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,[-1,1],[0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        Norm_Headway_cost, Target_v = self._compute_headway_distance(self.vehicle)
        speed_difference = abs(forward_speed - Target_v)
        scaled_speed = np.tanh(1 * (2 - speed_difference))
        # scaled_speed = utils.lmap(speed_difference, self.config["speed_difference_range"], [-1, 1])
        # print(forward_speed, Target_v, speed_difference, scaled_speed)
        # scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [-1, 1])

        if self.vehicle.lane_index == ('b', 'c', self.config["main_lanes_count"]): #如果自车在加速车道上
            # print(self.vehicle.lane_index)
            acc_lane = - np.exp(-(self.vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))
        else:
            acc_lane = 0
        # print(self.vehicle.lane_index, acc_lane)
        # print(self.vehicle.lane_index, self.vehicle.heading)
        # print(self.vehicle.lane_index, self.vehicle.heading)
        # if 0 in self.vehicle.lane_index and self.vehicle.position[1] < 4:  # 如果自车在目标车道上
        if 0 in self.vehicle.lane_index:  # 如果自车在目标车道上
            # center_lane = -(10*(self.vehicle.position[1] - 4) ** 2 + (self.vehicle.heading - 0) ** 2)
            center_lane = -((self.vehicle.position[1] - 4) ** 2)
            center_lane = utils.lmap(center_lane, [-4, 0], [-1, 0])
        else:
            center_lane = 0
        # print(self.vehicle.lane_index, self.vehicle.position[1], "{:.10f}".format(center_lane))
        goal_reward = self._is_success()
        # print(goal_reward)
        # print(self.vehicle.position[1])
        # print(acc_lane)
        # print(self.vehicle.position[0], sum(self.ends[:3]))
        # print(acc_lane)
        # if self.vehicle.lane_index == ('b', 'c', 1) or self.vehicle.lane_index == ('b', 'c', 0):
        #     left_lane = utils.lmap(self.vehicle.position[0], [200, 300], [0, 1])
        # else:
        #     left_lane = 0
        # compute headway cost
        # Headway_cost = np.log(
        #     headway_distance / (self.config["HEADWAY_TIME"] * self.vehicle.speed)) if self.vehicle.speed > 0 else 0
        # print(float(self.vehicle.on_road), np.clip(Norm_Headway_cost, -1, 1), acc_lane, scaled_speed)
        # return {
        #     "on_road_reward": float(self.vehicle.on_road),
        #     "Headway_reward": np.clip(Norm_Headway_cost, -1, 1),
        #     "acc_lane_reward": acc_lane,
        #     "high_speed_reward": np.clip(scaled_speed, -1, 1),
        #     "goal_reward": goal_reward,
        # }
        # print(float(self.vehicle.crashed))
        return {
                # "collision_reward": float(self.vehicle.crashed),
                "on_road_reward": float(self.vehicle.on_road),
                "Headway_reward": np.clip(Norm_Headway_cost, -1, 1),
                "acc_lane_reward": acc_lane,
                "speed_difference_reward": np.clip(scaled_speed, -1, 1),
                "center_reward": np.clip(center_lane, -1, 0),
                "goal_reward": goal_reward,
            }

    def _is_success(self):
        if self.time >= self.config["duration"] and 0 in self.vehicle.lane_index:
            goal_reached = 5
        else:
            goal_reached = 0
        return goal_reached

    # def _is_terminated(self) -> bool:
    #     """The episode is over when a collision occurs or when the access ramp has been passed."""
    #     return (self.vehicle.crashed or bool(self.vehicle.position[0] > 370) or
    #             self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (self.vehicle.crashed or self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        # print(self.time)
        return self.time >= self.config["duration"]

    # def _is_terminated(self) -> bool:
    #     """The episode is over when a collision occurs or when the access ramp has been passed."""
    #     return False
    #
    #
    # def _is_truncated(self) -> bool:
    #     """The episode is truncated if the time limit is reached."""
    #     # print(self.time)
    #     return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        self.ends = [150, 80, 80, 150]  # Before, converging, merge, after
        ends = self.ends
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # y = [-4, 0, StraightLane.DEFAULT_WIDTH]
        # y = [-4, 0, 4, 8]
        y = [4, 0, -4, -8]
        "c:连续 s：虚线 n：无线"
        line_type = [[c, s], [n, c], [c, c]]
        line_type_merge = [[c, s], [n, s]]
        a = line_type[0]
        # for i in range(2):
        #     net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i],speed_limit=30))
        #     net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i],speed_limit=30))
        #     net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i],speed_limit=30))

        for i in range(self.config["main_lanes_count"]):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[2], speed_limit=30))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[0],
                                                speed_limit=30))
            net.add_lane("c", "d",
                         StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[2], speed_limit=30))

        # i = 1
        # net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[2], speed_limit=30))
        # net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[0],
        #                                     speed_limit=30))
        # net.add_lane("c", "d",
        #              StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[2], speed_limit=30))
        #
        # i = 2
        # net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[2],speed_limit=30))
        # net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[0],speed_limit=30))
        # net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[2],speed_limit=30))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[s, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        # print(lbc.position(ends[2], 0))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     self.road.network.get_lane(("b", "c", self.config["main_lanes_count"])).position(0, 0),
                                                     speed=15,
                                                     # heading=-0.09,
                                                     heading = 0,
                                                     )
        # ego_vehicle = self.action_type.vehicle_class(self.road,
        #                                              self.road.network.get_lane(
        #                                                  ("k", "b", 0)).position(20, 0),
        #                                              speed=20,
        #                                              # heading=-0.09,
        #                                              heading=0,
        #                                              )
        self.road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        for others in other_per_controlled:
            # vehicle = Vehicle.create_random(
            #     self.road,
            #     speed=25,
            #     lane_id=1,
            #     spacing=self.config["ego_spacing"]
            # )
            # vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            # self.controlled_vehicles.append(vehicle)
            # self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random_merge(self.road, lane_from="a" ,lane_to="b", spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)


        # self.road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        # self.road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # self.road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        # merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        # merging_v.target_speed = 30
        # road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
