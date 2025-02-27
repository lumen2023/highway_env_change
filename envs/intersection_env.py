from typing import Dict, Tuple, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle

class IntersectionEnv(AbstractEnv):
    # 交叉路口环境类，继承自 AbstractEnv

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',  # 减速
        1: 'IDLE',  # 保持
        2: 'FASTER'  # 加速
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        # 返回默认配置字典
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",  # 观测类型
                "vehicles_count": 15,  # 车辆数量
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],  # 特征列表
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,  # 使用绝对坐标
                "flatten": False,  # 是否将观测展平为一维向量
                "observe_intentions": False  # 是否观测车辆意图
            },
            "action": {
                "type": "DiscreteMetaAction",  # 动作类型
                "longitudinal": True,  # 是否有纵向动作
                "lateral": False,  # 是否有横向动作
                "target_speeds": [0, 4.5, 9]  # 目标速度列表
            },
            "duration": 13,  # 场景持续时间 [s]
            "destination": "o1",  # 目标位置
            "controlled_vehicles": 1,  # 被控车辆数量
            "initial_vehicle_count": 10,  # 初始车辆数量
            "spawn_probability": 0.6,  # 车辆生成概率
            "screen_width": 600,  # 屏幕宽度
            "screen_height": 600,  # 屏幕高度
            "centering_position": [0.5, 0.6],  # 屏幕中心位置
            "scaling": 5.5 * 1.3,  # 缩放因子
            "collision_reward": -5,  # 碰撞惩罚
            "high_speed_reward": 1,  # 高速奖励
            "arrived_reward": 1,  # 到达奖励
            "reward_speed_range": [7.0, 9.0],  # 奖励速度范围
            "normalize_reward": False,  # 是否归一化奖励
            "offroad_terminal": False  # 是否在离开道路时结束场景
        })
        return config

    def _reward(self, action: int) -> float:
        # 计算所有被控车辆的平均奖励
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> Dict[Text, float]:
        # 计算多目标奖励
        agents_rewards = [self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        # 计算单个被控车辆的奖励
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        # 计算单个被控车辆的每个目标的奖励
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": vehicle.crashed,  # 碰撞奖励
            "high_speed_reward": np.clip(scaled_speed, 0, 1),  # 高速奖励
            "arrived_reward": self.has_arrived(vehicle),  # 到达奖励
            "on_road_reward": vehicle.on_road  # 在道路上奖励
        }

    def _is_terminated(self) -> bool:
        # 判断场景是否结束
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        # 判断单个被控车辆是否终止
        return (vehicle.crashed or self.has_arrived(vehicle))

    def _is_truncated(self) -> bool:
        # 判断场景是否被截断（例如时间限制）
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        # 提供有关当前状态的信息
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        return info

    def _reset(self) -> None:
        # 重置环境，创建道路和车辆
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 执行动作并更新环境
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        # 创建一个四路交叉路口
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # 右转弯半径 [m]
        left_turn_radius = right_turn_radius + lane_width  # 左转弯半径 [m]
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # 入口道路长度 [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # 入口
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # 右转
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # 左转
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # 直行
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # 出口
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        # 创建并添加多辆车辆到道路上
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # 低拥堵距离
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # 随机车辆
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # 挑战车辆
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # 被控车辆
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_vehicle = self.action_type.vehicle_class(
                             self.road,
                             ego_lane.position(60 + 5*self.np_random.normal(1), 0),
                             speed=ego_lane.speed_limit,
                             heading=ego_lane.heading_at(60))
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # 防止提前发生碰撞
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        # 随机生成一辆车辆
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=8 + self.np_random.normal() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        # 清理离开道路的车辆
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        # 判断车辆是否到达目标位置
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance


class MultiAgentIntersectionEnv(IntersectionEnv):
    # 多智能体交叉路口环境类，继承自 IntersectionEnv
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",  # 多智能体动作类型
                 "action_config": {
                     "type": "DiscreteMetaAction",  # 离散动作元行为
                     "lateral": False,
                     "longitudinal": True
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",  # 多智能体观测类型
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "controlled_vehicles": 2  # 控制两辆车辆
        })
        return config


class ContinuousIntersectionEnv(IntersectionEnv):
    # 连续动作交叉路口环境类，继承自 IntersectionEnv
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",  # 连续观测类型
                "vehicles_count": 5,  # 车辆数量
                "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],  # 特征列表
            },
            "action": {
                "type": "ContinuousAction",  # 连续动作类型
                "steering_range": [-np.pi / 3, np.pi / 3],  # 转向范围
                "longitudinal": True,
                "lateral": True,
                "dynamical": True  # 是否动态控制
            },
        })
        return config
    def _create_vehicles(self) -> None:
        # 调用父类的方法创建车辆
        super()._create_vehicles()
        # 禁用对未控制车辆的碰撞检测
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False  # 对于非受控车辆，不进行碰撞检测


# TupleMultiAgentIntersectionEnv = MultiAgentWrapper(MultiAgentIntersectionEnv)
