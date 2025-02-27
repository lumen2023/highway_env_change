import functools
import itertools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable, List
from gymnasium import spaces
import numpy as np

from highway_env import utils
from highway_env.utils import Vector
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.behavior import AggressiveVehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.controller import LYZ_MDPVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.controller import Half_ControlledVehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType(object):

    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        """
        The class of a vehicle able to execute the action.

        Must return a subclass of :py:class:`highway_env.vehicle.kinematics.Vehicle`.
        """
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-vehicle.

        Most of the action mechanics are actually implemented in vehicle.act(action), where
        vehicle is an instance of the specified :py:class:`highway_env.envs.common.action.ActionType.vehicle_class`.
        Must some pre-processing can be applied to the action based on the ActionType configurations.

        :param action: the action to execute
        """
        raise NotImplementedError

    def get_available_actions(self):
        """
        For discrete action space, return the list of available actions.
        """
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):

    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_RANGE = (-np.pi / 4, np.pi / 4)
    """Steering angle range: [-x, x], in rad."""

    def __init__(self,
                 env: 'AbstractEnv',
                 acceleration_range: Optional[Tuple[float, float]] = None,
                 steering_range: Optional[Tuple[float, float]] = None,
                 speed_range: Optional[Tuple[float, float]] = None,
                 longitudinal: bool = True,
                 lateral: bool = True,
                 dynamical: bool = False,
                 clip: bool = True,
                 **kwargs) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param acceleration_range: the range of acceleration values [m/s²]
        :param steering_range: the range of steering values [rad]
        :param speed_range: the range of reachable speeds [m/s]
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        :param dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        :param clip: clip action to the defined range
        """
        # print(steering_range)
        super().__init__(env)
        self.acceleration_range = acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        # print(self.steering_range)
        self.speed_range = speed_range
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError("Either longitudinal and/or lateral control must be enabled")
        self.dynamical = dynamical
        self.clip = clip
        self.size = 2 if self.lateral and self.longitudinal else 1
        self.last_action = np.zeros(self.size)
        self.usempc_controller = self.env.config['usempc_controller']
        # self.half_controller = self.env.config['half_controller']


    def space(self) -> spaces.Box:
        return spaces.Box(-1., 1., shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle if not self.dynamical else BicycleVehicle

    def act(self, action: np.ndarray) -> None:
        # print(1111111111)
        # 如果设置了 clip 标志，将 action 数值限制在 -1 到 1 之间
        if self.clip and not self.usempc_controller:
            action = np.clip(action, -1, 1)
            # print(222)
        # MDPVehicle.act(self.controlled_vehicle, action)
        # 如果设置了 speed_range，将控制车辆的最小速度和最大速度设置为给定的范围
        if self.speed_range:
            self.controlled_vehicle.MIN_SPEED, self.controlled_vehicle.MAX_SPEED = self.speed_range
        if self.longitudinal and self.lateral and not self.usempc_controller:
            self.controlled_vehicle.act({
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
                # "steering": 0,
            })
            # print(333)
        elif self.longitudinal and self.lateral and self.usempc_controller:
            self.controlled_vehicle.act({
                "acceleration": action[0],
                "steering": action[1],
            })
        elif self.longitudinal and not self.usempc_controller:
            self.controlled_vehicle.act({
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": 0,
            })
        elif self.lateral and not self.usempc_controller:
            self.controlled_vehicle.act({
                
                "acceleration": 0,
                "steering": utils.lmap(action[0], [-1, 1], self.steering_range),
                # "steering": utils.lmap(action[0], [-1, 1], self.steering_range),
            })

        self.last_action = action


class DiscreteAction(ContinuousAction):
    def __init__(self,
                 env: 'AbstractEnv',
                 acceleration_range: Optional[Tuple[float, float]] = None,
                 steering_range: Optional[Tuple[float, float]] = None,
                 longitudinal: bool = True,
                 lateral: bool = True,
                 dynamical: bool = False,
                 clip: bool = True,
                 actions_per_axis: int = 3,
                 **kwargs) -> None:
        super().__init__(env, acceleration_range=acceleration_range, steering_range=steering_range,
                         longitudinal=longitudinal, lateral=lateral, dynamical=dynamical, clip=clip)
        self.actions_per_axis = actions_per_axis

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(self.actions_per_axis**self.size)

    def act(self, action: int) -> None:
        cont_space = super().space()
        axes = np.linspace(cont_space.low, cont_space.high, self.actions_per_axis).T
        all_actions = list(itertools.product(*axes))
        super().act(all_actions[action])


class DiscreteMetaAction(ActionType):

    """
    An discrete action space of meta-actions: lane changes, and cruise control set-point.
    """

    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    """A mapping of action indexes to labels."""

    ACTIONS_LONGI = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    """A mapping of longitudinal action indexes to labels."""

    ACTIONS_LAT = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT'
    }
    """A mapping of lateral action indexes to labels."""

    def __init__(self,
             env: 'AbstractEnv',
             longitudinal: bool = True,
             lateral: bool = True,
             target_speeds: Optional[Vector] = None,
             **kwargs) -> None:
        """
        创建一个离散的动作空间，用于定义元动作（meta-actions）。

        :param env: 环境对象，类型为 AbstractEnv 的子类实例
        :param longitudinal: 是否包含纵向动作，默认为 True
        :param lateral: 是否包含横向动作，默认为 True
        :param target_speeds: 车辆能够跟踪的目标速度列表，默认为 None
        """

        # 初始化父类，并传入环境对象
        super().__init__(env)

        # 设置是否包含纵向和横向动作
        self.longitudinal = longitudinal
        self.lateral = lateral

        # 初始化目标速度数组，如果未提供则使用默认值
        self.target_speeds = np.array(target_speeds) if target_speeds is not None else MDPVehicle.DEFAULT_TARGET_SPEEDS

        # 根据是否包含纵向和横向动作选择可用的动作集合
        self.actions = self.ACTIONS_ALL if longitudinal and lateral \
            else self.ACTIONS_LONGI if longitudinal \
            else self.ACTIONS_LAT if lateral \
            else None

        # 如果没有选择任何动作类型，则抛出异常
        if self.actions is None:
            raise ValueError("必须至少包含纵向或横向动作之一")

        # 生成动作索引字典，方便通过动作名称查找索引
        self.actions_indexes = {v: k for k, v in self.actions.items()}


    def space(self) -> spaces.Space:
        # 返回一个一维的连续空间，范围在[-1, 1]之间
        return spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(LYZ_MDPVehicle, target_speeds=self.target_speeds)

    def act(self, action: Union[int, np.ndarray]) -> None:
        self.controlled_vehicle.act(action)

    def get_available_actions(self) -> List[int]:
        """
        获取当前可用的操作列表。
    
        在道路边界上不允许换道，在最大或最小速度时不允许改变速度。
    
        :return: 可用操作的列表
        """
    
        # 初始化可用操作列表，包含默认操作 'IDLE'
        actions = [self.actions_indexes['IDLE']]
        
        # 获取当前车辆所在道路网络
        network = self.controlled_vehicle.road.network
        
        # 遍历相邻车道，检查是否可以进行左右换道操作
        for l_index in network.side_lanes(self.controlled_vehicle.lane_index):
            if l_index[2] < self.controlled_vehicle.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_vehicle.position) \
                    and self.lateral:
                # 如果左侧车道可达且允许横向移动，则添加左换道操作
                actions.append(self.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.controlled_vehicle.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_vehicle.position) \
                    and self.lateral:
                # 如果右侧车道可达且允许横向移动，则添加右换道操作
                actions.append(self.actions_indexes['LANE_RIGHT'])
    
        # 检查是否可以加速
        if self.controlled_vehicle.speed_index < self.controlled_vehicle.target_speeds.size - 1 and self.longitudinal:
            # 如果当前速度不是最高速度且允许纵向移动，则添加加速操作
            actions.append(self.actions_indexes['FASTER'])
    
        # 检查是否可以减速
        if self.controlled_vehicle.speed_index > 0 and self.longitudinal:
            # 如果当前速度不是最低速度且允许纵向移动，则添加减速操作
            actions.append(self.actions_indexes['SLOWER'])
    
        return actions



class MultiAgentAction(ActionType):
    def __init__(self,
                 env: 'AbstractEnv',
                 action_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.action_config = action_config
        self.agents_action_types = []
        for vehicle in self.env.controlled_vehicles:
            action_type = action_factory(self.env, self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([action_type.space() for action_type in self.agents_action_types])

    @property
    def vehicle_class(self) -> Callable:
        return action_factory(self.env, self.action_config).vehicle_class

    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

    def get_available_actions(self):
        return itertools.product(*[action_type.get_available_actions() for action_type in self.agents_action_types])


class NoAction(ActionType):

    def space(self) -> spaces.Space:
        """One action that does nothing."""
        return spaces.Discrete(1)

    @property
    def vehicle_class(self) -> Callable:
        """The vehicle is an IDMVehicle, such that it executes IDM/MOBIL actions on its own."""
        return IDMVehicle

    def act(self, action: Action) -> None:
        """Does nothing, the IDMVehicle actions will be followed automatically."""
        pass


def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    if config["type"] == "Half_ContinuousAction":
        return Half_ContinuousLongitudinalAction(env, **config)
    if config["type"] == "DiscreteAction":
        return DiscreteAction(env, **config)
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    elif config["type"] == "NoAction":
        return NoAction(env, **config)
    else:
        raise ValueError("Unknown action type")
