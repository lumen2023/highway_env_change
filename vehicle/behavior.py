from typing import Tuple, Union

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

# IDMVehicle 是基础类，提供了车辆的纵向加速度控制（IDM模型）和横向变道决策（MOBIL模型）。
class IDVehicle(ControlledVehicle):
    """
    使用纵向和横向决策策略的车辆。

    - 纵向：IDM模型根据前车的距离和速度计算加速度。
    - 横向：MOBIL模型通过最大化周围车辆的加速度来决定何时变道。
    """

    # 纵向策略参数
    ACC_MAX = 6.0  # [m/s2]
    """最大加速度。"""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """期望的最大加速度。"""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """期望的最大减速度。"""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """期望的前车拥堵距离。"""

    TIME_WANTED = 1.5  # [s]
    """期望的前车时间间隔。"""

    DELTA = 4.0  # []
    """速度项的指数。"""

    DELTA_RANGE = [3.5, 4.5]
    """随机选择的delta范围。"""

    # 横向策略参数
    POLITENESS = 0.  # 在 [0, 1] 范围内
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        # 随机化车辆的delta值
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        从现有车辆创建一个新车辆。

        复制车辆动力学和目标动力学，其他属性使用默认值。

        :param vehicle: 一辆车辆
        :return: 相同动态状态的新车辆
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        """
        执行动作。

        目前不支持外部动作，因为车辆基于IDM和MOBIL模型自行决策加速和变道。

        :param action: 动作
        """
        if self.crashed:
            return
        action = {}
        # 横向：MOBIL模型
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # 纵向：IDM模型
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        # 当变道时，检查当前车道和目标车道
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            action['acceleration'] = min(action['acceleration'], target_idm_acceleration)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)  # 跳过 ControlledVehicle.act()，否则命令会被覆盖。

    def step(self, dt: float):
        """
        进行一步仿真。

        增加用于决策策略的计时器，并进行车辆动力学的更新。

        :param dt: 时间步长
        """
        self.timer += dt
        super().step(dt)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        使用智能驾驶模型（IDM）计算加速度命令。

        加速度选择的目标是：
        - 达到目标速度；
        - 与前车保持最小安全距离（以及安全时间）。

        :param ego_vehicle: 被控制的车辆（不一定是IDM车辆）
        :param front_vehicle: 前车
        :param rear_vehicle: 后车
        :return: 该车辆的加速度命令 [m/s2]
        """
        # 如果ego_vehicle为空或不是Vehicle实例，则不执行加速度计算，返回0
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        # 获取ego_vehicle的目标速度，如果不存在，则默认为0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        # 如果ego_vehicle所在车道存在且有速度限制，则将目标速度限制在速度限制范围内
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(ego_target_speed, 0, ego_vehicle.lane.speed_limit)
        # 计算加速度，考虑当前车速与目标速度的差异
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(
            max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)), self.DELTA))
        # # 如果存在前车，则考虑与前车的距离，计算加速度时避免碰撞
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        acceleration += 2
        # 返回计算得到的加速度
        return acceleration


    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        计算车辆与前车之间的期望距离。

        :param ego_vehicle: 被控制的车辆
        :param front_vehicle: 前车
        :param projected: 是否将二维速度投影到一维空间
        :return: 两车之间的期望距离 [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        """
        决定何时变道。

        基于：
        - 频率；
        - 目标车道的接近程度；
        - MOBIL模型。
        """
        # 如果当前正在变道
        if self.lane_index != self.target_lane_index:
            # 如果我们在正确的路线但错误的车道上：如果其他车辆也在变道到同一车道，则中止
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # 否则，以给定的频率
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # 决定是否进行变道
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # 候选车道是否足够接近？
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # 仅在车辆移动时进行变道
            if np.abs(self.speed) < 1:
                continue
            # MOBIL模型是否推荐变道？
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL车道变换模型：通过变道最小化总体制动

        车辆应仅在以下情况下变道：
        - 变道后（和/或跟随车辆）可以更快地加速；
        - 变道不会对新跟车施加不安全的制动。

        :param lane_index: 候选变道车道
        :return: 是否应执行变道
        """
        # 新的跟车车辆的制动是否不安全？
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # 我是否有特定车道的计划路线且能安全进入？
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # 错误方向
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # 需要不安全的制动
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # 变道对我和/或我的跟车车辆是否有加速优势？
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # 一切清楚，可以变道！
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        如果在错误车道上停下，尝试进行倒车。

        :param acceleration: IDM计算的期望加速度
        :return: 建议用于脱困的加速度
        """
        stopped_speed = 5
        safe_distance = 200
        # 车辆是否在错误车道上停止？
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # 检查后方是否有足够的空隙
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # 倒车
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

class IDMVehicle(ControlledVehicle):
    """
    使用纵向和横向决策策略的车辆。

    - 纵向：IDM模型根据前车的距离和速度计算加速度。
    - 横向：MOBIL模型通过最大化周围车辆的加速度来决定何时变道。
    """

    # 纵向策略参数
    ACC_MAX = 6.0  # [m/s2]
    """最大加速度。"""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """期望的最大加速度。"""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """期望的最大减速度。"""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """期望的前车拥堵距离。"""

    TIME_WANTED = 1.5  # [s]
    """期望的前车时间间隔。"""

    DELTA = 4.0  # []
    """速度项的指数。"""

    DELTA_RANGE = [3.5, 4.5]
    """随机选择的delta范围。"""

    # 横向策略参数
    POLITENESS = 0.  # 在 [0, 1] 范围内
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        # 随机化车辆的delta值
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        从现有车辆创建一个新车辆。

        复制车辆动力学和目标动力学，其他属性使用默认值。

        :param vehicle: 一辆车辆
        :return: 相同动态状态的新车辆
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        """
        执行动作。

        目前不支持外部动作，因为车辆基于IDM和MOBIL模型自行决策加速和变道。

        :param action: 动作
        """
        if self.crashed:
            return
        action = {}
        # 横向：MOBIL模型
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # 纵向：IDM模型
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        # 当变道时，检查当前车道和目标车道
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            action['acceleration'] = min(action['acceleration'], target_idm_acceleration)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)  # 跳过 ControlledVehicle.act()，否则命令会被覆盖。

    def step(self, dt: float):
        """
        进行一步仿真。

        增加用于决策策略的计时器，并进行车辆动力学的更新。

        :param dt: 时间步长
        """
        self.timer += dt
        super().step(dt)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        使用智能驾驶模型（IDM）计算加速度命令。

        加速度选择的目标是：
        - 达到目标速度；
        - 与前车保持最小安全距离（以及安全时间）。

        :param ego_vehicle: 被控制的车辆（不一定是IDM车辆）
        :param front_vehicle: 前车
        :param rear_vehicle: 后车
        :return: 该车辆的加速度命令 [m/s2]
        """
        # 如果ego_vehicle为空或不是Vehicle实例，则不执行加速度计算，返回0
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        # 获取ego_vehicle的目标速度，如果不存在，则默认为0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        # 如果ego_vehicle所在车道存在且有速度限制，则将目标速度限制在速度限制范围内
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(ego_target_speed, 0, ego_vehicle.lane.speed_limit)
        # 计算加速度，考虑当前车速与目标速度的差异
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(
            max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)), self.DELTA))

        # 如果存在前车，则考虑与前车的距离，计算加速度时避免碰撞
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        # 返回计算得到的加速度
        return acceleration


    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        计算车辆与前车之间的期望距离。

        :param ego_vehicle: 被控制的车辆
        :param front_vehicle: 前车
        :param projected: 是否将二维速度投影到一维空间
        :return: 两车之间的期望距离 [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        """
        决定何时变道。

        基于：
        - 频率；
        - 目标车道的接近程度；
        - MOBIL模型。
        """
        # 如果当前正在变道
        if self.lane_index != self.target_lane_index:
            # 如果我们在正确的路线但错误的车道上：如果其他车辆也在变道到同一车道，则中止
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # 否则，以给定的频率
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # 决定是否进行变道
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # 候选车道是否足够接近？
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # 仅在车辆移动时进行变道
            if np.abs(self.speed) < 1:
                continue
            # MOBIL模型是否推荐变道？
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL车道变换模型：通过变道最小化总体制动

        车辆应仅在以下情况下变道：
        - 变道后（和/或跟随车辆）可以更快地加速；
        - 变道不会对新跟车施加不安全的制动。

        :param lane_index: 候选变道车道
        :return: 是否应执行变道
        """
        # 新的跟车车辆的制动是否不安全？
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # 我是否有特定车道的计划路线且能安全进入？
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # 错误方向
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # 需要不安全的制动
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # 变道对我和/或我的跟车车辆是否有加速优势？
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # 一切清楚，可以变道！
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        如果在错误车道上停下，尝试进行倒车。

        :param acceleration: IDM计算的期望加速度
        :return: 建议用于脱困的加速度
        """
        stopped_speed = 5
        safe_distance = 200
        # 车辆是否在错误车道上停止？
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # 检查后方是否有足够的空隙
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # 倒车
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

# LinearVehicle 继承了 IDMVehicle，并引入了线性控制器，使得纵向和横向控制均与特定参数线性相关，适合进行模型训练和参数回归。

class LinearVehicle(IDMVehicle):
    """纵向和横向控制器均与参数线性相关的车辆。"""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5 * np.array(ACCELERATION_PARAMETERS), 1.5 * np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.5

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 data: dict = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer)
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: Union[dict, str] = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua * (self.ACCELERATION_RANGE[1] -
                                                                          self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub * (self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        使用线性模型计算加速度命令。

        加速度选择的目标是：
        - 达到目标速度；
        - 达到前车（或后车）速度，如果前车速度较低（或后车速度较高）；
        - 与前车保持最小安全距离。

        :param ego_vehicle: 被控制的车辆（不一定是线性车辆）
        :param front_vehicle: 前车
        :param rear_vehicle: 后车
        :return: 该车辆的加速度命令 [m/s2]
        """
        return float(np.dot(self.ACCELERATION_PARAMETERS,
                            self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle)))

    def acceleration_features(self, ego_vehicle: ControlledVehicle,
                              front_vehicle: Vehicle = None,
                              rear_vehicle: Vehicle = None) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_speed - ego_vehicle.speed
            d_safe = self.DISTANCE_WANTED + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        线性控制器，相对于参数。

        重写非线性控制器 ControlledVehicle.steering_control()

        :param target_lane_index: 要跟随的车道索引
        :return: 转向角命令 [rad]
        """
        return float(np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index)))

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        用于跟随车道的特征集合。

        :param target_lane_index: 要跟随的车道索引
        :return: 特征数组
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([utils.wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / utils.not_zero(self.speed),
                             -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2)])
        return features

    def longitudinal_structure(self):
        # 标称动力学：积分速度
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        # 目标速度动力学
        phi0 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ])
        # 前车速度控制
        phi1 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 0]
        ])
        # 前车位置控制
        phi2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 1, -self.TIME_WANTED, 0],
            [0, 0, 0, 0]
        ])
        # 禁用速度控制
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # 禁用前车位置控制
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([
            [0, 1],
            [0, 0]
        ])
        phi0 = np.array([
            [0, 0],
            [0, -1]
        ])
        phi1 = np.array([
            [0, 0],
            [-1, 0]
        ])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """存储特征和输出以用于参数回归。"""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)

# 激进的驾驶风格
class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]

# 保守的驾驶风格
class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]
