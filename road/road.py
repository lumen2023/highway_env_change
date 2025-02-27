import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

from highway_env.road.lane import LineType, StraightLane, AbstractLane, lane_from_config
from highway_env.vehicle.objects import Landmark

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class RoadNetwork(object):
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self):
        self.graph = {}
    
    # lane是双线对象,起点(x,y)
    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        在道路网络中添加一条车道，作为图中的边。
    
        :param _from: 车道起点的节点标识。
        :param _to: 车道终点的节点标识。
        :param AbstractLane lane: 车道的几何表示。
        """
        # 确保起点节点存在于图中，如果不存在则添加
        if _from not in self.graph:
            self.graph[_from] = {}
        # 确保终点节点存在于起点节点的邻接列表中，如果不存在则添加
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        # 将车道添加到起点和终点之间的连接中
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        根据给定的索引获取道路网络中对应的车道几何信息。

        :param index: 一个包含 (起始节点, 目的节点, 道路上的车道ID) 的元组。
        :return: 对应的车道几何信息。
        """
        # 将索引解包为起始节点、目的节点和车道ID
        _from, _to, _id = index
        # 如果车道ID为None且两节点之间仅有一条车道，则默认车道ID为0
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        # 返回对应起始节点、目的节点和车道ID的车道几何信息
        return self.graph[_from][_to][_id]


    def get_closest_lane_index(self, position: np.ndarray, heading: Optional[float] = None) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index: LaneIndex, route: Route = None, position: np.ndarray = None,
              np_random: np.random.RandomState = np.random) -> LaneIndex:
        """
        获取当前车道结束后应跟随的下一个车道的索引。

        - 如果有可用的计划且与当前车道匹配，则按照计划行驶。
        - 否则，随机选择下一个道路。
        - 如果下一个道路的车道数与当前道路相同，则保持在相同的车道上。
        - 否则，选择下一个道路中最接近的车道。
        :param current_index: 当前目标车道的索引。
        :param route: 如果有的话，规划的路线。
        :param position: 车辆的位置。
        :param np_random: 随机源。
        :return: 当前车道结束后应跟随的下一个车道的索引。
        """
        # 解码当前车道索引为组件
        _from, _to, _id = current_index
        next_to = next_id = None

        # 根据规划的路线选择下一个道路
        if route:
            if route[0][:2] == current_index[:2]:  # 刚刚完成了路线的第一步，删除它。
                route.pop(0)
            if route and route[0][0] == _to:  # 路线中的下一个道路从当前道路的终点开始。
                _, next_to, next_id = route[0]
            elif route:
                logger.warning("路线 {} 不是从当前道路 {} 的终点开始的。".format(route[0], current_index))

        # 计算当前投影（期望）位置
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)

        # 如果下一个路线未知
        if not next_to:
            # 选择与投影目标位置最接近的车道
            try:
                lanes_dists = [(next_to,
                                *self.next_lane_given_next_road(_from, _to, _id, next_to, next_id, projected_position))
                               for next_to in self.graph[_to].keys()]  # (next_to, next_id, distance)
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # 如果已知，按照它并获取最接近的车道
            next_id, _ = self.next_lane_given_next_road(_from, _to, _id, next_to, next_id, projected_position)

        # 返回下一个车道的索引
        return _to, next_to, next_id


    def next_lane_given_next_road(self, _from: str, _to: str, _id: int,
                                  next_to: str, next_id: int, position: np.ndarray) -> Tuple[int, float]:
        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            if next_id is None:
                next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,
                          key=lambda l: self.get_lane((_to, next_to, l)).distance(position))
        return next_id, self.get_lane((_to, next_to, next_id)).distance(position)

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        使用广度优先搜索查找从起点到目标的所有路径。
    
        :param start: 起始节点
        :param goal: 目标节点
        :return: 从起始节点到目标节点的所有路径列表
        """
        queue = [(start, [start])]  # 初始化队列，包含起始节点和当前路径
        while queue:
            (node, path) = queue.pop(0)  # 从队列中取出第一个元素
            if node not in self.graph:
                yield []  # 如果当前节点不在图中，生成一个空路径
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]  # 如果找到目标节点，生成当前路径加上目标节点
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))  # 将相邻节点及其路径加入队列


    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        使用广度优先搜索从起点到目标的最短路径。

        :param start: 起始节点
        :param goal: 目标节点
        :return: 从起始节点到目标节点的最短路径
        """
        # 使用 next 函数从 bfs_paths 生成器中获取最短路径，如果没有路径则返回空列表
        return next(self.bfs_paths(start, goal), [])


    def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [(lane_index[0], lane_index[1], i) for i in range(len(self.graph[lane_index[0]][lane_index[1]]))]

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
                :param lane_index: the index of a lane.
                :return: indexes of lanes next to a an input lane, to its right or left.
                """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(self, lane_index_1: LaneIndex, lane_index_2: LaneIndex, route: Route = None,
                          same_lane: bool = False, depth: int = 0) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
                or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                            for l1_to in self.graph.get(_to, {}).keys()])
        return False

    def lanes_list(self) -> List[AbstractLane]:
        return [lane for to in self.graph.values() for ids in to.values() for lane in ids]

    @staticmethod
    def straight_road_network(lanes: int = 4,
                              start: float = 0,
                              length: float = 10000,
                              angle: float = 0,
                              speed_limit: float = 30,
                              nodes_str: Optional[Tuple[str, str]] = None,
                              net: Optional['RoadNetwork'] = None) \
            -> 'RoadNetwork':
        net = net or RoadNetwork()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types, speed_limit=speed_limit))
        return net

    def position_heading_along_route(self, route: Route, longitudinal: float, lateral: float) \
            -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :return: position, heading
        """
        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]
        return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)

    def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
        _from = np_random.choice(list(self.graph.keys()))
        _to = np_random.choice(list(self.graph[_from].keys()))
        _id = np_random.randint(len(self.graph[_from][_to]))
        return _from, _to, _id

    @classmethod
    def from_config(cls, config: dict) -> None:
        net = cls()
        for _from, to_dict in config.items():
            net.graph[_from] = {}
            for _to, lanes_dict in to_dict.items():
                net.graph[_from][_to] = []
                for lane_dict in lanes_dict:
                    net.graph[_from][_to].append(
                        lane_from_config(lane_dict)
                    )
        return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(
                        lane.to_config()
                    )
        return graph_dict


class Road(object):

    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(self,
                 network: RoadNetwork = None,
                 vehicles: List['kinematics.Vehicle'] = None,
                 road_objects: List['objects.RoadObject'] = None,
                 np_random: np.random.RandomState = None,
                 record_history: bool = False) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(self, vehicle: 'kinematics.Vehicle', distance: float, count: Optional[int] = None,
                          see_behind: bool = True, sort: bool = True) -> object:
        vehicles = [v for v in self.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]

        if sort:
            vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i+1:]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)

    def neighbour_vehicles(self, vehicle: 'kinematics.Vehicle', lane_index: LaneIndex = None) \
            -> Tuple[Optional['kinematics.Vehicle'], Optional['kinematics.Vehicle']]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()
