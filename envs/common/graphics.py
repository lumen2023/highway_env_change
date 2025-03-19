import os
from typing import TYPE_CHECKING, Callable, List, Optional
import numpy as np
import pygame

from highway_env.envs.common.action import ActionType, DiscreteMetaAction, ContinuousAction
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action


class EnvViewer(object):

    """A viewer to render a highway driving environment."""

    SAVE_IMAGES = False
    agent_display = None

    def __init__(self, env: 'AbstractEnv', config: Optional[dict] = None) -> None:
        """
        初始化环境渲染器。

        参数：
        - env (`AbstractEnv`): 需要渲染的环境对象。
        - config (`Optional[dict]`): 可选的配置字典，如果未提供，则使用 `env.config`。
        """
        self.env = env
        self.config = config or env.config  # 使用传入的配置，否则使用环境默认配置
        self.offscreen = self.config["offscreen_rendering"]  # 是否启用离屏渲染（不打开窗口，仅获取图像）
        self.observer_vehicle = None  # 观察车辆（跟随的主车辆）
        self.agent_surface = None  # 代理（智能体）可视化的表面
        self.vehicle_trajectory = None  # 车辆轨迹（用于绘制历史轨迹）
        self.frame = 0  # 记录渲染的帧数
        self.directory = None  # 可用于存储渲染输出
        
        
        # 初始化 pygame
        pygame.init()
        pygame.display.set_caption("Highway-env")  # 设置窗口标题
        panel_size = (self.config["screen_width"], self.config["screen_height"])  # 设定窗口尺寸

        # 说明：
        # 在某些云计算环境或无图形界面的服务器上，我们可能不希望创建窗口。
        # 这里如果 `offscreen` 为 False，则创建一个窗口，否则不创建。
        if self.env.render_mode == "rgb_array":
            self.offscreen = True
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.config["screen_width"], self.config["screen_height"]])  # 创建显示窗口

        # 如果 `agent_display` 设为 True，则扩展显示区域
        if self.agent_display:
            self.extend_display()

        # 创建模拟表面 `sim_surface`，用于绘制环境
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get("scaling", self.sim_surface.INITIAL_SCALING)  # 获取缩放比例
        self.sim_surface.centering_position = self.config.get("centering_position",
                                                              self.sim_surface.INITIAL_CENTERING)  # 获取居中位置
        self.clock = pygame.time.Clock()  # 初始化时钟，用于控制帧率

        self.enabled = True  # 标记渲染是否启用

        # 检查是否在无图形界面的环境（如云计算服务器）运行
        # `SDL_VIDEODRIVER=dummy` 表示使用虚拟驱动，不进行实际渲染
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False  # 关闭渲染

    def set_agent_display(self, agent_display: Callable) -> None:
        """
        Set a display callback provided by an agent

        So that they can render their behaviour on a dedicated agent surface, or even on the simulation surface.

        :param agent_display: a callback provided by the agent to display on surfaces
        """
        if EnvViewer.agent_display is None:
            self.extend_display()
        EnvViewer.agent_display = agent_display

    def extend_display(self) -> None:
        if not self.offscreen:
            if self.config["screen_width"] > self.config["screen_height"]:
                self.screen = pygame.display.set_mode((self.config["screen_width"],
                                                       2 * self.config["screen_height"]))
            else:
                self.screen = pygame.display.set_mode((2 * self.config["screen_width"],
                                                       self.config["screen_height"]))
        self.agent_surface = pygame.Surface((self.config["screen_width"], self.config["screen_height"]))

    def set_agent_action_sequence(self, actions: List['Action']) -> None:
        """
        Set the sequence of actions chosen by the agent, so that it can be displayed

        :param actions: list of action, following the env's action space specification
        """
        if isinstance(self.env.action_type, DiscreteMetaAction):
            actions = [self.env.action_type.actions[a] for a in actions]
        if len(actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(actions,
                                                                          1 / self.env.config["policy_frequency"],
                                                                          1 / 3 / self.env.config["policy_frequency"],
                                                                          1 / self.env.config["simulation_frequency"])

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if EnvViewer.agent_display:
            EnvViewer.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1

    def get_image(self) -> np.ndarray:
        """
        返回渲染后的图像，以RGB数组形式表示。
    
        Gymnasium的通道约定是H x W x C。
        
        此函数根据当前的渲染配置获取图像数据，并确保图像数据的格式符合Gymnasium的要求。
        """
        # 根据配置决定使用哪个表面进行渲染
        surface = self.screen if self.config["render_agent"] and not self.offscreen else self.sim_surface
        
        # 使用pygame的surfarray模块获取表面的RGB数据，数据格式为W x H x C
        data = pygame.surfarray.array3d(surface)
        
        # 由于Gymnasium的通道约定是H x W x C，因此需要对数据进行转置
        return np.moveaxis(data, 0, 1)

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.observer_vehicle:
            return self.observer_vehicle.position
        elif self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()


class EventHandler(object):
    @classmethod
    def handle_event(cls, action_type: ActionType, event: pygame.event.EventType) -> None:
        """
        Map the pygame keyboard events to control decisions

        :param action_type: the ActionType that defines how the vehicle is controlled
        :param event: the pygame event
        """
        if isinstance(action_type, DiscreteMetaAction):
            cls.handle_discrete_action_event(action_type, event)
        elif action_type.__class__ == ContinuousAction:
            cls.handle_continuous_action_event(action_type, event)

    @classmethod
    def handle_discrete_action_event(cls, action_type: DiscreteMetaAction, event: pygame.event.EventType) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["FASTER"])
            if event.key == pygame.K_LEFT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["SLOWER"])
            if event.key == pygame.K_DOWN and action_type.lateral:
                action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            if event.key == pygame.K_UP:
                action_type.act(action_type.actions_indexes["LANE_LEFT"])

    @classmethod
    def handle_continuous_action_event(cls, action_type: ContinuousAction, event: pygame.event.EventType) -> None:
        action = action_type.last_action.copy()
        steering_index = action_type.space().shape[0] - 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0.7
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = -0.7
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = -0.7
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0.7
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = 0
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0
        action_type.act(action)


class ObservationGraphics(object):
    COLOR = (0, 0, 0)

    @classmethod
    def display(cls, obs, sim_surface):
        from highway_env.envs.common.observation import LidarObservation
        if isinstance(obs, LidarObservation):
            cls.display_grid(obs, sim_surface)

    @classmethod
    def display_grid(cls, lidar_observation, surface):
        psi = np.repeat(np.arange(-lidar_observation.angle/2,
                                  2 * np.pi - lidar_observation.angle/2,
                                  2 * np.pi / lidar_observation.grid.shape[0]), 2)
        psi = np.hstack((psi[1:], [psi[0]]))
        r = np.repeat(np.minimum(lidar_observation.grid[:, 0], lidar_observation.maximum_range), 2)
        points = [(surface.pos2pix(lidar_observation.origin[0] + r[i] * np.cos(psi[i]),
                                   lidar_observation.origin[1] + r[i] * np.sin(psi[i])))
                  for i in range(np.size(psi))]
        pygame.draw.lines(surface, ObservationGraphics.COLOR, True, points, 1)
