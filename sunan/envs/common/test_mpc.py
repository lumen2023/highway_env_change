import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from highway_env import utils
# env = gym.make('merge-v0', render_mode='human')
env = gym.make('merge-v0')
env.config["main_lanes_count"] = 1
env.config["vehicles_count"] = 7
obs, info = env.reset()
done = truncated = False
x = []
y = []
i = 0
a = [1,0.05]
x.append(info["position"][0])
y.append(info["position"][1])
while not (done or truncated or i > 100):
    obs, reward, done, truncated, info = env.step(a)
    x.append(info["position"][0])
    y.append(info["position"][1])
    i += 1
    print(info["position"][0], info["position"][1], info["speed"])


# 定义系统动态方程
def dynamics(dt, s, u):
    x, y, v, heading = s
    a, steering_angle = u

    # 车辆参数
    l_f = 2.5  # 前轮到车辆质心的距离
    l_r = 2.5  # 后轮到车辆质心的距离

    # 计算横摆角
    slip_angle = np.arctan(1 / 2 * np.tan(steering_angle))
    v_next = v + a * dt
    v_next = np.clip(v_next, -40, 40)
    # 计算下一个时刻的状态
    x_next = x + v_next * np.cos(heading + slip_angle) * dt
    y_next = y + v_next * np.sin(heading + slip_angle) * dt
    heading_next = heading + v_next * np.sin(slip_angle) / l_r * dt
    s_next = np.array([x_next, y_next, v_next, heading_next])
    return s_next

# 模拟第二个轨迹
dt = 0.1
num_steps = 100
initial_state = np.array([230, 8, 15, 0])  # 初始状态 [x, y, v, heading]
trajectory_2 = [initial_state]
ACTION = np.clip(a, -1, 1)
ACTION1 = [utils.lmap(ACTION[0], [-1, 1], [-4, 4]), utils.lmap(ACTION[1], [-1, 1], [-0.1, 0.1])]
for i in range(num_steps):
    next_state = dynamics(dt, trajectory_2[-1], ACTION1)
    trajectory_2.append(next_state)

# 提取第二个轨迹中的 x 和 y 坐标
x_trajectory_2 = [state[0] for state in trajectory_2]
y_trajectory_2 = [state[1] for state in trajectory_2]

# 绘制轨迹
plt.figure(figsize=(8, 6))
plt.plot(x_trajectory_2, y_trajectory_2, label='Combined Trajectory', marker='o')
plt.plot(x, y, label='x', marker='o')

plt.title('Vehicle Trajectory Comparison')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

