import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def initialize_plot():
    """初始化图形，并返回绘图元素"""
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots(figsize=(20, 2))
    lines = []
    for y_line in [2, 6, 10]:
        line = ax.axhline(y_line, color='gray', linestyle='--')
        lines.append(line)
    return fig, ax


class Risk_field:
    def rotate_point(self, x, y, theta):
        """ 使用旋转矩阵旋转点(x, y)角度theta """
        return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

    def get_vehicle_corners(self, x_center, y_center, length, width, theta):
        """ 计算车辆四个角的全局坐标 """
        half_length = length / 2
        half_width = width / 2
        corners_local = [(-half_length, -half_width),
                         (-half_length, half_width),
                         (half_length, half_width),
                         (half_length, -half_width)]

        corners_global = []
        for x, y in corners_local:
            x_rot, y_rot = self.rotate_point(x, y, theta)
            x_glob = x_rot + x_center
            y_glob = y_rot + y_center
            corners_global.append((x_glob, y_glob))

        return corners_global


    def update_plot(self, fig, ax, X, Y, U_field, vehicles_obs, done):
        """更新图形内容"""
        # 清除当前轴的所有内容，除了颜色条和水平线
        ax.cla()  # Clear the current axes
        contourf = ax.contourf(X, Y, U_field, levels=50, cmap='rainbow')  # 重新绘制等高线图
        colorbar = fig.colorbar(contourf, label='Risk value')
        ax.invert_yaxis()
        # 重新添加水平线
        for y_line in [2, 6, 10]:
            ax.axhline(y_line, color='gray', linestyle='--')

        x_start = vehicles_obs[0, 0]
        # 更新车辆位置和速度注释
        for vehicle in vehicles_obs:
            x_center, y_center, speed, theta = vehicle
            if x_start-97.5 <= x_center <= x_start+97.5:
                corners = self.get_vehicle_corners(x_center, y_center, 5, 2, theta)
                polygon = Polygon(corners, closed=True, edgecolor='black', facecolor='none')
                ax.add_patch(polygon)
                ax.annotate(f"Speed: {speed:.2f} m/s", (x_center, y_center), textcoords="offset points", xytext=(0, 10),
                             ha='center', fontsize=8)
                ax.plot(x_center, y_center, 'bo')

        plt.draw()
        plt.pause(0.01)  # 稍微暂停以便更新显示
        colorbar.remove()
        # 如果完成，则关闭图形
        if done:
            plt.close('all')

