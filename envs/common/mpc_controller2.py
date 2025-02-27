import casadi as ca
import casadi.tools as ca_tools
import math
import time
import numpy as np
import matplotlib.pyplot as plt

def polt_result(th, xh, data):
    time_values = th
    x_values = [state[0] for state in xh]
    y_values = [state[1] for state in xh]
    v_values = [state[2] for state in xh]
    phi_values = [state[3] for state in xh]

    # Extract ax and df values from uh
    a_values = [control[0] for control in uh]
    delta_values = [control[1] for control in uh]

    new_data = data[:, [0, 1, 2, 3, 4, 5 ,6]]
    time_values1 = new_data[:, 0]
    x_values1 = new_data[:, 1]
    y_values1 = new_data[:, 2]
    v_values1 = new_data[:, 3]
    phi_values1 = new_data[:, 4]
    a_values1 = new_data[:, 5]
    delta_values1 = new_data[:, 6]

    time_points = np.arange(0, 5.1, 0.1)
    function_values = [quadratic_function(t) for t in time_points]
    function_values = np.array(function_values)

    # Create a 2x2 grid of subplots
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))

    # Plot y vs time
    ax[0, 0].plot(time_values, y_values)
    # ax[0, 0].plot(time_values1, y_values1)
    ax[0, 0].plot(time_points, function_values[:, 0])

    ax[0, 0].set_xlabel('T(s)')
    ax[0, 0].set_ylabel(r'$y$')
    ax[0, 0].grid(True)

    # Plot vx vs time
    ax[0, 1].plot(time_values, v_values)
    # ax[0, 1].plot(time_values1, v_values1)
    ax[0, 1].plot(time_points, function_values[:, 1])
    ax[0, 1].set_xlabel('T(s)')
    ax[0, 1].set_ylabel(r'$v$')
    ax[0, 1].grid(True)

    # Plot phi vs time
    ax[0, 2].plot(time_values,  phi_values)
    # ax[0, 2].plot(time_values1, phi_values1)
    ax[0, 2].set_xlabel('T(s)')
    ax[0, 2].set_ylabel(r'$phi$')
    ax[0, 2].grid(True)

    # Plot a vs time
    ax[1, 0].plot(time_values[:-1], a_values)
    # ax[1, 0].plot(time_values1, a_values1)
    ax[1, 0].set_xlabel('T(s)')
    ax[1, 0].set_ylabel(r'$a$')
    ax[1, 0].grid(True)

    # Plot df vs time
    ax[1, 1].plot(time_values[:-1], delta_values)
    # ax[1, 1].plot(time_values1, delta_values1)
    ax[1, 1].set_xlabel('T(s)')
    ax[1, 1].set_ylabel(r'$\delta_{f}$')
    ax[1, 1].grid(True)

    # Plot df vs time
    ax[1, 2].plot(x_values, y_values)
    # ax[1, 2].plot(x_values1, y_values1)
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel(r'y')
    ax[1, 2].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def quadratic_function(t):
    a1 = (25 - 15) / 5 ** 2  # Coefficient for x^2
    b1 = 0  # Coefficient for x
    c1 = 15  # Constant term
    v_ref = a1 * t ** 2 + b1 * t + c1
    a2 = -1 / 5  # Coefficient for x^2
    b2 = 0  # Coefficient for x
    c2 = 8  # Constant term
    y_ref = a2 * t**2 + b2 * t + c2
    return np.array([y_ref, v_ref]).reshape(-1,1)

def shift_movement(T, t0, x0, u, f):
    # 小车运动到下一个位置
    f_value = f(x0, u[0, :])
    state_next_ = x0 + T * f_value.T
    # 时间增加
    t_ = t0 + T
    # 准备下一个估计的最优控制，因为u[0:, :]已经采纳，我们就简单地把后面的结果提前
    u_next_ = ca.vertcat(u[1:, :], u[-1, :])
    return t_, state_next_, u_next_

# Y_max=9, Y_min=3 匝道合流
class MPC:
    def __init__(self, T=0.2, N=20, Y_max=14, Y_min=-2, v_max=35, v_min=15,
                 a_max=4, a_min=-4, jerk_max=1, jerk_min=-1,
                 delta_max=0.1, delta_min=-0.1, delta_dot_max=np.pi / 36, delta_dot_min=-np.pi / 36,
                 lf=2.5, lr=2.5):
        self.T = T
        self.N = N
        self.Y_max, self.Y_min = Y_max, Y_min
        self.v_max, self.v_min = v_max, v_min
        self.a_max, self.a_min = a_max, a_min
        self.jerk_max, self.jerk_min = jerk_max, jerk_min
        self.delta_max, self.delta_min = delta_max, delta_min
        self.delta_dot_max, self.delta_dot_min = delta_dot_max, delta_dot_min
        self.lf, self.lr = lf, lr
        self.f = self.setup_model1()
        self.lbg, self.lbx, self.ubg, self.ubx, self.nlp_prob, self.opts_setting = self.setup_model()
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts_setting)

    def setup_model1(self):
        "定义状态和控制符号"
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        p = ca.SX.sym('p')
        states = ca.vertcat(x, y, v, p)
        self.n_states = states.size()[0]
        a = ca.SX.sym('a')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(a, delta)
        self.n_controls = controls.size()[0]
        "定义系统动力学"
        bt = ca.arctan(self.lr / (self.lf + self.lr) * ca.tan(delta))
        rhs = ca.horzcat(v * ca.cos(p + bt))
        rhs = ca.horzcat(rhs, v * ca.sin(p + bt))
        rhs = ca.horzcat(rhs, a)
        rhs = ca.horzcat(rhs, v / self.lr * ca.sin(bt))
        f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
        return f

    def setup_model(self):
        U = ca.SX.sym('U', self.N, self.n_controls)
        X = ca.SX.sym('X', self.N+1, self.n_states)
        P = ca.SX.sym('P', self.n_states + 2)

        "定义代价函数权重"
        Q1, Q2= 1, 1
        M1, M2 = 1, 50

        "定义初始化条件"
        X[0, :] = P[:4]

        # 定义预测时域内的状态关系
        for i in range(self.N):
            f_value = self.f(X[i, :], U[i, :])
            X[i+1, :] = X[i, :] + f_value*self.T

        # 代价函数
        obj = 0
        for i in range(self.N):
            obj = obj + Q1 * (X[i + 1, 1] - P[4]) ** 2 # 跟踪y_ref
            obj = obj + Q2 * (X[i + 1, 2] - P[5]) ** 2  # 跟踪v_ref
            if i > 0:
                obj = obj + M1 * (U[i, 0] - U[i-1, 0]) ** 2   # 加速度a
                obj = obj + M2 * ((U[i, 1] - U[i - 1, 1])) ** 2  # 前轮转角delta

        # 约束
        g = []
        for i in range(self.N + 1):
            g.append(X[i, 1])
            g.append(X[i, 2])
            if i > 0 and i < self.N:
                g.append(U[i, 0] - U[i - 1, 0])
                g.append(U[i, 1] - U[i - 1, 1])

        # NLP 问题
        nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P, 'g': ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter': 100,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6,
                        }

        lbg = []
        ubg = []
        for j in range(self.N + 1):
            lbg.append(self.Y_min)
            ubg.append(self.Y_max)
            lbg.append(self.v_min)
            ubg.append(self.v_max)
            if j > 0 and j < self.N:
                lbg.append(self.jerk_min)
                ubg.append(self.jerk_max)
                lbg.append(self.delta_dot_min)
                ubg.append(self.delta_dot_max)

        lbx = []
        ubx = []
        for _ in range(self.N):
            lbx.append(self.a_min)
            ubx.append(self.a_max)

        for _ in range(self.N):
            lbx.append(self.delta_min)
            ubx.append(self.delta_max)

        return lbg, lbx, ubg, ubx, nlp_prob, opts_setting

    def sovler_mpc(self, u0, c_p):
        res = self.solver(x0=u0, p=c_p, lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx)
        u_sol = ca.reshape(res['x'], self.N, self.n_controls)
        u_attach = ca.reshape(u_sol[0, :], -1, 1)
        return u_sol, u_attach, self.f



if __name__ == '__main__':
    start_time = time.time()
    # 仿真
    t0 = 0.0
    T = 0.1
    N = 20
    sim_time = 10
    mpciter = 0
    interval = 1
    plot = True
    "初始位置：[x,y,v,phi]"
    x0 = np.array([230, 8, 15, 0]).reshape(-1, 1)# initial state
    u0 = np.array([0, 0]*N).reshape(-1, 2)# np.ones((N, 2)) # controls
    xh = [x0]
    uh = []
    th = [t0]
    data = np.load('episode_merge_data(3).npy')
    new_data = data[:, [2, 3]]
    mpc=MPC()

    while(mpciter - sim_time/T<0.0):
        # 设置参数
        if mpciter % interval ==0:
            xs = quadratic_function(t0)
        # idx = int(mpciter / (0.1 / T))
        # xs = new_data[idx, :].reshape(-1, 1)
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        u_sol, u_attach, f = mpc.sovler_mpc(init_control, c_p)
        uh.append(u_attach.full())  # 保存动作轨迹
        th.append(t0)  # 保存时间轨迹
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)
        x0 = ca.reshape(x0, -1, 1)
        xh.append(x0.full())
        mpciter = mpciter + 1
    print(time.time() - start_time)
    if plot:
        polt_result(th, xh, data)

    # # 生成时间序列
    # time_points = np.arange(0, 5.1, 0.1)
    #
    # # 计算二次函数在每个时间点的值
    # function_values = [quadratic_function(t) for t in time_points]
    # function_values = np.array(function_values)
    # plt.plot(time_points, function_values[:,0])
    # plt.plot(time_points, function_values[:,1])
    # plt.show()
