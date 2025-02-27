import casadi as ca
import casadi.tools as ca_tools
import math
import time
import numpy as np


def shift_movement(T, t0, x0, u, f):
    # 小车运动到下一个位置
    f_value = f(x0, u[0, :])
    state_next_ = x0 + T*f_value.T
    # 时间增加
    t_ = t0 + T
    # 准备下一个估计的最优控制，因为u[0:, :]已经采纳，我们就简单地把后面的结果提前
    u_next_ = ca.vertcat(u[1:, :], u[-1, :])
    return t_, state_next_, u_next_


if __name__ == '__main__':
    start_time = time.time()
    # MPC 参数
    T = 0.05  # 采样时间 [s]
    N = 20  # 预测时域

    "定义约束"
    Y_max, Y_min = 10, 2
    v_max, v_min = 30, 0
    a_max, a_min = 5, -5
    jerk_max, jerk_min = 1, -1
    delta_max, delta_min = np.pi / 4, -np.pi / 4
    delta_dot_max, delta_dot_min = np.pi / 36, -np.pi / 36
    lf, lr = 2.5, 2.5

    "定义状态和控制符号"
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    v = ca.SX.sym('v')
    p = ca.SX.sym('p')

    states = ca.vertcat(x, y, v, p)
    n_states = states.size()[0]

    a = ca.SX.sym('a')
    delta = ca.SX.sym('delta')
    controls = ca.vertcat(a, delta)
    n_controls = controls.size()[0]

    "定义系统动力学"
    bt = ca.arctan(lr/(lf+lr)*ca.tan(delta))
    rhs = ca.horzcat(v * ca.cos(p + bt))
    rhs = ca.horzcat(rhs, v * ca.sin(p + bt))
    rhs = ca.horzcat(rhs, a)
    rhs = ca.horzcat(rhs, v/lr * ca.sin(bt))

    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
    U = ca.SX.sym('U', N, n_controls)
    X = ca.SX.sym('X', N+1, n_states)
    # P = ca.SX.sym('P', n_states+3)
    P = ca.SX.sym('P', n_states + 2)

    "定义代价函数权重"
    Q1, Q2, Q3 = 100, 1000, 1000
    R = np.array([[1, 0.0], [0.0, 1]])
    R1, R2 = 1, 1
    M1, M2 = 100, 1

    "定义初始化条件"
    X[0, :] = P[:4]

    # 定义预测时域内的状态关系
    for i in range(N):
        f_value = f(X[i, :], U[i, :])
        X[i+1, :] = X[i, :] + f_value*T

    # 代价函数
    obj = 0
    for i in range(N):
        obj = obj + Q1 * (X[i + 1, 1] - P[4]) ** 2
        obj = obj + Q2 * (X[i + 1, 2] - P[5]) ** 2
        # obj = obj + Q3 * (X[i + 1, 3] - P[6]) ** 2

        # obj = obj + R1 * (U[i, 0]) ** 2
        # obj = obj + R2 * (U[i, 1]) ** 2
        obj = obj + ca.mtimes([U[i, :], R, U[i, :].T])
        # 添加加速度变化率和前轮转角变化率的代价
        if i > 0 and i < N-1:
            obj = obj + M1 * (U[i, 0] - U[i-1, 0]) ** 2
            obj = obj + M2 * (U[i, 1] - U[i-1, 1]) ** 2

    # 约束
    g = []
    for i in range(N+1):
        g.append(X[i, 1])
        g.append(X[i, 2])
        if i > 0 and i < N:
            g.append(U[i, 0]-U[i-1, 0])
            g.append(U[i, 1]-U[i-1, 1])

    # NLP 问题
    nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100,
                    'ipopt.print_level':0,
                    'print_time':0,
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = []
    ubg = []

    for j in range(N+1):
        lbg.append(Y_min)
        ubg.append(Y_max)
        lbg.append(v_min)
        ubg.append(v_max)
        if j > 0 and j < N:
            lbg.append(jerk_min*T)
            ubg.append(jerk_max*T)
            lbg.append(delta_dot_min*T)
            ubg.append(delta_dot_max*T)

    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(a_min)
        ubx.append(a_max)

    for _ in range(N):
        lbx.append(delta_min)
        ubx.append(delta_max)

    # 仿真
    t0 = 0.0
    "初始位置：[x,y,v,phi]"
    x0 = np.array([230, 8, 15, 0]).reshape(-1, 1)# initial state
    # xs = np.array([4, 25]).reshape(-1, 1) # final state
    u0 = np.array([0, 0]*N).reshape(-1, 2)# np.ones((N, 2)) # controls
    xh = [x0]  # contains for the history of the state
    uh = []
    th = [t0]  # for the time
    sim_time = 5
    # start MPC
    mpciter = 0

    # 读取.npy文件
    # data = np.load('episode_merge_data.npy')
    # data = np.load('episode_merge_data(1).npy')
    # data = np.load('episode_merge_data(2).npy')
    data = np.load('episode_merge_data(3).npy')
    # 提取第三列和第四列
    # new_data = data[:, [2, 3, 4]]
    new_data = data[:, [2, 3]]

    while(mpciter - sim_time/T<0.0 ):
        # 设置参数
        idx = int(mpciter / (0.1/T))

        "xs为上层RL输出的参考姿态[参考车速，参考横向位置]，每0.1s发出一次"
        "这一块需要改写代码！！！！！！！！！！"
        xs = new_data[idx, :].reshape(-1, 1)
        c_p = np.concatenate((x0, xs))

        init_control = ca.reshape(u0, -1, 1)

        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        u_sol = ca.reshape(res['x'],  N, n_controls)
        u_attach = ca.reshape(u_sol[0, :], -1, 1)   # 取u_sol的第一个元素
        uh.append(u_attach.full())   # 保存动作轨迹
        th.append(t0)   # 保存时间轨迹

        "t0: 大概率从highway环境中读取"
        "x0: 大概率从highway环境中读取"
        "u0: "

        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)
        "输入"
        "T: 仿真时间"
        "t0：系统时间"
        "x0：自车状态"
        "u_sol：动作序列"
        "输出"
        "t0: 下一刻仿真时间"
        "x0：下一刻自车状态"
        "u0：下一刻动作序列"

        x0 = ca.reshape(x0, -1, 1)
        xh.append(x0.full())
        mpciter = mpciter + 1
    print(time.time() - start_time)

    import matplotlib.pyplot as plt
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

    # Create a 2x2 grid of subplots
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))

    # Plot y vs time
    ax[0, 0].plot(time_values, y_values)
    ax[0, 0].plot(time_values1, y_values1)
    ax[0, 0].set_xlabel('T(s)')
    ax[0, 0].set_ylabel(r'$y$')
    ax[0, 0].grid(True)

    # Plot vx vs time
    ax[0, 1].plot(time_values, v_values)
    ax[0, 1].plot(time_values1, v_values1)
    ax[0, 1].set_xlabel('T(s)')
    ax[0, 1].set_ylabel(r'$v$')
    ax[0, 1].grid(True)

    # Plot phi vs time
    ax[0, 2].plot(time_values,  phi_values)
    ax[0, 2].plot(time_values1, phi_values1)
    ax[0, 2].set_xlabel('T(s)')
    ax[0, 2].set_ylabel(r'$phi$')
    ax[0, 2].grid(True)

    # Plot a vs time
    ax[1, 0].plot(time_values[:-1], a_values)
    ax[1, 0].plot(time_values1, a_values1)
    ax[1, 0].set_xlabel('T(s)')
    ax[1, 0].set_ylabel(r'$a$')
    ax[1, 0].grid(True)

    # Plot df vs time
    ax[1, 1].plot(time_values[:-1], delta_values)
    ax[1, 1].plot(time_values1, delta_values1)
    ax[1, 1].set_xlabel('T(s)')
    ax[1, 1].set_ylabel(r'$\delta_{f}$')
    ax[1, 1].grid(True)

    # Plot df vs time
    ax[1, 2].plot(x_values, y_values)
    ax[1, 2].plot(x_values1, y_values1)
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel(r'y')
    ax[1, 2].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

