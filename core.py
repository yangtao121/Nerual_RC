import RC
import numpy as np
import scipy.signal as ss
from Nerve_Core import Network as Center
import matplotlib.pyplot as plt
import tensorflow as tf

# RC网络参数的设定
r0 = 1
r1 = 1
r2 = 1
c1 = 1
c2 = 10
params = [r0, r1, r2, c1, c2]
nx = 2
nu = 1
T = 40
dt = 0.1
N = int(T // dt)
t = np.linspace(0., T - dt, N)
CPU = Center(nu, nx, 10, 10, 0.01)

total_episodes = 10000

for ep in range(total_episodes):
    last_state = np.array([0, 0])
    CPU.get_last_state(last_state)
    U = np.random.normal(size=(N, nu)) * ss.square(t).reshape((N, nu))
    # U = np.ones((N, nu)) * np.sin(t).reshape((N, nu))
    real = RC.RCSimulation(params, t, U, method="exact")
    episode_state = []
    for i in range(N):
        flag = CPU.get_batch(U[i], real[i])
        if flag:
            CPU.get_last_state(last_state)
            last_state = real[i]
            predict = CPU.learn()
            predict = np.vstack(predict)
            episode_state.append(predict)
            # print()

    # 画出训练效果图
    episode_state = np.vstack(episode_state)
    # print(real)
    # print("-------------")
    # print(episode_state)
    plt.subplot(221)
    plt.plot(real[:, 0], 'b')
    plt.plot(episode_state[:, 0], 'r', linestyle='-.')
    plt.subplot(222)
    plt.plot(real[:, 1], 'b')
    plt.plot(episode_state[:, 1], 'r', linestyle='-.')

    fig_name = "./fig/" + str(ep) + ".jpg"
    plt.savefig(fig_name)
    plt.close()

# test by square
# u = np.random.normal(size=(N, nu)) * ss.square(t).reshape((N, nu))
# a = RC.RCSimulation(params, t, u, method="exact")


CPU.save_model()
