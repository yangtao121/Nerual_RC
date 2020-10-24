import RC
import numpy as np
import scipy.signal as ss
from Nerve_Core import Network as Center

# RC网络参数的设定
r0 = 1
r1 = 1
c1 = 1
c2 = 10
params = [r0, r1, c1, c2]
nx = 2
nu = 1
T = 40
dt = 0.1
N = int(T // dt)
t = np.linspace(0., T - dt, N)
CPU = Center(nu, nx, 100, 10, 0.01)

total_episodes = 100

for ep in range(total_episodes):
    last_state = np.array([0, 0])
    U = np.random.normal(size=(N, nu)) * ss.square(t).reshape((N, nu))
    real = RC.RCSimulation(params, t, U, method="exact")
    for i in range(N):
        flag = CPU.get_batch(U[i], real[i])
        if flag:
            CPU.get_last_state(last_state)
            last_state = real[i]
            CPU.learn()

    # 伪测试
