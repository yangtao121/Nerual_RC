import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.signal as ss
import matplotlib.pyplot as plt
import math
plt.close('all')



"""
The following is the real RC circuit code, and it works
"""
def RCSimulation(params, t, u, method = "exact"):
    """
    Simulate a simple RC circuit

    Input:
        method, a string should be in the list, ['first_order', 'exact', 'ss']
    Output:
        x, voltage on capacitors
    """
    #### System parameters
    nx = 2 # system order
    nu = 1 # input order
    r0, r1, r2, c1, c2 = params
    P = (r0 * np.ones((nx, nx)) + np.diag([r1, r2])) @ np.diag([c1, c2])
    Ac = - nla.solve(P, np.eye(nx))
    Bc = nla.solve(P, np.ones((nx, 1)))

    #### Simulation parameters
    N = len(t)
    dt = (t[-1] - t[0]) / (N - 1) # sample interval
    T = t[-1]
    #print(T)
    t = np.linspace(0., T - dt, N)
    # print(t.shape)

    #### Simulation
    if method == "ss":
        C = np.eye(nx)
        D = np.eye(nx, nu)
        sys = ss.StateSpace(Ac,Bc,C,D)
        t, y, x = ss.lsim(sys, u, t)
    else:
        #### Approximating the discret-time system
        if method == 'first_order':
            # First order
            print("using first order approximation")
            Ad = np.eye(nx) + Ac * dt
            Bd = Bc * dt
        elif method == 'exact':
            # Exact
            print("using the exact discrete A, B")
            Ad = sla.expm(Ac*dt)
            Ai = nla.inv(Ac)
            Bd = Ai @ (Ad - np.eye(nx)) @ Bc

        ## Initialize variables
        x = np.zeros((N, nx))
        ## The actual simulation
        for k in range(N):
            if k != N - 1:
                x[k+1] = Ad @ x[k] + Bd @ u[k]
        ## Solve the current
        i = np.zeros((N, nx))
        R = np.diag([r1, r2]) + r0
        for k in range(N):
            i[k] = nla.solve(R, np.array([u[k] - x[k,0], u[k] - x[k,1]])).reshape((-1))
    #### Plot states and input
  #   plt.figure(figsize = [8, 7])
  #   plt.plot(t, x[:,0], label = "v1")
  #   plt.plot(t, x[:, 1], label = "v2")
  # #  plt.plot(t, i[:, 0], label = "i1")
  # #  plt.plot(t, i[:, 1], label = "i2")
  # #  plt.plot(t, np.sum(i, axis = 1), label = "i")
  #   plt.plot(t, u, label = "u")
  #   plt.title("RC Circuit Simulation\n method = %s, r0 = %.2f, r1 = %.2f, r2 = %.2f, c1 = %.2f, c2 = %.2f"%(method, r0, r1, r2, c1, c2))
  #   plt.legend(loc = 0)
  #   plt.show()
    return i


def systemidentify(inp,outp):
    m=2
    u1=inp[m:N:1]
    u2=inp[m-1:N-1:1]
    x1=outp[m-1:N-1:1]
    x2=outp[m-2:N-2:1]
    x3=outp[m:N:1].reshape((N-2,nu))
    m=np.column_stack((u1,u2,-x1,-x2))
    theta = np.linalg.pinv(np.transpose(m)@m)@np.transpose(m)@x3
    print(theta)


def nonlinear_system(params, t, u, method = "exact"):
    nx = 2 # system order
    nu = 1 # input order
    r0, r1, r2, c1, c2 = params
    P = (r0 * np.ones((nx, nx)) + np.diag([r1, r2])) @ np.diag([c1, c2])
    Ac = - nla.solve(P, np.eye(nx))
    Bc = nla.solve(P, np.ones((nx, 1)))

    #### Simulation parameters
    N = len(t)
    dt = (t[-1] - t[0]) / (N - 1) # sample interval
    T = t[-1]
    t = np.linspace(0., T - dt, N)

    #### Simulation
    if method == "ss":
        C = np.eye(nx)
        D = np.eye(nx, nu)
        sys = ss.StateSpace(Ac,Bc,C,D)
        t, y, x = ss.lsim(sys, u, t)
    else:
        #### Approximating the discret-time system
        if method == 'first_order':
            # First order
            print("using first order approximation")
            Ad = np.eye(nx) + Ac * dt
            Bd = Bc * dt
        elif method == 'exact':
            # Exact
            print("using the exact discrete A, B")
            Ad = sla.expm(Ac*dt)
            Ai = nla.inv(Ac)
            Bd = Ai @ (Ad - np.eye(nx)) @ Bc

        ## Initialize variables
        x = np.zeros((N, nx))
        ## The actual simulation
        for k in range(N):
            if k != N - 1:
                x[k+1] = Ad @ x[k] + Bd @ u[k]
        a=x[:,0]
        nonlinear1=[2.5 if x>2.5 or x<-2.5 else x for x in a]
        ## Solve the current
        i = np.zeros((N, nx))
        R = np.diag([r1, r2]) + r0
        for k in range(N):
            i[k] = nla.solve(R, np.array([u[k] - nonlinear1[k], u[k] - x[k,1]])).reshape((-1))
    #### Plot states and input
  #   plt.figure(figsize = [8, 7])
  #   plt.plot(t, nonlinear1, label = "v1")
  #   #plt.plot(t, x[:, 1], label = "v2")
  #   plt.plot(t, i[:, 0], label = "i1")
  # #  plt.plot(t, i[:, 1], label = "i2")
  # #  plt.plot(t, np.sum(i, axis = 1), label = "i")
  #   plt.plot(t, u, label = "u")
  #   plt.title("nonlinear rc Simulation\n method = %s, r0 = %.2f, r1 = %.2f, r2 = %.2f, c1 = %.2f, c2 = %.2f"%(method, r0, r1, r2, c1, c2))
  #   plt.legend(loc = 0)
  #   plt.ion()
    return i



# r0=1
# r1 = 1
# r2 = 1
# c1=1
# c2 = 10
# params = [r0, r1, r2, c1, c2]
# nx = 2
# nu = 1
# T = 40
# dt=.1 # horizon
# N = int(T // dt)
# print(N)
# t = np.linspace(0., T - dt, N)
# # print(t.shape)
# # u = np.ones((N, nu)) * np.sin(t).reshape((N, nu))
# u = np.random.normal(size=(N, nu))  * ss.square(t).reshape((N,nu))
# a = RCSimulation(params, t, u, method = "exact")
# # print(a)
# inp=u
# outp=np.sum(a,axis=1)
# systemidentify(inp,outp)
# # b=nonlinear_system(params,t,u,method='exact')
# # inp=u
# # outp=np.sum(b,axis=1)
# # systemidentify(inp,outp)




    