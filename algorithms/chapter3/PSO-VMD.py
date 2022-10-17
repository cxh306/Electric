import vmdpy
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.svm import SVC
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vmdpy import VMD
from scipy.fftpack import hilbert,fft,ifft
from math import log
import pandas as pd

from data_process import load_data

dataset = load_data("澳大利亚.xlsx", "Sheet2")
load = dataset['电力负荷']
x = load[0:2048]


tau = 0.  # noise-tolerance (no strict fidelity enforcement)
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7



# 3.初始化参数
W = 0.5                                # 惯性因子
c1 = 0.2                                # 学习因子
c2 = 0.5                                # 学习因子
n_iterations = 50                       # 迭代次数
n_particles = 30                       # 种群规模
low = [3, 100]
up = [10, 3000]
var_num = 2
bound = (low,up)


# 4.设置适应度值
def fitness_function(position):
    
    K = int(position[0])
    alpha = position[1]
    if K < bound[0][0]:
        K = bound[0][0]
    if K > bound[1][0]:
        K = bound[1][0]
        
    if alpha < bound[0][1]:
        alpha = bound[0][1]
    if alpha > bound[1][1]:
        alpha =bound[1][1]

    
    u, u_hat, omega = vmdpy.VMD(x, alpha, tau, K, DC, init, tol)
    #  
    EP = []
    for i in range(K):
        H = np.abs(hilbert(u[i,:]))
        e1 = []
        for j in range(len(H)):
            p = H[j]/np.sum(H)
            e = -p*log(p,2)
            e1.append(e)
        E = np.sum(e1)  
        EP.append(E)
    s = np.sum(EP)/K
    return s



## 5.粒子图
# def plot(position):
#     x = []
#     y = []
#     for i in range(0, len(particle_position_vector)):
#         x.append(particle_position_vector[i][0])
#         y.append(particle_position_vector[i][1])
#     colors = (0, 0, 0)
#     plt.scatter(x, y, c=colors, alpha=0.1)
#     # 设置横纵坐标的名称以及对应字体格式
#     #font2 = {'family': 'Times New Roman','weight': 'normal', 'size': 20,}
#     plt.xlabel('gamma')
#     plt.ylabel('C')
#     plt.axis([0, 10, 0, 10],)
#     plt.gca().set_aspect('equal', adjustable='box')
#     return plt.show()

# 6.初始化粒子位置，进行迭代
pop_x = np.zeros((n_particles,var_num))
g_best = np.zeros(var_num)
temp = -1
for i in range(n_particles):
    for j in range(var_num):
        pop_x[i][j] = np.random.rand()*(bound[1][j]-bound[0][j])+bound[0][j]
    fit = fitness_function(pop_x[i])
   
    if fit > temp:
        g_best = pop_x[i]
        temp = fit
# particle_position_vector = np.array([np.array([random.random() * 100, random.random() * 100]) for _ in range(n_particles)])
# print('zzz',particle_position_vector)
pbest_position = pop_x
pbest_fitness_value = np.zeros(n_particles)
# print(pbest_fitness_value)
gbest_fitness_value = np.zeros(var_num)
# print(gbest_fitness_value[1])
gbest_position = g_best
velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])
iteration = 0

while iteration < n_iterations:
    # plot(particle_position_vector)
    print(iteration)
    for i in range(n_particles):
        # print(pop_x[i])
        fitness_cadidate = fitness_function(pop_x[i])
        # print("error of particle-", i, "is (training, test)", fitness_cadidate)
        # print(" At (K, alpha): ",int(pop_x[i][0]),pop_x[i][1])

        if (pbest_fitness_value[i] > fitness_cadidate):
            pbest_fitness_value[i] = fitness_cadidate
            pbest_position[i] = pop_x[i]

        elif (gbest_fitness_value[1] > fitness_cadidate):
            gbest_fitness_value[1] = fitness_cadidate
            gbest_position = pop_x[i]

        elif (gbest_fitness_value[0] < fitness_cadidate):
            gbest_fitness_value[0] = fitness_cadidate
            gbest_position = pop_x[i]

    for i in range(n_particles):
        new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                    pbest_position[i] - pop_x[i]) + (c2 * random.random()) * (
                                   gbest_position - pop_x[i])
        new_position = new_velocity + pop_x[i]
        pop_x[i] = new_position

    iteration = iteration + 1

plt.plot()
# 7.输出最终结果
print("The best position is ", int(gbest_position[0]),gbest_position[1], "in iteration number", iteration, "with error (train, test):",
      fitness_function(gbest_position))





