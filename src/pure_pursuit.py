'''
* path_following_algorithms
* pure_pursuit.py
* Copyright (c) 2021, Seunghyun Lim
'''

import numpy as np
import matplotlib.pyplot as plt
import time

global L, wp, vx, theta, w_limit
L = 1.
wp = 0
vx = 0.3
theta = 0.5
w_limit = 0.48
waypoints = [[0, 8], [5, 8], [5, 0], [12, 0]]
position = [0, 8.5]

def position_vector(vx, w, theta):
    global w_limit
    tmp = np.dot(np.array([[np.cos(theta), 0],[np.sin(theta), 0],[0, 1]]),np.array([[vx],[w]]))
    Xdot = tmp[0][0]
    Ydot = tmp[1][0]
    tdot = tmp[2][0]
    if tdot >= w_limit:
        tdot = w_limit
    elif tdot <= -w_limit:
        tdot = -w_limit
    return Xdot, Ydot, tdot

def pure_pursuit(position, waypoints):
    global L, wp, vx, theta
    X = position[0]
    Y = position[1]
    x_n0 = waypoints[wp][0]
    y_n0 = waypoints[wp][1]
    x_n1 = waypoints[wp+1][0]
    y_n1 = waypoints[wp+1][1]
    up = ((X - x_n0)*(x_n1 - x_n0)+(Y - y_n0)*(y_n1 - y_n0))/((x_n1 - x_n0)**2+(y_n1 - y_n0)**2)
    xo = x_n0 + up*(x_n1 - x_n0)
    yo = y_n0 + up*(y_n1 - y_n0)

    error = np.sqrt((X - xo)**2+(Y - yo)**2)

    if np.sqrt((x_n1 - xo)**2+(y_n1 - yo)**2) <= L:
        if wp<len(waypoints)-2:
            wp=wp+1

    do = np.sqrt((X - xo)**2+(Y - yo)**2)
    dl = np.sqrt(L**2-do**2)

    wpvec = np.array([x_n1 - x_n0, y_n1 - y_n0])
    norm_vec = wpvec/np.linalg.norm(wpvec)

    [xl, yl] = [xo, yo] + dl*norm_vec
    rot_matrix = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    tmp = np.dot(rot_matrix, np.array([[xl-X],[yl-Y]]))
    yd = tmp[1][0]

    w = 2*yd*vx/(L**2)

    return vx, w, error

x_way = []
y_way = []
for item in waypoints:
    x_way.append(item[0])
    y_way.append(item[1])

w = 0
dt = 0.1
iter = 0
fig = plt.subplots()
plt.xlim(-2, 14)
plt.ylim(-2, 12)
command_list = []
error_list = []
odom_list = [[],[]]

while np.linalg.norm(np.array(position)-waypoints[len(waypoints)-1])>0.1:
    vx, w, error = pure_pursuit(position, waypoints)
    command_list.append([vx, w])
    error_list.append(error)
    Xdot, Ydot, tdot = position_vector(vx, w, theta)
    position[0] = position[0]+dt*Xdot
    position[1] = position[1]+dt*Ydot
    odom_list[0].append(position[0])
    odom_list[1].append(position[1])
    plt.plot(odom_list[0], odom_list[1], 'b')
    theta = theta+dt*tdot
    iter += 1
    plt.plot(x_way, y_way, 'r--')
    plt.arrow(position[0], position[1], Xdot, Ydot, width=0.05)
    plt.scatter(position[0], position[1])
    plt.title("Pure pursuit")
    plt.legend(['odometry','path'])
    plt.xlim(-2, 14)
    plt.ylim(-2, 12)
    plt.pause(dt*0.01)
    if np.linalg.norm(np.array(position)-waypoints[len(waypoints)-1])>0.1:
        plt.cla()
    else:
        plt.waitforbuttonpress()
        plt.cla()
        plt.axis(option='auto')
        plt.title("Position RMSE (L: {}m)".format(L))
        plt.plot(error_list)
        plt.waitforbuttonpress()
        plt.cla()
        plt.axis(option='auto')
        plt.title("Command input (L: {}m)".format(L))
        plt.plot(command_list)
        plt.legend(['Vx(m/s)','w(rad/s)'])
        plt.show()
print("Average Error: {}".format(np.average(error_list)))
