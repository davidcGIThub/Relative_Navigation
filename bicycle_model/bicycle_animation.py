#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from bicycle_model import BicycleModel
import os

x_limits = 5
y_limits = 5
sec = 90
time_array = np.linspace(0,sec,int(sec/0.1+1))
dt = time_array[1]
L = 1
lr = 0.5
R = 0.2
v_max = 5
delta_max = np.pi/4
bike = BicycleModel(x = 0, 
                    y = 0,
                    theta = np.pi/4,
                    delta = 0,
                    lr = lr,
                    L = L,
                    R = R,
                    alpha = np.array([0.1,0.01,0.1,0.01]),
                    dt = dt,
                    delta_max = delta_max)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
front_wheel_fig = plt.Polygon(bike.getFrontWheelPoints(),fc = 'k')
back_wheel_fig = plt.Polygon(bike.getBackWheelPoints(),fc = 'k')
body_fig = plt.Polygon(bike.getBodyPoints(),fc = 'g')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

global v_c, phi_c               
v_c = (1 + 0.5*np.cos(.5*time_array))/3
phi_c = np.cos(0.1*time_array)

def init():
    #initialize animation
    ax.add_patch(front_wheel_fig)
    ax.add_patch(back_wheel_fig)
    ax.add_patch(body_fig)
    time_text.set_text('')
    return front_wheel_fig, back_wheel_fig, body_fig, time_text

def animate(i):
    global bike, controller, traj_gen
    # propogate robot motion
    # x_d = 5
    # y_d = 5
    states = bike.getState() 
    t = time_array[i]
    input = np.array([v_c[i], phi_c[i]])
    bike.vel_motion_model(input)
    front_wheel_fig.xy = bike.getFrontWheelPoints()
    back_wheel_fig.xy = bike.getBackWheelPoints()
    body_fig.xy = bike.getBodyPoints()
    
    # update time
    time_text.set_text('time = %.1f' % time_array[i])

    return  front_wheel_fig, back_wheel_fig, body_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_array), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show()

# file_name = os.getcwd() + "/bike_animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# ani.save(file_name, writer=writergif)

file_name = os.getcwd() + "/bike_animation.gif"
ani.save(file_name, writer='imagemagick', fps=60)