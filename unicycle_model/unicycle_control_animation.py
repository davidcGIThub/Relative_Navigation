#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from unicycle_model import UnicycleModel
from unicycle_kinematic_controller import UnicycleKinematicController
import os

x_limits = 25
y_limits = 25
sec = 90
time_array = np.linspace(0,sec,int(sec/0.1+1))
dt = time_array[1]
R = 0.2
v_max = 5
unicycle = UnicycleModel(x = 0, 
                    y = 0,
                    theta = np.pi/4,
                    alpha = np.array([0.1,0.01,0.01,0.1]),
                    dt = dt)
controller = UnicycleKinematicController(kp_xy = 1, 
                                         kd_xy = .8,
                                         kp_theta = 1,
                                         kd_theta = .5, 
                                         v_max = 7,
                                         omega_max = np.pi/4,
                                         tolerance = 0.05)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
robot_fig = plt.Polygon(unicycle.getPoints(),fc = 'g')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

global v_c, omega_c               
v_c = (1 + 0.5*np.cos(.5*time_array))/3
omega_c = np.cos(0.1*time_array)

def init():
    #initialize animation
    ax.add_patch(robot_fig)
    time_text.set_text('')
    return robot_fig, time_text

def animate(i):
    global unicycle, controller
    # propogate robot motion
    x_d = 10
    y_d = 10
    states_desired = np.array([x_d,y_d])
    states = unicycle.getState()
    previous_states = unicycle.getPreviousState()
    v_c, omega_c = controller.pd_control(states, previous_states, states_desired, dt)
    input = np.array([v_c, omega_c])
    unicycle.vel_motion_model(input)
    robot_fig.xy = unicycle.getPoints()
    # update time
    # time_text.set_text('time = %.1f' % time_array[i])
    time_text.set_text('omega_c = %.1f' % omega_c)

    return robot_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_array), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show()

# file_name = os.getcwd() + "/bike_animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# ani.save(file_name, writer=writergif)

file_name = os.getcwd() + "/unicycle_animation.gif"
ani.save(file_name, writer='imagemagick', fps=60)