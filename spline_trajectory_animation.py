#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from unicycle_model.unicycle_model import UnicycleModel
from unicycle_model.unicycle_kinematic_trajectory_tracker import UnicycleKinematicTrajectoryTracker
from trajectory_generator.piecewise_bsplines import PiecewiseBsplineEvaluation

control_point_list =  [np.array([[ 4.15162277,  1.54394911, -4.32741921, -3.08691883,  1.07655706, 4.60722739,  5.1963863 ,  4.60722739],
                                [ 4.1347214 ,  0.9326393 ,  4.1347214 ,  6.29927616,  1.85869251, -0.3374413 ,  4.04896016,  8.14160067]]),
                       np.array([[4.77998022, 5.11000989, 4.77998022, 6.24770048, 6.96363333, 8.16987292, 9.41506354, 8.16987292], 
                                 [0.71283079, 4.02886477, 7.17171014, 8.20135749, 5.12612709, 2.88899245, 4.44078393, 9.34787182]])]
scale_factor_list =  [0.8479041973472522, 0.6458879348741837]
traj_gen = PiecewiseBsplineEvaluation(3,control_point_list,scale_factor_list)
number_of_data_points = 1000
path, time_array = traj_gen.get_spline_data(number_of_data_points)

x_limits = np.array([-5,10])
y_limits = np.array([0,10])
sec = time_array[-1]
dt = time_array[1]
max_velocity = 8
max_turn_rate = 50

unicycle = UnicycleModel(x = 1, 
                         y = 2,
                         theta = np.pi,
                         alpha = np.array([0.1,0.01,0.01,0.1]),
                         dt = dt)
unicycle.setState(1,2,np.pi,-5,0,0)

controller = UnicycleKinematicTrajectoryTracker(dt = dt,
                                                kp_p = 3, 
                                                kp_i = 0,
                                                kv_p = 1,
                                                ktheta_p = 1,
                                                v_max = max_velocity,
                                                omega_max = max_turn_rate,
                                                tolerance = 0.005)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(x_limits[0],x_limits[1]), ylim=(y_limits[0],y_limits[1]))
ax.grid()
robot_fig = plt.Polygon(unicycle.getPoints(),fc = 'g')
desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
states_array = unicycle.getState()


def init():
    #initialize animation
    # ax.add_patch(robot_fig)
    ax.add_patch(desired_position_fig)
    time_text.set_text('')
    return desired_position_fig, time_text #, robot_fig

def animate(i):
    global unicycle, controller, traj_gen, states_array
    # propogate robot motion
    # x_d = 10
    # y_d = 10
    # states_desired = np.array([x_d,y_d])
    t = time_array[i]
    position = traj_gen.get_spline_at_time(t).flatten()
    velocity = traj_gen.get_spline_derivative_at_time(t,1).flatten()
    print("velocity: " , velocity)
    states_desired = np.array([position[0],position[1],None,velocity[0],velocity[1],None])
    states = unicycle.getState()
    if i > 0:
        states_array = np.vstack((states_array,states))
    v_c, omega_c = controller.trajectory_tracker(states, states_desired)
    input = np.array([v_c, omega_c])
    unicycle.velMotionModel(input)
    robot_fig.xy = unicycle.getPoints()
    # update time
    # time_text.set_text('time = %.1f' % time_array[i])
    time_text.set_text('omega_c = %.1f' % omega_c)
    plt.plot(path[0,:],path[1,:])
    desired_position_fig.center = (position[0],position[1])
    return desired_position_fig, time_text#, robot_fig

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_array), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)
plt.show()

plt.figure(1)
plt.plot(time_array,states_array[:,0],label="robot state")
plt.plot(time_array,path[:,0],label="desired state")
plt.xlabel("Time")
plt.ylabel("X")
plt.legend()
plt.show()