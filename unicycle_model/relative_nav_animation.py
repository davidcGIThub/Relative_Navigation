#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from unicycle_model import UnicycleModel
from MeasurementModel import MeasurementModel as mmd
from EKF_SLAM_Unicycle import EKF
from unicycle_kinematic_trajectory_tracker import UnicycleKinematicTrajectoryTracker
from figure_eight_trajectory import FigureEightTrajectoryGenerator
from data_initialization import *


rb = UnicycleModel(x0,y0,theta0,np.array([alpha1,alpha2,alpha3,alpha4]),dt)
rb_est = UnicycleModel(x0,y0,theta0,np.array([alpha1,alpha2,alpha3,alpha4]),dt)
measDevice = mmd(sig_r,sig_b)
ekf = EKF(dt,alpha,sig_r,sig_b)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
robot_fig = plt.Polygon(rb.getPoints(),fc = 'g')
robot_est_fig = plt.Polygon(rb_est.getPoints(),fill=False)
desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')
lmd_figs, = ax.plot([],[], 'bo', ms=ms); 
lmdMeas_figs, = ax.plot([],[], 'ko', fillstyle = 'none', ms=ms)
cov_figs, =  ax.plot([],[], '.', ms = .1)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

controller = UnicycleKinematicTrajectoryTracker(dt = dt,
                                                kp_p = 3, 
                                                kp_i = 0,
                                                kv_p = 1,
                                                ktheta_p = 1,
                                                v_max = 15,
                                                omega_max = np.pi,
                                                tolerance = 0.005)
amplitude = 10
frequency = 0.15
traj_gen = FigureEightTrajectoryGenerator(amplitude, frequency)
path = traj_gen.evaluate_trajectory_over_time_interval(t)

def init():
    #initialize animation
    ax.add_patch(robot_fig)
    ax.add_patch(robot_est_fig)
    ax.add_patch(desired_position_fig)
    lmd_figs.set_data(landmarks[:,0],landmarks[:,1])
    lmdMeas_figs.set_data([],[])
    cov_figs.set_data([],[])
    time_text.set_text('')
    return robot_fig, robot_est_fig,desired_position_fig, cov_figs, lmd_figs, lmdMeas_figs, time_text

def animate(i):
    global rb, rb_est, landmarks, t, vc, wc, mu, Sig, mu_dot, c , detected_flag, controller, traj_gen
    #propogate robot motion
    plt.plot(path[:,0],path[:,1])
    position = traj_gen.evaluate_trajectory_at_time_t(t[i])
    velocity = traj_gen.evaluate_derivative_at_time_t(t[i])
    states_desired = np.array([position[0],position[1],None,velocity[0],velocity[1],None])
    estimated_states = np.array([mu[0],mu[1],mu[2],mu_dot[0],mu_dot[1],mu_dot[2]])
    v_c, omega_c = controller.trajectory_tracker(estimated_states, states_desired)
    u = np.array([v_c, omega_c])
    rb.velMotionModel(u)
    robot_fig.xy  = rb.getPoints()
    state = rb.getState()
    #measure landmark position
    Ranges = measDevice.getRanges(state,landmarks)
    (Bearings,c) = measDevice.getBearings(state,landmarks,fov)
    z = np.concatenate((Ranges,Bearings),1)
    #estimate robot motion
    (mu, Sig, mu_dot) = ekf.EKF_SLAM(mu,Sig,u,z,c,detected_flag)
    detected_flag[c > 0] = 1
    rb_est.setState(mu[0],mu[1],mu[2],mu_dot[0],mu_dot[1],mu_dot[2])
    #print("xy", mu[0], mu[1])
    robot_est_fig.xy = rb_est.getPoints()
    desired_position_fig.center = (position[0],position[1])
    #update landmark estimates
    #landmark_meas = measDevice.getLandmarkEstimates(state,Ranges,Bearings)
    landmark_meas = np.reshape(mu[3:3+2*N],(N,2))
    lmdMeas_figs.set_data(landmark_meas[:,0], landmark_meas[:,1])
    lmdMeas_figs.set_markersize(ms)
    #plot covariance bounds
    cov[:,i] = Sig.diagonal()
    points = measDevice.getCovariancePoints(mu[3:3+2*N],cov[:,i][3:3+2*N])
    cov_figs.set_data(points[:,0], points[:,1])
    cov_figs.set_markersize(.5)
    #update time
    time_text.set_text('time = %.1f' % t[i])
    #save state information
    x_true[i] = state[0]
    y_true[i] = state[1]
    theta_true[i] = state[2]
    x_est[i] = mu[0]
    y_est[i] = mu[1]
    theta_est[i] = mu[2]
    return robot_fig, robot_est_fig,desired_position_fig, lmd_figs, lmdMeas_figs, time_text, cov_figs

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show()


err_bnd_x = 2*np.sqrt(cov[0][:])
err_bnd_y = 2*np.sqrt(cov[1][:])
err_bnd_th = 2*np.sqrt(cov[2][:])

figure1, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t,x_true, label = 'true')
ax1.plot(t,x_est, label = 'estimate')
ax1.plot(t,path[:,0],label="desired")
ax1.legend()
ax1.set(ylabel = 'x position (m)')
ax2.plot(t,y_true)
ax2.plot(t,y_est)
ax2.plot(t,path[:,1],label="desired")
ax2.set(ylabel = 'y position (m)')
ax3.plot(t,theta_true)
ax3.plot(t,theta_est)
ax3.set(ylabel = 'heading (deg)', xlabel= ("time (s)"))
plt.show()

figure2, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t,x_true-x_est, label = 'error', color = 'b')
ax1.plot(t,err_bnd_x, label = 'error_bound', color = 'r')
ax1.plot(t,-err_bnd_x, color = 'r')
ax1.legend()
ax1.set(ylabel = 'x error')
ax2.plot(t,y_true-y_est, color = 'b')
ax2.plot(t,err_bnd_y, color = 'r')
ax2.plot(t,-err_bnd_y, color = 'r')
ax2.set(ylabel = 'y error (m)')
heading_diff = theta_true-theta_est
heading_diff -= np.pi * 2 * np.floor((heading_diff + np.pi) / (2 * np.pi))
ax3.plot(t,heading_diff,color = 'b')
ax3.plot(t,err_bnd_th,color = 'r')
ax3.plot(t,-err_bnd_th,color = 'r')
ax3.set(ylabel = 'heading error (rad)', xlabel= ("time (s)"))
plt.show()

