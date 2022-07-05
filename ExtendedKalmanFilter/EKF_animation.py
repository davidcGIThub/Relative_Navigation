#homework 2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from RobotMotion import RobotMotion as robot
from LandmarkModel import LandmarkModel as lmd
from ExtendedKalmanFilter import EKF

t = np.linspace(0,20,int(20/0.1+1));
x_true = t * 0;
y_true = t * 0;
theta_true = t * 0;
x_est = t * 0;
y_est = t * 0;
theta_est = t * 0;

dt = 0.1;
vc = 1 + 0.5*np.cos(2.0*np.pi*.2*t);
wc = -0.2 + 2*np.cos(2*np.pi*0.6*t);
x0 = -5.0; #m
y0 = -3.0; #m
theta0 = np.pi/2.0; #rad
state = np.array([x0,y0,theta0])
mu = np.array([x0,y0,theta0]);
alpha1 = 0.1;
alpha2 = 0.01;
alpha3 = 0.01;
alpha4 = 0.1;
step = 0;
sig_r = 0.1;
sig_b = 0.05;
Sig = np.array([[1,0,0],
               [0,1,0],
               [0,0,0.1]]);
cov = np.zeros((3,np.size(t)));


rb = robot(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt);
rb_est = robot(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt);
landmarks = np.array([[6,4],[-7,8],[6,-4]]);
lmdModel = lmd(sig_r,sig_b);
ekf = EKF(dt,alpha1,alpha2,alpha3,alpha4,sig_r,sig_b,landmarks);

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-10, 10), ylim=(-10, 10));
ax.grid();
robot_fig = plt.Polygon(rb.getPoints(),fc = 'g');
robot_est_fig = plt.Polygon(rb_est.getPoints(),fill=False);
lmd1 = plt.Circle(landmarks[0], radius = 0.5, fc = 'b');
lmd2 = plt.Circle(landmarks[1], radius = 0.5, fc = 'b');
lmd3 = plt.Circle(landmarks[2], radius = 0.5, fc = 'b');
lmd1_meas = plt.Circle(landmarks[0], radius = 0.5, fill = False)
lmd2_meas = plt.Circle(landmarks[1], radius = 0.5, fill = False)
lmd3_meas = plt.Circle(landmarks[2], radius = 0.5, fill = False)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes);

def init():
    #initialize animation
    ax.add_patch(robot_fig);
    ax.add_patch(robot_est_fig);
    ax.add_patch(lmd1);
    ax.add_patch(lmd2);
    ax.add_patch(lmd3);
    ax.add_patch(lmd1_meas);
    ax.add_patch(lmd2_meas);
    ax.add_patch(lmd3_meas);
    time_text.set_text('');
    return robot_fig, robot_est_fig, lmd1_meas, lmd2_meas, lmd3_meas, time_text

def animate(i):
    global rb, rb_est, landmarks, t, vc, wc, mu, Sig;
    #propogate robot motion
    u = np.array([vc[i],wc[i]]);
    rb.vel_motion_model(u);
    robot_fig.xy  = rb.getPoints();
    state = rb.getState();
    #estimate robot motion
    (mu, Sig, landmarks_meas)  = ekf.EKF_Localization(mu,Sig,u,state);
    rb_est.setState(mu[0],mu[1],mu[2]);
    robot_est_fig.xy  = rb_est.getPoints();
    #measure landmark position
    lmd1_meas.center = landmarks_meas[0];
    lmd2_meas.center = landmarks_meas[1];
    lmd3_meas.center = landmarks_meas[2];
    #update time
    time_text.set_text('time = %.1f' % t[i])
    #save state information
    x_true[i] = state[0];
    y_true[i] = state[1];
    theta_true[i] = state[2];
    x_est[i] = mu[0];
    y_est[i] = mu[1];
    theta_est[i] = mu[2];
    cov[0][i] = Sig[0][0];
    cov[1][i] = Sig[1][1];
    cov[2][i] = Sig[2][2];

    return robot_fig, robot_est_fig, lmd1_meas, lmd2_meas, lmd3_meas, time_text

from time import time
animate(0);

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show();

err_bnd_x = 2*np.sqrt(cov[0][:]);
err_bnd_y = 2*np.sqrt(cov[1][:]);
err_bnd_th = 2*np.sqrt(cov[2][:]);

figure1, (ax1, ax2, ax3) = plt.subplots(3,1);
ax1.plot(t,x_true, label = 'true');
ax1.plot(t,x_est, label = 'estimate');
ax1.legend();
ax1.set(ylabel = 'x position (m)');
ax2.plot(t,y_true);
ax2.plot(t,y_est);
ax2.set(ylabel = 'y position (m)');
ax3.plot(t,theta_true);
ax3.plot(t,theta_est);
ax3.set(ylabel = 'heading (deg)', xlabel= ("time (s)"));
plt.show();

figure2, (ax1, ax2, ax3) = plt.subplots(3,1);
ax1.plot(t,x_true-x_est, label = 'error', color = 'b');
ax1.plot(t,err_bnd_x, label = 'error_bound', color = 'r');
ax1.plot(t,-err_bnd_x, color = 'r');
ax1.legend();
ax1.set(ylabel = 'x error');
ax2.plot(t,y_true-y_est, color = 'b');
ax2.plot(t,err_bnd_y, color = 'r');
ax2.plot(t,-err_bnd_y, color = 'r');
ax2.set(ylabel = 'y error (m)');
ax3.plot(t,theta_true-theta_est,color = 'b');
ax3.plot(t,err_bnd_th,color = 'r');
ax3.plot(t,-err_bnd_th,color = 'r');
ax3.set(ylabel = 'heading error (rad)', xlabel= ("time (s)"));
plt.show();