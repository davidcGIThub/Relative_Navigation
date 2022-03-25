import numpy as np
from bicycle_model import BicycleModel
from bicycle_kinematic_controller import BicycleKinematicController


L = 1
lr = 0.5
R = 0.2
v_max = 5
delta_max = np.pi/4
dt = 0.1
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

controller = BicycleKinematicController(kp_xy = 1, 
                                        kd_xy = 0,
                                        kp_theta = 1,
                                        kd_theta = 0, 
                                        kp_delta = 1,
                                        v_max = v_max,
                                        delta_max = delta_max,
                                        l = lr,
                                        L = L)
xd = -2
yd = 1
v_c , phi_c = controller.p_control(0, 0, 3*np.pi/4, 0, xd, yd)
print("phi_c: " , phi_c)
print("v_c: " , v_c)

# v_c , phi_c = controller.p_control(0, 0, 3*np.pi/4, 0, xd, yd)
# print("phi_c: " , phi_c)
# print("v_c: " , v_c)