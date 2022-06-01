"""
Unicycle Model Class
"""
import numpy as np

#velocity motion model
import numpy as np 

class UnicycleModel:

    def __init__(self, 
                 x = 0, 
                 y = 0, 
                 theta = np.pi/2.0, 
                 alpha = np.array([0.1,0.01,0.01,0.1]),
                 dt = 0.1,
                 height = 1,
                 width = 0.5):
        self.x = x
        self.y = y
        self.theta = self.wrapAngle(theta)
        self.x_prev = x
        self.y_prev = y
        self.theta_prev = theta
        self.alpha1 = alpha[0]
        self.alpha2 = alpha[1]
        self.alpha3 = alpha[2]
        self.alpha4 = alpha[3]
        self.dt = dt
        self.height = height
        self.width = width
    
    def setState(self,x,y,theta):
        self.x = x
        self.y = y
        self.theta = self.wrapAngle(theta)

    def setPreviousState(self, x_prev, y_prev, theta_prev):
        self.x_prev = x_prev
        self.y_prev = y_prev
        self.theta = self.wrapAngle(theta_prev)

    def vel_motion_model(self,u):
        v = u[0]
        w = u[1]
        v_hat = v + (self.alpha1 * v**2 + self.alpha2 * w**2) * np.random.randn()
        w_hat = w + (self.alpha3 * v**2 + self.alpha4 * w**2) * np.random.randn()
        self.x_prev = self.x
        self.y_prev = self.y
        self.theta_prev = self.theta
        self.x = self.x + v_hat * np.cos(self.theta) * self.dt
        self.y = self.y + v_hat * np.sin(self.theta) * self.dt
        print("w: " , w)
        print("w_hat: " , w_hat)
        print("w_hat*self.dt: " , w_hat*self.dt)
        print("prev theta: " , self.theta)
        self.theta = self.wrapAngle(self.theta + w_hat*self.dt)
        print("unwrapped theta: " , self.theta + w_hat*self.dt)
        print("theta: " , self.theta)

    def getState(self):
        return np.array([self.x,self.y,self.theta])

    def getPreviousState(self):
        return np.array([self.x_prev,self.y_prev,self.theta_prev])

    def getPoints(self):
        R = self.getRotationMatrix(self.theta)
        xy = np.array([[-self.height, self.height, -self.height],
                       [self.width, 0, -self.width]])
        xy = np.dot(R,xy)
        xy = xy + np.array([[self.x],[self.y]])
        return np.transpose(xy)

    def getRotationMatrix(self, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def wrapAngle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

