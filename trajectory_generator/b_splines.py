"""
This module contains code to evaluate an open uniform b spline 
using the matrix form method
"""

from matplotlib import scale
import numpy as np 

class BsplineMatrixEvaluation:
    """
    This class contains code to evaluate an open uniform b spline using the
    matrix form method and creates a list of data points through time
    """

    def __init__(self, control_points, order, start_time, scale_factor=1,clamped=True):
        '''
        Constructor for the BsplineMatrixEvaluation class, each column of
        control_points is a control point. Start time should be an integer.
        '''
        self._control_points = control_points
        self._order = order
        self._scale_factor = scale_factor
        self._start_time = start_time
        self._clamped = clamped
        if clamped:
            self._knot_points = self.__create_clamped_knot_points()
        else:
            self._knot_points = self.__create_knot_points()

    def get_spline_data(self , number_of_data_points):
        '''
        Returns equally distributed data points for the spline, as well
        as time data for the parameterization
        '''
        number_of_control_points = self.__get_number_of_control_points()
        end_knot_point = self._knot_points[number_of_control_points]
        total_time_interval = end_knot_point - self._start_time
        end_time = end_knot_point - total_time_interval*.0001
        time_data = np.linspace(self._start_time, end_time, number_of_data_points)
        dimension = self._control_points.ndim
        if dimension == 1:
            spline_data = np.zeros(number_of_data_points)
        else:
            spline_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_data[i] = self.__evaulate_spline_at_time_t(t)
            else:
                spline_data[:,i][:,None] = self.__evaulate_spline_at_time_t(t)
        return spline_data, time_data

    def get_defined_knot_points(self):
        '''
        returns the knot points that are defined along the curve
        '''
        number_of_control_points = self.__get_number_of_control_points()
        defined_knot_points = self._knot_points[self._order:number_of_control_points]
        return defined_knot_points

    def get_knot_points(self):
        '''
        returns all the knot points
        '''
        return self._knot_points

    def get_spline_at_knot_points(self):
        '''
        Returns spline data evaluated at the knot points for
        which the spline is defined.
        '''
        time_data = self.get_defined_knot_points()
        number_of_data_points = len(time_data)
        dimension = self._control_points.ndim
        if dimension == 1:
            spline_data = np.zeros(number_of_data_points)
        else:
            spline_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_data[i] = self.__evaulate_spline_at_time_t(t)
            else:
                spline_data[:,i][:,None] = self.__evaulate_spline_at_time_t(t)
        return spline_data

    def get_spline_at_time_t(self, time):
        return self.__evaulate_spline_at_time_t(time)

    def get_spline_derivative_data(self,number_of_data_points, rth_derivative):
        '''
        Returns equally distributed data points for the derivative of the spline, 
        as well as time data for the parameterization
        '''
        number_of_control_points = self.__get_number_of_control_points()
        end_knot_point = self._knot_points[number_of_control_points]
        total_time_interval = end_knot_point - self._start_time
        end_time = end_knot_point - total_time_interval*.0001
        time_data = np.linspace(self._start_time, end_time, number_of_data_points)
        dimension = self._control_points.ndim
        if dimension == 1:
            spline_derivative_data = np.zeros(number_of_data_points)
        else:
            spline_derivative_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_derivative_data[i] = self.__evaluate_spline_derivative_at_time_t(t,rth_derivative)
            else:
                spline_derivative_data[:,i][:,None] = self.__evaluate_spline_derivative_at_time_t(t,rth_derivative)
        return spline_derivative_data, time_data

    def __evaluate_spline_derivative_at_time_t(self,time,rth_derivative):
        """
        This function evaluates the B spline derivative at the given time
        """
        preceding_knot_index = self.__find_preceding_knot_index(time)
        preceding_knot_point = self._knot_points[preceding_knot_index]
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        M = self.__get_M_matrix(initial_control_point_index)
        i_p = initial_control_point_index
        tau = (time - preceding_knot_point)
        P = np.zeros((dimension,self._order+1))
        for y in range(self._order+1):
            P[:,y] = self._control_points[:,i_p+y]
        T = np.ones((self._order+1,1))
        for i in range(self._order-rth_derivative+1):
            T[i,0] = (tau**(self._order-rth_derivative-i)*np.math.factorial(self._order-i)) /  (self._scale_factor**(self._order-i)*np.math.factorial(self._order-i-rth_derivative))
        spline_derivative_at_time_t = np.dot(P, np.dot(M,T))
        return spline_derivative_at_time_t


    def __evaulate_spline_at_time_t(self, time):
        """
        This function evaluates the B spline at the given time
        """
        preceding_knot_index = self.__find_preceding_knot_index(time)
        preceding_knot_point = self._knot_points[preceding_knot_index]
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        spline_at_time_t = np.zeros((dimension,1))
        if self._order > 5:
            # TODO
        else:
            M = self.__get_M_matrix(initial_control_point_index)
            i_p = initial_control_point_index
            tau = time - preceding_knot_point
            P = np.zeros((dimension,self._order+1))
            for y in range(self._order+1):
                P[:,y] = self._control_points[:,i_p+y]
            T = np.ones((self._order+1,1))
            for kappa in range(self._order+1):
                T[kappa,0] = (tau/self._scale_factor)**(self._order-kappa)
            spline_at_time_t = np.dot(P, np.dot(M,T))
        return spline_at_time_t

    def __evaulate_spline_at_time_t(self, time, method):
        """
        This function evaluates the B spline at the given time
        """
        preceding_knot_index = self.__find_preceding_knot_index(time)
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        spline_at_time_t = np.zeros((dimension,1))
        for i in range(initial_control_point_index , initial_control_point_index+self._order + 1):
            if dimension == 1:
                control_point = self._control_points[i]
            else:
                control_point = self._control_points[:,i][:,None]
            if method == "table":
                iteration_value = self.__cox_de_boor_formula(time, i,self._order,"table") * control_point
            else: #method = "recursion"
                iteration_value = self.__cox_de_boor_formula(time, i,self._order,"recursion") * control_point
            spline_at_time_t += iteration_value
        return spline_at_time_t

    def __get_M_matrix(self, initial_control_point_index):
        if self._order == 2:
            M = self.__get_2_order_matrix(initial_control_point_index)
        elif self._order == 3:
            M = self.__get_3_order_matrix(initial_control_point_index)
        elif self._order == 4:
            M = self.__get_4_order_matrix(initial_control_point_index)
        elif self._order == 5:
            M = self.__get_5_order_matrix(initial_control_point_index)
        return M

    def __get_2_order_matrix(self,initial_control_point_index):
        i_t = initial_control_point_index + self._order
        n = self.__get_number_of_control_points()
        M = .5*np.array([[1,-2,1],
                         [-2,2,1],
                         [1,0,0]])
        if self._clamped:
            if i_t == 2 and self._knot_points[2] < self._knot_points[n-1]:
                M = .5*np.array([[2,-4,2],
                                [-3,4,0],
                                [1,0,0]])
            elif i_t == n - 1 and self._knot_points[2] < self._knot_points[n-1]:
                M = .5*np.array([[1,-2,1],
                                [-3,2,1],
                                [2,0,0]])
            elif i_t == 2 and self._knot_points[2] == self._knot_points[n-1] :
                M = 0.5*np.array([[2,-4,2],
                                  [-4,4,0],
                                  [2,0,0]])
        return M

    def __get_3_order_matrix(self, initial_control_point_index):
        i_t = initial_control_point_index + self._order
        n = self.__get_number_of_control_points()
        M = np.array([[-2 ,  6 , -6 , 2],
                      [ 6 , -12 ,  0 , 8],
                      [-6 ,  6 ,  6 , 2],
                      [ 2 ,  0 ,  0 , 0]])/12
        if self._clamped:
            if i_t == 3 and self._knot_points[3] < self._knot_points[n-1]:
                M = np.array([[-12 ,  36 , -36 , 12],
                            [ 21 , -54 ,  36 , 0],
                            [-11 ,  18 ,  0  , 0],
                            [ 2  ,  0  ,  0  , 0]])/12.0
            elif i_t == 4 and self._knot_points[4] < self._knot_points[n-2]:
                M = np.array([[-3 ,  9 , -9 , 3],
                            [ 7 , -15 ,  3 , 7],
                            [-6 ,  6 , 6 , 2],
                            [ 2 ,  0 ,  0 , 0]])/12

            elif i_t == n - 2 and self._knot_points[4] < self._knot_points[n-2]:
                M = np.array([[-2  , 6   ,  -6 , 2],
                            [6 , -12  , 0 ,  8],
                            [ -7,    6 ,  6 , 2 ],
                            [ 3,  0   , 0  , 0]])/12
            elif i_t == n - 1 and self._knot_points[3] < self._knot_points[n-1]:
                M = np.array([[-2  , 6   ,  -6 , 2],
                            [11 , -15  , -3 ,  7],
                            [-21,  9   ,  9 ,  3],
                            [12 ,  0   ,  0 ,  0]])/12.0
            elif i_t == 4 and self._knot_points[4] == self._knot_points[n-2]:
                M = np.array([[-3 , 9 , -9 , 3],
                    [7, -15 , 3 , 7],
                    [-7 , 6 , 6 , 2],
                    [3 , 0 , 0 , 0]])/12
            elif i_t == 3 and self._knot_points[3] == self._knot_points[n-1]:
                 M = np.array([[-12 , 36 , -36 , 12],
                      [36 , -72 , 36 , 0],
                      [-36 , 36 , 0 , 0],
                      [12 , 0 , 0 , 0]])/12
        return M

    def __get_4_order_matrix(self, initial_control_point_index):
        M = np.array([[ 1 , -4  ,  6 , -4  , 1],
                      [-4 ,  12 , -6 , -12 , 11],
                      [ 6 , -12 , -6 ,  12 , 11],
                      [-4 ,  4  ,  6 ,  4  , 1],
                      [ 1 ,  0  ,  0 ,  0  , 0]])/24
        return M

    def __get_5_order_matrix(self, initial_control_point_index):
        M = np.array([[-1  ,  5  , -10 ,  10 , -5  , 1],
                      [ 5  , -20 ,  20 ,  20 , -50 , 26],
                      [-10 ,  30 ,  0  , -60 ,  0  , 66],
                      [ 10 , -20 , -20 ,  20 ,  50 , 26],
                      [-5  ,  5  ,  10 ,  10 ,  5  , 1 ],
                      [ 1  ,  0  ,  0  ,  0  ,  0  , 0]])/120
        return M

    def __create_knot_points(self):
        '''
        This function creates evenly distributed knot points
        '''
        number_of_control_points = self.__get_number_of_control_points()
        number_of_knot_points = number_of_control_points + self._order + 1
        knot_points = np.arange(number_of_knot_points)*self._scale_factor + self._start_time - self._order*self._scale_factor
        return knot_points

    def __create_clamped_knot_points(self):
        """
        Creates the list of knot points in the closed interval [t_0, t_{k+p}]
        with the first k points equal to t_k and the last k points equal to t_{p}
        where k = order of the polynomial, and p = number of control points
        """
        number_of_control_points = len(self._control_points[0])
        number_of_knot_points = number_of_control_points + self._order + 1
        number_of_unique_knot_points = number_of_knot_points - 2*self._order
        unique_knot_points = np.arange(0,number_of_unique_knot_points) * self._scale_factor + self._start_time
        knot_points = np.zeros(number_of_knot_points) + self._start_time
        knot_points[self._order : self._order + number_of_unique_knot_points] = unique_knot_points
        knot_points[self._order + number_of_unique_knot_points: 2*self._order + number_of_unique_knot_points] = unique_knot_points[-1]
        return knot_points
        
    def __find_preceding_knot_index(self,time):
        """ 
        This function finds the knot point preceding
        the current time
        """
        preceding_knot_index = -1
        number_of_control_points = len(self._control_points[0])
        for knot_index in range(self._order,number_of_control_points+1):
            preceding_knot_index = number_of_control_points - 1
            knot_point = self._knot_points[knot_index]
            next_knot_point = self._knot_points[knot_index + 1]
            if time >= knot_point and time < next_knot_point:
                preceding_knot_index = knot_index
                break
        return preceding_knot_index

    def __get_number_of_control_points(self):
        if self._control_points.ndim == 1:
            number_of_control_points = len(self._control_points)
        else:
            number_of_control_points = len(self._control_points[0])
        return number_of_control_points

    def __cox_de_boor_formula(self,time,i,kappa,method):
        
        if method == "table":
            return self.__cox_de_boor_formula_table_method(time,i)
        else: #method = "recursion"
            return self.__cox_de_boor_formula_recursion_method(time,i,kappa)

    def __cox_de_boor_formula_recursion_method(self, time, i, kappa):
        t = time
        t_i = self._knot_points[i]
        t_i_1 = self._knot_points[i+1]
        t_i_k = self._knot_points[i+kappa]
        t_i_k_1 = self._knot_points[i+kappa+1]
        if kappa == 0:
            if t >= t_i and t < t_i_1:
                basis_function = 1
            else:
                basis_function = 0
        else:
            if t_i < t_i_k:
                term1 = (t - t_i) / (t_i_k - t_i) * self.__cox_de_boor_formula_recursion_method(t,i,kappa-1)
            else:
                term1 = 0
            if t_i_1 < t_i_k_1:
                term2 = (t_i_k_1 - t)/(t_i_k_1 - t_i_1) * self.__cox_de_boor_formula_recursion_method(t,i+1,kappa-1)
            else:
                term2 = 0
            basis_function = term1 + term2
        return basis_function

    def __cox_de_boor_formula_table_method(self, time, i):
        table = np.zeros((self._order+1,self._order+1))
        #loop through rows to create the first column
        for y in range(self._order+1):
            t_i_y = self._knot_points[i+y]
            t_i_y_1 = self._knot_points[i+y+1]
            if time >= t_i_y and time < t_i_y_1:
                table[y,0] = 1
        # loop through remaining columns
        for kappa in range(1,self._order+1): 
            # loop through rows
            number_of_rows = self._order+1 - kappa
            for y in range(number_of_rows):
                t_i_y = self._knot_points[i+y]
                t_i_y_1 = self._knot_points[i+y+1]
                t_i_y_k = self._knot_points[i+y+kappa]
                t_i_y_k_1 = self._knot_points[i+y+kappa+1]
                horizontal_term = 0
                diagonal_term = 0
                if t_i_y_k > t_i_y:
                    horizontal_term = table[y,kappa-1] * (time - t_i_y) / (t_i_y_k - t_i_y)
                if t_i_y_k_1 > t_i_y_1:
                    diagonal_term = table[y+1,kappa-1] * (t_i_y_k_1 - time)/(t_i_y_k_1 - t_i_y_1)
                table[y,kappa] = horizontal_term + diagonal_term
        return table[0,self._order]


   