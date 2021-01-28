import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class EyeTracker:
    def __init__(self):
        self.eye_center_tracker = None
        self.iris_diameter_tracker = None
        self.pupil_diameter_tracker = None
        self.outliers_num = 0
        self.prediction = None
        self.prediction_pupil = 0

    def initialize(self, iris, pupil):
        if iris is not None:
            self.eye_center_tracker = self.initialize_kalman_2d(iris[:2])
            self.iris_diameter_tracker = self.initialize_kalman_1d(iris[2])
            self.prediction = iris
            if pupil is not None:
                self.pupil_diameter_tracker = self.initialize_kalman_1d(pupil[2])
                self.prediction_pupil = pupil[2]

    def update_trackers(self, iris, pupil):
        self.eye_center_tracker.update(np.array(iris[:2]))
        self.iris_diameter_tracker.update(iris[2])
        if self.pupil_diameter_tracker is None:
            return
        if pupil is not None and (type(pupil) is list or type(pupil) is np.array):
            self.pupil_diameter_tracker.update(pupil[2])
        elif pupil is not None:
            self.pupil_diameter_tracker.update(pupil)

    def update(self, iris, pupil):
        if self.eye_center_tracker is None:
            self.initialize(iris, pupil)
            return

        dist_far = 10
        dist_near = 3
        if iris is not None:
            iris_center = np.array(iris[:2])

            if abs(iris[2] - self.prediction[2]) < dist_near and \
                    np.linalg.norm(iris_center - np.array(self.prediction[:2])) < dist_near:
                self.outliers_num = 0
                self.update_trackers(iris, pupil)
            elif abs(iris[2] - np.array(self.prediction[2])) > dist_far or \
                    np.linalg.norm(iris_center - np.array(self.prediction[:2])) > dist_far:
                self.outliers_num += 1
                if self.outliers_num > 10:
                    self.initialize(iris, pupil)
                    self.outliers_num = 0
                else:
                    self.update_trackers(np.array(self.prediction), self.prediction_pupil)
            else:
                self.update_trackers(iris, pupil)
        else:
            iris = self.prediction
        self.eye_center_tracker.predict()
        self.iris_diameter_tracker.predict()
        self.prediction = [self.eye_center_tracker.x[0], self.eye_center_tracker.x[2], self.iris_diameter_tracker.x[0]]
        if self.pupil_diameter_tracker is not None:
            self.pupil_diameter_tracker.predict()
            self.prediction_pupil = self.pupil_diameter_tracker.x[0]
        return iris

    def initialize_kalman_2d(self, init_point):
        """ initialize kalman filter for 2D point, set initial localization from keypoint, speed = 0 """
        init_value = [init_point[0], 0., init_point[1], 0.] if init_point is not None else [0., 0., 0., 0.]
        dt = 0.03
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        kf.x = np.array(init_value)  # initial state
        kf.P *= 0.1  # initial uncertainty
        z_std = 0.1
        kf.R = np.diag([z_std ** 2, z_std ** 2])  # 1 standard
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1 ** 2, block_size=2)
        kf.predict()
        return kf

    def initialize_kalman_1d(self, init_value):
        """ initialize kalman filter for 1D values of iris diameter, speed = 0 """
        init_value = [init_value, 0.] if init_value is not None else [0., 0.]
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1.,1.], [0.,1.]])
        kf.H = np.array([[1.,0.]])
        kf.x = np.array(init_value)  # initial state
        kf.P *= 0.1  # initial uncertainty
        kf.R = np.array([[0.01]])
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.03, var=0.1 ** 2)
        kf.predict()
        return kf

