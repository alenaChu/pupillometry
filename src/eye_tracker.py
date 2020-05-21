import numpy as np


class EyeTracker:
    def __init__(self):
        self.eye_id = None  # left/ right
        self.eye_centers = []
        self.pupil_diameters = []
        self.iris_diameters = []
        self.eye_center_on_track = False
        self.iris_on_track = False

    def update(self, pupil, iris):
        eye_centers = []
        if pupil is not None:
            eye_centers.append(np.array(pupil[1:3]))
            self.pupil_diameters.append(((pupil[3] + pupil[4])/2))
        if iris is not None:
            eye_centers.append(np.array(iris[1:3]))
            iris_diam = (iris[3] + iris[4])/2
            self.iris_diameters.append(iris_diam)
            if len(self.iris_diameters) < 5:
                pass
            elif self.iris_on_track and abs(iris_diam - np.median(np.array(self.iris_diameters[-5:]))) < 3:
                pass
            elif not self.iris_on_track and abs(iris_diam - np.median(np.array(self.iris_diameters[-5:]))) < 3:
                self.iris_on_track = True
            elif self.iris_on_track and abs(iris_diam - np.median(np.array(self.iris_diameters[-5:]))) >= 3:
                self.iris_on_track = False
        if len(eye_centers) > 0:
            eye_center = np.mean(np.array(eye_centers), axis=0)
            self.eye_centers.append(eye_center)
            if len(self.eye_centers) < 5:
                pass
            elif self.eye_center_on_track and \
                np.linalg.norm(eye_center - np.median(np.array(self.eye_centers[-5:]), axis=0)) < 3:
                pass
            elif not self.eye_center_on_track and \
                np.linalg.norm(eye_center - np.median(np.array(self.eye_centers[-5:]), axis=0)) < 3:
                self.eye_center_on_track = True
            elif self.eye_center_on_track and \
                np.linalg.norm(eye_center - np.median(np.array(self.eye_centers[-5:]), axis=0)) > 3:
                self.eye_center_on_track = False
        print(f'on track: eye: {self.eye_center_on_track}, iris_diam: {self.iris_on_track}')


