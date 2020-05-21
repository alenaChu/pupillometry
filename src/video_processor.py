import cv2
import argparse
import yaml
import dlib
import os
import shutil
import time
import numpy as np
from tqdm import tqdm
from imutils import face_utils
from src.eyes_utils import eye_aspect_ratio, save_eye_patches, get_pupil_and_iris_params
from src.eye_tracker import EyeTracker


class VideoProcessor:
    """
    process all video files in given directory, save videos with drawn predictions, save json file with annotations
    """
    def __init__(self, config):

        self.src_dir = config.get('src_dir')
        self.output_folder = config['output_dir']
        self.supported_extensions = ['.mp4', '.avi', '.MOV', '.mov']
        self.frame_range = config['frames_to_process']
        self.codec_fourcc = "mp4v"
        self.video_codec = cv2.VideoWriter_fourcc(*self.codec_fourcc)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # grab the indexes of the facial landmarks for the left and right eye, respectively
        self.leye_ids = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.reye_ids = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        if config.get('apply_tracker', False):
            self.left_eye_tracker = EyeTracker()
            self.right_eye_tracker = EyeTracker()
        else:
            self.left_eye_tracker, self.right_eye_tracker = None, None

    def process_folder(self):
        listOfFiles = []
        for (dirpath, dirnames, filenames) in os.walk(self.src_dir):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames if file[-4:] in self.supported_extensions]
        for i, path in enumerate(listOfFiles):
            self.process_video(path)

    def process_video(self, path):
        filename = os.path.basename(path).split('.')[0] + '.mov'
        prepare_dir(self.output_folder, False)

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise Exception("An error occured opening the video: {}".format(path))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_size = (width, height)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Frame size: {frame_size}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print(os.path.join(self.output_folder, filename))

        frame_range = self.frame_range if self.frame_range is not None else (0, total_frames)

        video_writer = cv2.VideoWriter(os.path.join(self.output_folder, filename),
                                       self.video_codec, fps, frame_size)

        print(f"Started processing {filename} at {time.ctime()}")

        for frame_number in tqdm(range(total_frames)):

            ret, frame = cap.read()
            if not ret:
                break
            if frame_number > frame_range[1]:
                break

            if frame_number < frame_range[0]:
                continue

            eyes, pupils, irises, eye_ar = self.predict_frame(frame)
            if self.left_eye_tracker is not None:
                self.left_eye_tracker.update(pupils[0], irises[0])
                self.right_eye_tracker.update(pupils[1], irises[1])

            save_eye_patches(eyes, frame, frame_number)
            frame = annotate_frame(frame, eyes, pupils, irises, eye_ar, frame_label=frame_number)
            video_writer.write(frame)

    def predict_frame(self, frame):
        # frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)
        # loop over the face detections
        if rects is not None and len(rects) > 0:
            rect = rects[0]
        else:
            return None, None, None, 0
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        l_eye = shape[self.leye_ids[0]:self.leye_ids[1]]
        r_eye = shape[self.reye_ids[0]:self.reye_ids[1]]
        l_pupil, l_iris = get_pupil_and_iris_params(l_eye, frame, tracker=self.left_eye_tracker)
        r_pupil, r_iris = get_pupil_and_iris_params(r_eye, frame, tracker=self.right_eye_tracker)

        l_eye_AR = eye_aspect_ratio(l_eye)
        r_eye_AR = eye_aspect_ratio(r_eye)
        # average the eye aspect ratio together for both eyes
        eye_ar = (l_eye_AR + r_eye_AR) / 2.0

        return [l_eye, r_eye], [l_pupil, r_pupil], [l_iris, r_iris], eye_ar


def annotate_frame(frame, eyes, pupils, irises, eye_ar, frame_label=None):
    if frame_label is not None:
        frame = cv2.putText(frame, str(frame_label), (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, f"eye open={np.round(eye_ar,2)}", (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.1) + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    if eyes is not None:
        for eye in eyes:
            if eye is not None:
                eye_hull = cv2.convexHull(eye)
                cv2.drawContours(frame, [eye_hull], -1, (0, 255, 0), 1)
    if pupils is None:
        return frame
    for pupil in pupils:
        if pupil is not None:
            yc, xc, a, b = [int(round(x)) for x in pupil[1:5]]
            orientation = pupil[5]
            frame = cv2.ellipse(frame, (xc, yc), (b, a), orientation, 0, 360, color=(0, 255, 255), thickness=1)
    for iris in irises:
        if iris is not None:
            yc, xc, a, b = [int(round(x)) for x in iris[1:5]]
            orientation = iris[5]
            frame = cv2.ellipse(frame, (xc, yc), (b, a), orientation, 0, 360, color=(255, 0, 255), thickness=1)
    return frame


def prepare_dir(dir, clear=False):
    """ Create directory if doesn't exist, clear its content if clear=True"""
    if clear:
        shutil.rmtree(dir, ignore_errors=True)
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and annotate videos in source folder using neural nets.')
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as config_stream:
        config_str = config_stream.read()
        config = yaml.load(config_str, Loader=yaml.FullLoader)
    processor = VideoProcessor(config)
    if config.get('video_path'):
        processor.process_video(config['video_path'])
    else:
        processor.process_folder()
