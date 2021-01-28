import cv2
import argparse
import yaml
import dlib
import os
import shutil
import time
from tqdm import tqdm
from src.eyes_utils import save_eye_patches, predict_frame, annotate_frame
from src.eye_tracker import EyeTracker


class VideoProcessor:
    """
    process all video files in given directory, save videos with drawn predictions, save json file with annotations
    """
    def __init__(self, config):

        self.config = config
        self.src_dir = config.get('src_dir')
        self.output_folder = config['output_dir']
        self.supported_extensions = ['.mp4', '.avi', '.MOV', '.mov']
        self.frame_range = config['frames_to_process']
        self.codec_fourcc = "mp4v"
        self.video_codec = cv2.VideoWriter_fourcc(*self.codec_fourcc)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('detector_data/shape_predictor_68_face_landmarks.dat')

        if config.get('apply_tracker', False):
            self.eye_trackers = [EyeTracker(), EyeTracker()]
        else:
            self.eye_trackers = [None, None]

    def process_folder(self):
        listOfFiles = []
        for (dirpath, dirnames, filenames) in os.walk(self.src_dir):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames if file[-4:] in self.supported_extensions]
        for i, path in enumerate(listOfFiles):
            self.process_video(path)

    def process_video(self, path):
        filename = config.get('prefix', '') + os.path.basename(path).split('.')[0] + '.mov'
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
            if not ret or frame_number > frame_range[1]:
                break
            if frame_number < frame_range[0]:
                continue

            eyes, pupils, irises = predict_frame(frame, self.detector, self.predictor, self.eye_trackers)

            if self.config.get('save_eye_patches', False):
                save_eye_patches(eyes, frame, frame_number)
            frame = annotate_frame(frame, pupils, irises, eyes, frame_label=frame_number)
            video_writer.write(frame)


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
