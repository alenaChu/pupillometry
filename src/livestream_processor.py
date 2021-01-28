import cv2
import dlib
import argparse
import yaml
from src.eye_tracker import EyeTracker
from src.eyes_utils import predict_frame, annotate_frame


class LifeStreamProcessor:

    def __init__(self, config):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('detector_data/shape_predictor_68_face_landmarks.dat')
        if config.get('apply_tracker', False):
            self.eye_trackers = [EyeTracker(), EyeTracker()]
        else:
            self.eye_trackers = [None, None]

    def process_stream(self):
        cap = cv2.VideoCapture(0)

        while 1:
            ret, img = cap.read()
            eyes, pupils, irises = predict_frame(img, self.detector, self.predictor, self.eye_trackers)

            img = annotate_frame(img, pupils, irises, eyes)

            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and annotate videos in source folder using neural nets.')
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as config_stream:
        config_str = config_stream.read()
        config = yaml.load(config_str, Loader=yaml.FullLoader)
    processor = LifeStreamProcessor(config)
    processor.process_stream()
