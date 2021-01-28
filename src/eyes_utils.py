from scipy.spatial import distance as dist
import cv2
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import feature
from imutils import face_utils

# grab the indexes of the facial landmarks for the left and right eye, respectively
leye_ids = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
reye_ids = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def detect_eyes(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale_factor = 480 / min(gray.shape[:2])
    gray = cv2.resize(gray, dsize=None, fx=scale_factor, fy=scale_factor)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    if rects is not None and len(rects) > 0:
        rect = rects[0]
    else:
        return None, None
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    shape = np.array(shape / scale_factor, dtype=np.int)
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    l_eye = shape[leye_ids[0]:leye_ids[1]]
    r_eye = shape[reye_ids[0]:reye_ids[1]]
    return l_eye, r_eye


def predict_frame(frame, detector, predictor, eye_trackers=(None, None)):
    l_eye, r_eye = detect_eyes(frame, detector, predictor)
    if l_eye is None and r_eye is None:
        return None, None, None, 0
    if l_eye is not None:
        l_pupil, l_iris = get_pupil_and_iris_params(l_eye, frame, eye_trackers[0])
        if eye_trackers[0] is not None:
            eye_trackers[0].update(l_iris, l_pupil)

    if r_eye is not None:
        r_pupil, r_iris = get_pupil_and_iris_params(r_eye, frame, eye_trackers[1])
        if eye_trackers[1] is not None:
            eye_trackers[1].update(r_iris, r_pupil)

    return [l_eye, r_eye], [l_pupil, r_pupil], [l_iris, r_iris]


def get_pupil_and_iris_params(eye_params, frame, tracker=None, eye=None):
    x_min, y_min = 0, 0
    if eye is None:
        x_min = np.min(np.array(eye_params)[:, 0])
        x_max = np.max(np.array(eye_params)[:, 0])
        y_min = np.min(np.array(eye_params)[:, 1])
        y_max = np.max(np.array(eye_params)[:, 1])
        add = int((y_max - y_min) / 2)
        x_min, x_max = x_min - add, x_max + add
        y_min, y_max = y_min - add, y_max + add
        eye = frame[y_min: y_max, x_min: x_max]
        debug = False
    else:
        debug = True

    scale_factor = 120 / max(eye.shape)
    eye_scaled = cv2.resize(eye, dsize=None, fx=scale_factor, fy=scale_factor)
    eye_gray = cv2.cvtColor(eye_scaled, cv2.COLOR_BGR2GRAY)
    eye_gray = cv2.equalizeHist(eye_gray)

    if tracker is not None and tracker.prediction is not None:
        tracker_iris = tracker.prediction.copy()
        tracker_iris[0] = (tracker_iris[0] - y_min) * scale_factor
        tracker_iris[1] = (tracker_iris[1] - x_min) * scale_factor
        tracker_iris[2] *= scale_factor
        tracker_pupil = tracker.prediction_pupil * scale_factor
    else:
        tracker_iris, tracker_pupil = None, 0

    best_iris = get_best_iris(eye_gray, tracker_iris, debug)
    best_pupil = get_best_pupil(eye_gray, best_iris, tracker_pupil, debug) if best_iris is not None else None

    if debug:
        output = annotate_frame(eye_scaled, [best_pupil], [best_iris])
        cv2.imshow("output", output)
        cv2.waitKey(0)
    if best_iris is not None:
        best_iris = [f / scale_factor for f in best_iris]
        best_iris[0] += y_min
        best_iris[1] += x_min
    if best_pupil is not None:
        best_pupil = [f / scale_factor for f in best_pupil]
        best_pupil[0] += y_min
        best_pupil[1] += x_min

    return best_pupil, best_iris


def get_best_iris(eye, tracker_iris, debug, output=None):
    all_edges = (feature.canny(eye_gray, sigma=2) * 255).astype('uint8')
    if np.sum(all_edges > 0) / (all_edges.shape[0] * all_edges.shape[1]) < 0.03:
        all_edges = (feature.canny(eye_gray, sigma=1) * 255).astype('uint8')
    masked_edges = np.zeros_like(all_edges)
    if True:
        yc, xc = int(eye.shape[0] / 2), int(eye.shape[1] / 2)
        a_max = max(eye.shape[0] - yc, yc)
        a_min = int(a_max / 2)
        masked_edges[:, xc - a_max: xc + a_max] = all_edges[:, xc - a_max: xc + a_max]

    # try to find circle on masked edges
    best_iris = None
    hough_res = hough_circle(all_edges, list(range(a_min, a_max)))
    if hough_res is not None and len(hough_res) > 0:
        accums, cx, cy, radii = hough_circle_peaks(hough_res, list(range(a_min, a_max)), total_num_peaks=16)
        if len(radii) > 0:
            # weights = [(abs(cy[i] - yc) + abs(cx[i] - xc)) for i in range(len(cx))]
            weights = [eye[cy[i]][cx[i]] / 4 for i in range(len(cx))]
            if tracker_iris is not None:
                weights = [weights[i] + abs(radii[i] - tracker_iris[2]) +
                           max(abs(cx[i] - tracker_iris[1]), abs(cy[i] - tracker_iris[0]))
                           for i in range(len(radii))]
            iris_id = np.argmin(weights)

            best_iris = [cy[iris_id], cx[iris_id], radii[iris_id]]
            if debug and output is not None:
                cv2.imshow(f"masked_edges_iris", masked_edges)

    return best_iris


def get_best_pupil(eye, best_iris, tracker_pupil=0, debug=False):
    mask_iris = np.zeros_like(eye)
    mask_iris = cv2.circle(mask_iris, (int(best_iris[1]), int(best_iris[0])), int(best_iris[2] - 2), (1, 1, 1), -1)
    iris = cv2.equalizeHist(eye * mask_iris)
    threshold = np.quantile(iris[iris > 0], 0.5)
    mask_pupil = (iris < threshold).astype('uint8')
    pupil_edges = (feature.canny(iris, sigma=1) * 255).astype('uint8') * mask_pupil
    # mask_pupil = cv2.dilate(mask_pupil, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1).astype('uint8')
    # pupil_edges = all_edges * mask_iris * mask_pupil
    if debug:
        cv2.imshow(f"eye", eye)
        cv2.imshow(f"pupil_edges", pupil_edges)
        cv2.imshow(f"mask_pupil", mask_pupil * 255)
        cv2.imshow(f"iris", iris)

    best_pupil = None
    pupil_upperbound, pupil_lowerbound = int(best_iris[2]) - 3, int(best_iris[2] / 5)
    hough_res = hough_circle(pupil_edges, list(range(pupil_lowerbound, pupil_upperbound)))
    if hough_res is not None:
        accums, cx, cy, radii = hough_circle_peaks(hough_res, list(range(pupil_lowerbound, pupil_upperbound)),
                                                   total_num_peaks=16)
        weights = []
        for i in range(len(radii)):
            mask1 = np.zeros(eye.shape[:2], dtype=np.uint8)
            mask1 = cv2.circle(mask1, (cx[i], cy[i]), radii[i], (1, 1, 1), -1)
            mask1 = cv2.circle(mask1, (int(best_iris[1]), int(best_iris[0])), int(best_iris[2] - 2), (1, 1, 1), 1)
            cv2.imshow(f"pupil {i}", mask1 * 255)

            mean_intensity = np.sum(eye * mask1) / (radii[i] * radii[i] * np.pi)
            weights += [mean_intensity / 4 + abs(cy[i] - best_iris[0]) + abs(cx[i] - best_iris[1])]
            if tracker_pupil is not None and tracker_pupil > 0:
                weights[-1] += abs(radii[i] - tracker_pupil)
        if len(weights) > 0:
            pupil_id = np.argmin(weights)
            best_pupil = [cy[pupil_id], cx[pupil_id], radii[pupil_id]]
            # print(weights, best_iris[:2], best_pupil, eye[cy[pupil_id]][cx[pupil_id]] )
        else:
            best_pupil = None
    return best_pupil


def save_eye_patches(eyes, frame, frame_id):
    if eyes is None or eyes[0] is None:
        return
    for i, eye_params in enumerate(eyes):
        x_min = np.min(np.array(eye_params)[:, 0])
        x_max = np.max(np.array(eye_params)[:, 0])
        y_min = np.min(np.array(eye_params)[:, 1])
        y_max = np.max(np.array(eye_params)[:, 1])
        add = int((y_max - y_min) / 2)
        x_min, x_max = x_min - add, x_max + add
        y_min, y_max = y_min - add, y_max + add
        eye = frame[y_min: y_max, x_min: x_max]

        cv2.imwrite(f"results/patches/eye_{frame_id}_{i}.png", eye)


def annotate_frame(frame, pupils, irises, eyes=None, frame_label=None):
    if frame_label is not None:
        frame = cv2.putText(frame, str(frame_label), (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
     # if eyes is not None:
    #     for eye in eyes:
    #         if eye is not None:
    #             eye_hull = cv2.convexHull(eye)
    #             cv2.drawContours(frame, [eye_hull], -1, (0, 255, 0), 1)
    if pupils is None:
        return frame
    for pupil in pupils:
        if pupil is not None:
            yc, xc, a = [int(round(x)) for x in pupil]
            frame = cv2.circle(frame, (xc, yc), a, color=(255, 255, 255), thickness=1)
    for iris in irises:
        if iris is not None:
            yc, xc, a = [int(round(x)) for x in iris]
            frame = cv2.circle(frame, (xc, yc), a, color=(255, 0, 255), thickness=1)
    return frame


if __name__ == "__main__":   # for debug
    eye = cv2.imread('results/patches/eye_0_1.png')
    height = int(eye.shape[0])
    get_pupil_and_iris_params(None, frame=eye, eye=eye, tracker=None)
