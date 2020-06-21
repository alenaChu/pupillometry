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

    l_eye_AR = eye_aspect_ratio(l_eye)
    r_eye_AR = eye_aspect_ratio(r_eye)
    # average the eye aspect ratio together for both eyes
    eye_ar = (l_eye_AR + r_eye_AR) / 2.0

    return [l_eye, r_eye], [l_pupil, r_pupil], [l_iris, r_iris], eye_ar


def get_pupil_and_iris_params(eye_params, frame, tracker=None, eye=None):
    if eye is None:
        x_min = np.min(np.array(eye_params)[:, 0])
        x_max = np.max(np.array(eye_params)[:, 0])
        y_min = np.min(np.array(eye_params)[:, 1])
        y_max = np.max(np.array(eye_params)[:, 1])
        add = int((y_max - y_min) / 2)
        x_min, x_max = x_min - add, x_max + add
        y_min, y_max = y_min - add, y_max + add
        eye = frame[y_min: y_max, x_min: x_max]
        debug=False
    else:
        debug = True

    if tracker is not None and tracker.prediction is not None:
        tracker_iris = tracker.prediction.copy()
        tracker_pupil = tracker.prediction_pupil
        tracker_iris[0] -= y_min
        tracker_iris[1] -= x_min
    else:
        tracker_iris, tracker_pupil = None, 0
    eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    all_edges = (feature.canny(eye_gray, sigma=2) * 255).astype('uint8')
    if np.sum(all_edges > 0) / (all_edges.shape[0] * all_edges.shape[1]) < 0.03:
        all_edges = (feature.canny(eye_gray, sigma=1) * 255).astype('uint8')

    best_iris = get_best_iris(eye, all_edges, tracker_iris, debug)
    best_pupil = get_best_pupil(eye, all_edges, best_iris, tracker_pupil, debug) if best_iris is not None else None

    if not debug:
        if best_iris is not None:
            best_iris[1] += y_min
            best_iris[2] += x_min
        if best_pupil is not None:
            best_pupil[1] += y_min
            best_pupil[2] += x_min
    else:
        output = np.zeros_like(eye)
        output[:, :, 0] = all_edges
        output[:, :, 1] = all_edges
        output[:, :, 2] = all_edges
        if best_iris is not None:
            output = cv2.circle(output, (int(best_iris[2]), int(best_iris[1])), int(best_iris[3]),
                                color=(255, 255, 0), thickness=1)
        if best_pupil is not None:
            output = cv2.circle(output, (int(best_pupil[2]), int(best_pupil[1])), int(best_pupil[3]),
                                color=(0, 100, 255), thickness=1)

        cv2.imshow("output", output)
        cv2.waitKey(0)

    return best_pupil, best_iris


def get_best_iris(eye, all_edges, tracker_iris, debug, output=None):
    masked_edges = np.zeros_like(all_edges)
    if True:
        yc, xc = int(eye.shape[0] / 2), int(eye.shape[1]/2)
        a_max = max(eye.shape[0] - yc, yc)
        a_min = int(a_max / 2)
        masked_edges[:,xc - a_max: xc + a_max] = all_edges[:,xc - a_max: xc + a_max]

    # try to find circle on masked edges
    best_iris = None
    hough_res = hough_circle(all_edges, list(range(a_min, a_max)))
    if hough_res is not None and len(hough_res) > 0:
        accums, cx, cy, radii = hough_circle_peaks(hough_res, list(range(a_min, a_max)), total_num_peaks=4)
        if len(radii) > 0:
            if tracker_iris is not None:
                weights = [abs(radii[i] - tracker_iris[-1]) * \
                                np.linalg.norm(np.array([cy[i], cx[i]]) - np.array([yc, xc]))
                                for i in range(len(radii))]
                # weights = [abs(radii[i] - tracker_iris[-1]) for i in range(len(radii))]
                iris_id = np.argmin(weights)
            else:
                dist = np.array([np.linalg.norm(np.array([cy[i], cx[i]]) - np.array([yc, xc])) for i in range(len(cx))])
                iris_id = np.argmin(dist)

            best_iris = [0, int(cy[iris_id]), int(cx[iris_id]), int(radii[iris_id]),int(radii[iris_id]), 0]
            if debug and output is not None:
                cv2.imshow(f"masked_edges_iris", masked_edges)

    return best_iris


def get_best_pupil(eye, all_edges, best_iris, tracker_pupil=0, debug=False):
    mask_iris = np.zeros_like(all_edges)
    mask_iris = cv2.circle(mask_iris, (int(best_iris[2]), int(best_iris[1])),
                                     int(best_iris[3] - 2), (1,1,1), -1)
    iris = np.zeros_like(eye)
    iris[:,:,0] = eye[:,:,0] * mask_iris
    iris[:,:,1] = eye[:,:,1] * mask_iris
    iris[:,:,2] = eye[:,:,2] * mask_iris
    eye_hsv = cv2.cvtColor(iris, cv2.COLOR_RGB2HSV)[:,:,2]
    threshold = np.quantile(eye_hsv[eye_hsv > 0], 0.5)
    mask_pupil = (eye_hsv < threshold).astype('uint8')
    # mask_pupil = cv2.dilate(mask_pupil, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1).astype('uint8')

    pupil_edges = all_edges * mask_iris
    pupil_edges = pupil_edges * mask_pupil
    if debug:
        cv2.imshow(f"eye", eye)
        cv2.imshow(f"all_edges", all_edges)
        cv2.imshow(f"pupil_edges", pupil_edges)
        cv2.imshow(f"mask_iris", mask_iris * 255)
        cv2.imshow(f"eye_hsv", eye_hsv)

    best_pupil = None
    pupil_upperbound = int(best_iris[3]) - 3
    hough_res = hough_circle(pupil_edges, list(range(2, pupil_upperbound)))
    if hough_res is not None:
        accums, cx, cy, radii = hough_circle_peaks(hough_res, list(range(3, pupil_upperbound)), total_num_peaks=10)
        # print(radii, tracker_pupil)
        weights = [np.linalg.norm(np.array([cy[i], cx[i]])-best_iris[1:3]) * (radii[i] - tracker_pupil)
                   for i in range(len(radii))]
        # TODO: add weight - iou with darker area pupil intensity
        # print(radii, weights)
        if len(weights) > 0:
            pupil_id = np.argmax(weights)
            best_pupil = [0, cy[pupil_id], cx[pupil_id], radii[pupil_id], radii[pupil_id], 0]
        else:
            best_pupil = None
    return best_pupil


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    #  eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    res = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return res


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


def annotate_frame(frame, eyes, pupils, irises, eye_ar, frame_label=None):
    if frame_label is not None:
        frame = cv2.putText(frame, str(frame_label), (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    # if eye_ar is not None:
    #     frame = cv2.putText(frame, f"eye open={np.round(eye_ar,2)}", (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.1) + 50),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    # if eyes is not None:
    #     for eye in eyes:
    #         if eye is not None:
    #             eye_hull = cv2.convexHull(eye)
    #             cv2.drawContours(frame, [eye_hull], -1, (0, 255, 0), 1)
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


if __name__ == "__main__":   # for debug
    eye = cv2.imread('results/patches/eye_37_0.png')
    height = int(eye.shape[0])
    get_pupil_and_iris_params(None, frame=eye, eye=eye, tracker=None)
