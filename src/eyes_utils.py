from scipy.spatial import distance as dist
import cv2
import numpy as np
from skimage.transform import hough_ellipse, hough_circle, hough_circle_peaks
from skimage import feature
import math


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


def get_pupil_and_iris_params(eye_params, frame, eye=None, tracker=None, debug=False):
    if eye is None:
        x_min = np.min(np.array(eye_params)[:, 0])
        x_max = np.max(np.array(eye_params)[:, 0])
        y_min = np.min(np.array(eye_params)[:, 1])
        y_max = np.max(np.array(eye_params)[:, 1])
        add = int((y_max - y_min) / 2)
        x_min, x_max = x_min - add, x_max + add
        y_min, y_max = y_min - add, y_max + add
        eye = frame[y_min: y_max, x_min: x_max]

    eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    all_edges = (feature.canny(eye_gray, sigma=2) * 255).astype('uint8')
    if np.sum(all_edges > 0) / (all_edges.shape[0] * all_edges.shape[1]) < 0.03:
        all_edges = (feature.canny(eye_gray, sigma=1) * 255).astype('uint8')

    best_pupil, mask_rest, output = get_best_pupil(eye, all_edges, tracker, debug)
    best_iris = get_best_iris(eye, mask_rest * all_edges, best_pupil, tracker, debug, output)
    if not debug and best_pupil is not None:
        best_pupil[1] += y_min
        best_pupil[2] += x_min
    if not debug and best_iris is not None:
        best_iris[1] += y_min
        best_iris[2] += x_min
    if debug:
        cv2.imshow("output", output)
        cv2.waitKey(0)
    return best_pupil, best_iris


def get_best_pupil(eye, all_edges, tracker=None, debug=False):
    eye_hsv = cv2.cvtColor(eye, cv2.COLOR_RGB2HSV)
    threshold = np.quantile(eye_hsv[:,:,2], 0.04 )
    mask_pupil = (eye_hsv[:,:,2] < threshold).astype('uint8')
    mask_pupil = cv2.dilate(mask_pupil, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1).astype('uint8')
    # if tracker and tracker.eye_center_on_track and tracker.iris_on_track:
    #     print(tracker.eye_centers)
    #     mask_prev_pupil = np.zeros_like(mask_pupil)
    #     mask_prev_pupil = cv2.circle(mask_prev_pupil, (int(tracker.eye_centers[-1][0]), int(tracker.eye_centers[-1][1])),
    #                                  int(tracker.iris_diameters[-1]), (255,255,255), -1)
    #     mask_pupil = mask_prev_pupil * mask_pupil
    mask_rest = (mask_pupil == 0).astype('uint8')
    pupil_edges = mask_pupil * all_edges
    if debug:
        cv2.imshow(f"eye", eye)
        cv2.imshow(f"mask_pupil", (mask_pupil * 255).astype('uint8'))
        cv2.imshow(f"pupil_edges", pupil_edges)
        output = np.zeros_like((eye), dtype='uint8')
        output[:, :, 0] = all_edges
        output[:, :, 1] = all_edges
        output[:, :, 2] = all_edges
    else:
        output = None

    best_pupil = None
    if tracker is not None and tracker.eye_center_on_track:
        pupil_upperbound = max(int(tracker.iris_diameters[-1]), int(math.sqrt(1.5 * np.sum(mask_pupil)/3.14)))
    else:
        pupil_upperbound = int(math.sqrt(1.5 * np.sum(mask_pupil)/3.14))
    hough_res = hough_circle(pupil_edges, list(range(2, pupil_upperbound)))
    if hough_res is not None:
        accums, cx, cy, radii = hough_circle_peaks(hough_res, list(range(3, pupil_upperbound)), total_num_peaks=10)
        # print(radii)
        if tracker is not None and tracker.eye_center_on_track:
            weights = [1/(np.linalg.norm(np.array([cy[i], cx[i]])-tracker.eye_centers[-1])+1e-7) * radii[i] for i in range(len(radii))]
            # print(radii, weights)
            pupil_id = np.argmax(weights)
        else:
            pupil_id = np.argmax(radii)
        best_pupil = [0, cy[pupil_id], cx[pupil_id], radii[pupil_id], radii[pupil_id], 0]
        if debug:
            output = cv2.circle(output, (cx[pupil_id], cy[pupil_id]),  radii[pupil_id], color=(0, 100, 255), thickness=1)
            cv2.imshow("output", output)
    if best_pupil is None:
        result = hough_ellipse(pupil_edges)
        if result is None or len(result) == 0:
            return None, mask_rest, output
        result.sort(order='accumulator') #, accuracy=20, threshold=250, min_size=100, max_size=120)
        for res in result:
            if min(res[3], res[4]) / (max(res[3], res[4])) > 0.5:
                best_pupil = list(res)
        if best_pupil is mask_rest:
            best_pupil = list(result[-1])
        if min(best_pupil[2], best_pupil[3]) < 1:
            print('too small')
            return None, mask_rest, output
        if min(best_pupil[3], best_pupil[4]) / ( max(best_pupil[3], best_pupil[4])) < 0.5:
            print('pupil not round')
            return None, mask_rest, output
        if debug:
            yc, xc, a, b = [int(round(x)) for x in best_pupil[1:5]]
            output = cv2.ellipse(output, (xc, yc), (b, a), best_pupil[5], 0, 360, color=(0, 255, 255), thickness=1)
            cv2.imshow("output", output)

    return best_pupil, mask_rest, output


def get_best_iris(eye, rest_edges, best_pupil, tracker, debug, output):
    if best_pupil is not None:
        yc, xc, a, b = [int(round(x)) for x in best_pupil[1:5]]
    else:
        yc, xc, a, b = int(eye.shape[0] / 2), int(eye.shape[1]/2), 5, 5
    iris_diam_upperbound = max(eye.shape[0] - yc, yc)
    rest_edges[:,:xc - iris_diam_upperbound] = 0
    rest_edges[:,xc + iris_diam_upperbound:] = 0

    best_iris = None
    hough_res = hough_circle(rest_edges, list(range(max(a,b), iris_diam_upperbound)))
    if hough_res is not None and len(hough_res) > 0:
        accums, cx, cy, radii = hough_circle_peaks(hough_res, list(range(max(a,b), iris_diam_upperbound)), total_num_peaks=13)
        # print('iris radii', radii)
        if len(radii) > 0:
            if tracker is not None and tracker.iris_on_track:
                weights = [np.linalg.norm(np.array(radii[i]) - tracker.iris_diameters[-1]) * \
                           np.linalg.norm(np.array([cy[i], cx[i]]) - np.array([yc, xc])) for i in
                           range(len(radii))]
                iris_id = np.argmin(weights)
            else:
                dist = np.array([np.linalg.norm(np.array([cy[i], cx[i]]) - np.array([yc, xc])) for i in range(len(cx))])
                iris_id = np.argmin(dist)

            # if dist[min_id] < 4:
            best_iris = [0, int(cy[iris_id]), int(cx[iris_id]), int(radii[iris_id]),int(radii[iris_id]), 0]
            if debug:
                cv2.imshow(f"rest_edges", rest_edges)
                output = cv2.circle(output, (int(cx[iris_id]), int(cy[iris_id])), int(radii[iris_id]),
                                     color=(255, 255, 0), thickness=1)
    if best_iris is None:
        result = hough_ellipse(rest_edges, min_size=max(a,b), max_size=iris_diam_upperbound)
        if result is None or len(result) == 0:
            print('no iris found')
            return None
        result.sort(order='accumulator') #, accuracy=20, threshold=250, min_size=100, max_size=120)
        dist = []  # np.array([np.linalg.norm(np.array([res[1], res[2]]) - np.array([yc, xc])) for res in result])
        for res in result:
            if min(res[3], res[4]) / (max(res[3], res[4])) > 0.6:
                dist.append(np.linalg.norm(np.array([res[1], res[2]]) - np.array([yc, xc])))
            else:
                dist.append(1000)

        min_id = np.argmin(dist)
        best_iris = list(result[min_id])
        if debug:
            output = cv2.ellipse(output, (xc, yc), (b, a), best_iris[5], 0, 360, color=(255, 0, 255), thickness=1)

    return best_iris


if __name__ == "__main__":
    eye = cv2.imread('results/patches/eye_152_1.png') #103_1, 121_0, 131_0, 142_0
    height = int(eye.shape[0])
    # eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    # # all_edges = cv2.Canny(eye_gray, 60, 120)
    # all_edges = (feature.canny(eye_gray, sigma=1) * 255).astype('uint8')
    # cv2.imshow(f"edges, sigma1", all_edges)
    # print('default', np.sum(all_edges>1) / (all_edges.shape[0] * all_edges.shape[1]))
    # all_edges = (feature.canny(eye_gray, sigma=2) * 255).astype('uint8')
    # cv2.imshow(f"edges, sigma2", all_edges)
    # print('sigma1', np.sum(all_edges>1) / (all_edges.shape[0] * all_edges.shape[1]))
    # all_edges = (feature.canny(eye_gray, sigma=3)* 255).astype('uint8')
    # cv2.imshow(f"edges, sigma3", all_edges)
    # print('sigma3', np.sum(all_edges>1) / (all_edges.shape[0] * all_edges.shape[1]))
    # cv2.waitKey(0)

    get_pupil_and_iris_params(None, frame=eye, eye=eye, tracker=None, debug=True)
    # process_eye(eye)
