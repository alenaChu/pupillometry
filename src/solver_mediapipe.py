import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, List

# debug only
import matplotlib.pyplot as plt
import time


# manually found IDs of eyes contours markdowns
LEFT_EYE_LANDMARKS_ID = [133, 155, 154, 153, 145, 144, 7, 130, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_LANDMARKS_ID = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]


def convert_proportions_to_pixels(
        x_in: float, y_in: float, img_width: int, img_height: int) -> Tuple[int, int]:
    """Converts proportion values (x, y) to pixels coordinates.

    Parameters
    ----------
    x_in : float
        X coordinate of point in proportion of img_width
    y_in : float
        Y coordinate of point in proportion of img_height
    img_width : int
    img_height : int

    Returns
    -------
    x_px : int
        X coordinate of point in pixels in range [0, img_width - 1]
    y_px : int
        Y coordinate of point in pixels in range [0, img_height - 1]
    """

    # filter incorrect values
    x_in = max(0.0, x_in)
    x_in = min(x_in, 1.0)
    y_in = max(0.0, y_in)
    y_in = min(y_in, 1.0)

    # coordinates of point could be in range [ 0, image_size )
    x_px = min(int(x_in * img_width), img_width - 1)
    y_px = min(int(y_in * img_height), img_height - 1)
    return x_px, y_px


def add_mask_by_points(img: np.ndarray, points: np.ndarray, margin: int = 10) -> np.ndarray:
    """Cut from img rectangular roi with margins. And append 4th channel with a binary mask.

    Parameters
    ----------
    img : np.ndarray
        Image from web cam or video file. np.array with 3 channels and 8bit values
    points : np.ndarray
        2D numpy array with eye contour points coordinates. Each row is point, first column is X, second is Y.
    margin : int
        Extra space for each side of roi in pixels.

    Returns
    -------
    roi : np.ndarray
        Rectangular part of the img array 3 color channels of image and 4th channel with binary mask
        filled with 0 outside eye contour.
    """

    # roi of an eye in img coordinates
    top_corner = np.asarray([max(points[:, 0].min() - margin, 0), max(points[:, 1].min() - margin, 0)])
    bottom_corner = np.asarray([min(points[:, 0].max() + margin, img.shape[1] - 1),
                                min(points[:, 1].max() + margin, img.shape[0] - 1)])

    # points of mask's border in roi coordinates
    mask_points = points - top_corner
    # transform into cv2 style contour
    contour = mask_points.reshape((-1, 1, 2)).astype(np.int32)
    # create and fill mask with ones inside polygon
    mask = np.zeros((bottom_corner[1] - top_corner[1], bottom_corner[0] - top_corner[0]), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (1, 1, 1), -1)
    # cut roi from image and apply mask
    roi = img[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0]].copy()
    return np.dstack([roi, mask])


def find_pupil(image: np.ndarray) -> np.ndarray:
    """Localisation of black and round eye pupil.

    Parameters
    ----------
    image : np.ndarray
        Rectangular ROI image of an eye. 4 channel [B, G, R, binary mask] ndarray uint8.
        Mask is binary 2D image with 1 and 0. Skin and eye lid are 0, sclera and iris and pupil are 1.

    Returns
    -------
    np.ndarray
        Same width and height as input image ndarray with 3 duplicated channels (BGR).
        Contains binary mask of pupil with values 255 and 0 elsewhere.
        When pupil wasn't found only 0 values is returned.
    """
    # extract mask and data; convert uint8 BGR data to gray floating point
    mask = image[..., 3]
    value = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2HSV)[..., -1].astype(np.float)
    img = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float)

    # apply mask
    img[mask == 0.0] = 0.0
    value[mask == 0.0] = 0.0
    # increasing contrast
    img = img * value
    # alias interested data from image as eye
    eye = img[mask == 1.0]
    # normalize data and return it to img
    eye = (eye - eye.mean()) / eye.std()
    img[mask == 1] = eye

    # create mask for inpainting
    inpaint_mask = np.zeros(img.shape, dtype=np.uint8)
    inpaint_mask[img > 1.0] = 1
    # inpaint and fix created salt noise
    img = cv2.inpaint(img.astype(np.float32), inpaint_mask, 3, cv2.INPAINT_TELEA)
    img = cv2.medianBlur(img, ksize=3)

    # find (y, x) coordinate of minimum value on the image
    argmin = np.argmin(img) // img.shape[1], np.argmin(img) % img.shape[1]

    mask_flooded = np.zeros(img.shape, np.uint8)

    start_flood_fill_mask(img, mask_flooded, argmin[1], argmin[0], threshold=0.33)

    mask_flooded[mask_flooded == 2] = 0
    kernel = np.ones((5, 5), np.uint8)
    mask_flooded = cv2.morphologyEx(mask_flooded, cv2.MORPH_CLOSE, kernel)

    if check_pupil(mask_flooded):
        return np.dstack([mask_flooded, mask_flooded, mask_flooded]) * 255
    else:
        zeros = np.zeros_like(mask_flooded)
        return np.dstack([zeros, zeros, zeros])


def apply_flood_fill(image, mask, x, y, target_color, step, threshold=0.1):

    if step < 500:

        is_image_border_met = (x < 0) or (y < 0) or (x >= image.shape[1]) or (y >= image.shape[0])
        is_different_color = abs(image[y, x] / target_color - 1.0) >= threshold

        if is_image_border_met or is_different_color:
            mask[y, x] = 2
            return

        mask[y, x] = 1
        target_color = update_average(target_color, step, image[y, x])

        if x < mask.shape[1] - 1 and mask[y, x + 1] == 0:
            apply_flood_fill(image, mask, x + 1, y, target_color, step + 1, threshold)
        if x < mask.shape[1] - 1 and y < mask.shape[0] - 1 and mask[y + 1, x + 1] == 0:
            apply_flood_fill(image, mask, x + 1, y + 1, target_color, step + 1, threshold)
        if y < mask.shape[0] - 1 and mask[y + 1, x] == 0:
            apply_flood_fill(image, mask, x, y + 1, target_color, step + 1, threshold)
        if y < mask.shape[0] - 1 and x > 0 and mask[y + 1, x - 1] == 0:
            apply_flood_fill(image, mask, x - 1, y + 1, target_color, step + 1, threshold)
        if x > 0 and mask[y, x - 1] == 0:
            apply_flood_fill(image, mask, x - 1, y, target_color, step + 1, threshold)
        if x > 0 and y > 0 and mask[y - 1, x - 1] == 0:
            apply_flood_fill(image, mask, x - 1, y - 1, target_color, step + 1, threshold)
        if y > 0 and mask[y - 1, x] == 0:
            apply_flood_fill(image, mask, x, y - 1, target_color, step + 1, threshold)
        if y > 0 and x < mask.shape[1] - 1 and mask[y - 1, x + 1] == 0:
            apply_flood_fill(image, mask, x + 1, y - 1, target_color, step + 1, threshold)


def start_flood_fill_mask(image, mask, x, y, threshold=0.1):
    target_color = image[y, x]
    apply_flood_fill(image, mask, x, y, target_color, 0, threshold)


def update_average(average, size, new_value):
    return (size * average + new_value) / (size + 1)


def construct_frame(main_frame: np.ndarray,
                    eyes: List[np.ndarray],
                    pupils: List[np.ndarray],
                    areas: np.ndarray,
                    areas_plot_raw: np.ndarray,
                    areas_plot_sliding: np.ndarray,
                    margin: int = 5,
                    height: int = 720) -> np.ndarray:
    # total number of found eyes
    eyes_num = len(eyes)

    # calc total video size as 23:9 ratio from its height and create empty frame
    size = (height, int((height//9)*23))
    output = np.zeros([size[0], size[1], 3], dtype=np.uint8)

    # scale main video to fit it by height
    scaled_main_frame = cv2.resize(main_frame, dsize=(int(main_frame.shape[1] * (height/main_frame.shape[0])), height))
    output[:scaled_main_frame.shape[0], :scaled_main_frame.shape[1]] = scaled_main_frame

    # start of columns with additional data X coordinate
    data_cols_start = scaled_main_frame.shape[1] + margin * 2
    # calc total width free space to place 3 columns: eye, pupil and some numbers and 2 margins between
    data_columns_free_width = size[1] - data_cols_start - 2 * margin
    col_w = data_columns_free_width // 3

    # calc X coordinate of centers of each column
    col_c = [data_cols_start + int(0.5 * col_w),
             data_cols_start + int(1.5 * col_w) + margin,
             data_cols_start + int(2.5 * col_w) + 2 * margin]
    # calc Y coordinate of centers of each row
    # row_c = [int(0.5 * col_w), int(1.5 * col_w) + margin, int(2.5 * col_w) + 2 * margin]

    text = []
    for area in areas:
        cell = np.zeros([col_w, col_w, 3], dtype=np.uint8)
        cv2.putText(cell, str(area), (0, 2 * (col_w // 3), ),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        text.append(cell)

    # place eyes as square grid
    for i in range(eyes_num):
        row = int((i + 0.5) * col_w) + i * margin
        for col, data in zip(col_c, [eyes, pupils, text]):
            y0 = row - data[i].shape[0] // 2
            y1 = y0 + data[i].shape[0]
            x0 = col - data[i].shape[1] // 2
            x1 = x0 + data[i].shape[1]
            output[y0:y1, x0:x1] = data[i][..., :3]

    canvas_start = [eyes_num * col_w + eyes_num * margin, data_cols_start + 30]
    canvas_size = [output.shape[0] - canvas_start[0], output.shape[1] - canvas_start[1]]
    canvas = np.zeros(canvas_size)

    # TODO: add canvas x and y titles
    for i in range(eyes_num):
        ax_size = [canvas_size[0] // eyes_num, canvas_size[1]]
        ax_start = [i * ax_size[0], 0]
        ax = np.zeros([ax_size[0], ax_size[1]], dtype=np.uint8)

        for j, d in enumerate([areas_plot_raw, areas_plot_sliding]):
            t = np.arange(0, d.shape[-1], 1)
            # fitting to canvas width
            t = ax_size[1] * (t / d.shape[-1])
            curve = ax_size[0] * (d[i, :] / d[i, :].max())
            pts = np.vstack((t, curve)).T.astype(np.int)
            cv2.polylines(ax, [pts], False, 120 * (j + 1), thickness=2+j)

        canvas[ax_start[0]: ax_start[0] + ax_size[0],
               ax_start[1]: ax_start[1] + ax_size[1]] = np.flipud(ax)

    canvas = np.dstack([canvas, canvas, canvas]).astype(np.uint8)
    output[canvas_start[0]:canvas_start[0]+canvas_size[0],
           canvas_start[1]:canvas_start[1]+canvas_size[1]] = canvas

    # TODO: this is temprorary titles, will do it right later
    cv2.putText(output, "eye roi", (col_c[0] - 60, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.putText(output, "pupil mask", (col_c[1] - 85, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.putText(output, "area, px", (col_c[2] - 50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    canvas_title = np.zeros([30, 150], dtype=np.uint8)
    # cv2.putText(canvas_title, "left", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1)
    # output[canvas_start[0]:canvas_start[0]+150, canvas_start[1]-30:canvas_start[1]] = np.flipud(
    #     np.dstack([canvas_title.T, canvas_title.T, canvas_title.T]))
    #
    # canvas_title = np.zeros([30, 150], dtype=np.uint8)
    # cv2.putText(canvas_title, "right", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1)
    # output[-150:, canvas_start[1] - 30:canvas_start[1]] = np.flipud(
    #     np.dstack([canvas_title.T, canvas_title.T, canvas_title.T]))

    return output


def calc_pupil_area(image):
    return int(np.sum(image) / 255)


def find_min_diameter(image: np.ndarray) -> int:
    """Finds and returns length of smaller side of circumscribed rectangle.

    Parameters
    ----------
    image : np.ndarray
        1 channel uint8 2D ndarray.
        Binary image with only 1 object.
    Returns
    -------
    int
        Length of smaller side of circumscribed rectangle.
    """
    nonzero = np.nonzero(image)
    y_min = nonzero[0].min()
    y_max = nonzero[0].max()
    x_min = nonzero[1].min()
    x_max = nonzero[1].max()

    height = y_max - y_min
    width = x_max - x_min
    return min(height, width) + 1


def create_circle_kernel(size: int) -> np.ndarray:
    """Create and return square ndarray (inputted size) with inscribed circle filled with ones.

    Parameters
    ----------
    size : int
        Size of returning square kernel. Same as circle diameter.

    Returns
    -------
    circle : np.ndarray
        1 channel uint8 2D ndarray with shape = [size, size]
        Square binary ndarray filled with 1 (circle) and 0 (background) with inputted size.
    """
    circle = np.zeros([size, size], dtype=np.uint8)
    fill_box_with_circle(circle)
    return circle


def fill_box_with_circle(image: np.ndarray) -> None:
    """Draw circle inscribed in inputted image.

    Parameters
    ----------
    image : np.ndarray
        1 channel uint8 2D ndarray with square shape

    Returns
    -------
    None
    """
    assert image.shape[0] == image.shape[1]
    diam = image.shape[0]
    for i in range(diam):
        for j in range(diam):
            x = (i - (diam - 1) / 2)
            y = (j - (diam - 1) / 2)
            r = diam / 2
            if x**2 + y**2 < r**2:
                image[j, i] = 1


def check_pupil(image) -> bool:
    # area conditions
    pupil_area = np.sum(image)
    if (pupil_area < 20) or (pupil_area > 0.1 * (image.shape[0] * image.shape[1])):
        return False

    # circle-ish condition
    diam = find_min_diameter(image)
    circle = create_circle_kernel(diam)
    circle_area = np.sum(circle)

    matrix_convolved = cv2.filter2D(image.astype(np.float), -1, circle.astype(np.float))
    # best area of overlap pupil and circle is max value of convolution (when both images are binary)
    intersection = np.max(matrix_convolved)
    union = circle_area + pupil_area - intersection
    iou = np.sum(intersection) / np.sum(union)

    # debug visualisation util
    """
    circle_on_image = np.zeros_like(image)
    best_pose_idx = np.argmax(matrix_convolved)
    circle_on_image[best_pose_idx // image.shape[1] - circle.shape[0]//2:
                    best_pose_idx // image.shape[1] - circle.shape[0]//2 + circle.shape[0], 
                    best_pose_idx % image.shape[1] - circle.shape[1]//2:
                    best_pose_idx % image.shape[1] - circle.shape[1]//2 + circle.shape[1]] = circle
    plt.imshow(np.hstack([image, 2 * matrix_convolved / intersection, image + circle_on_image]))
    plt.show()
    """

    # magic threshold ^_^ could be optimized or automatized
    if iou < 0.667:
        return False

    return True


def solve(vid_filename: str):
    stopped_by_user = False
    # init and set google mediapipe face mesh detection and drawing settings
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    # cv2 I/O initialization
    # cap = cv2.VideoCapture(vid_filename)
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(r"../data/085435.avi")

    # TODO: make output video size variable value consistent with construct_frame(..., height=720)
    out = cv2.VideoWriter(vid_filename[:-4] + "_result.avi",
                          cv2.VideoWriter_fourcc(*"MJPG"), 20, (1840, 720))

    curve_raw = np.zeros([2, 100], dtype=np.uint16)
    curve_mean = np.zeros([2, 100], dtype=np.uint16)

    # performance measurement
    time_for_process = []

    while cap.isOpened():
        success, frame_input = cap.read()
        if not success:
            print("End of the file.")
            break

        # google mediapipe has RGB input
        image = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        eyes_list = []
        pupils_list = []
        areas_list = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_landmarks_px = []
                right_eye_landmarks_px = []
                # filling lists of found eyes borders with coordinates in px
                for eye_landmarks, eye_pixels in zip([LEFT_EYE_LANDMARKS_ID, RIGHT_EYE_LANDMARKS_ID],
                                                     [left_eye_landmarks_px, right_eye_landmarks_px]):
                    for point_id in eye_landmarks:
                        eye_pixels.append(convert_proportions_to_pixels(face_landmarks.landmark[point_id].x,
                                                                        face_landmarks.landmark[point_id].y,
                                                                        image.shape[1], image.shape[0]))
                    eye_pixels = np.asarray(eye_pixels)
                    # cut eyes arrays from image
                    eyes_list.append(add_mask_by_points(image, eye_pixels))

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        if eyes_list:

            # debug time measurement
            time_start = time.time()

            pupils_list = []
            for eye in eyes_list:
                pupils_list.append(find_pupil(eye))

            areas_list = []
            for pupil in pupils_list:
                areas_list.append(calc_pupil_area(pupil))
            areas_list = np.asarray(areas_list)

            curve_raw = np.roll(curve_raw, -1, axis=1)
            curve_raw[:, -1] = areas_list

            curve_mean = np.roll(curve_mean, -1, axis=1)
            mean_area_list = areas_list.copy()
            for i in range(len(areas_list)):
                if areas_list[i] == 0:
                    mean_area_list[i] = curve_mean[i, -2]
                elif areas_list[i].astype(np.float) / curve_mean[i, -2].astype(np.float) > 1.2:
                    mean_area_list[i] = int(areas_list[i] * 1.2)
                elif areas_list[i].astype(np.float) / curve_mean[i, -2].astype(np.float) < 0.8:
                    mean_area_list[i] = int(curve_mean[i, -2] * 0.8)
            curve_mean[:, -1] = np.mean(np.hstack([mean_area_list[:, np.newaxis], curve_mean[:, -5:]]), axis=1)

            # debug time measurement
            time_end = time.time()
            time_for_process.append(time_end - time_start)

        output = construct_frame(image, eyes_list, pupils_list, areas_list, curve_raw, curve_mean, margin=5, height=720)

        out.write(output)

        cv2.imshow('MediaPipe FaceMesh', output)
        key = cv2.waitKey(5)
        if (key == ord('q')) or (key == 27):
            stopped_by_user = True
            break
        if key == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    out.release()

    return stopped_by_user


if __name__ == '__main__':

    if len(sys.argv) == 2:
        vid_path_generator = [Path(sys.argv[1])]  # actually list not a generator ^_^
    else:
        cwd_path = Path.cwd()
        vid_path_generator = cwd_path.parent.glob('data_new/*.avi')

    if vid_path_generator:
        for vid_path in vid_path_generator:
            manual_stop = solve(str(vid_path))
            if manual_stop:
                break
    else:
        print("no '../data/*.avi' were found")

    print("finished")
