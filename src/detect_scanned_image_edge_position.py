import cv2
import numpy as np


def detect_scanned_image_edge_position(image: np.array, args):
    def calcOrthant(x_c: int, y_c: int, x: int, y: int):
        """calculate orthant from 2 point

        Args:
            x_c (int): center position x coordinate
            y_c (int): center position y coordinate
            x (int): x coordinate
            y (int): y coordinate

        Returns:
            int: orrthant(1:right top, 2:left top, 3: left bottom, 4:right bottom)
        """
        return (
            1
            if (x - x_c > 0) and (y - y_c <= 0)
            else 2
            if (x - x_c <= 0) and (y - y_c <= 0)
            else 3
            if (x - x_c <= 0) and (y - y_c > 0)
            else 4
            if (x - x_c > 0) and (y - y_c > 0)
            else None
        )

    # convert input image to hsl and make saturation max
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float) / 255.0
    gray_image = (
        (hsv_image[..., 1] > 0.3)
        | (
            (
                (hsv_image[..., 2] > 0.3)
                if args.background_is_black
                else (hsv_image[..., 2] < 0.7)
            )
        )
    ).astype(np.uint8) * 255
    # binarize
    _, binarized_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # if background is white, reverse value
    if not args.background_is_black:
        binarized_image = np.bitwise_not(binarized_image)
    # find outer boundary
    contours, _ = cv2.findContours(
        binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # leave only the largest area
    arc_lengths = [cv2.arcLength(contour, True) for contour in contours]
    max_arc_length = max(arc_lengths)
    max_index = arc_lengths.index(max(arc_lengths))
    contours = [contours[max_index]]

    # decrease points num of contours
    while contours[0].shape[0] > 4:
        epsilon = 0.1 * max_arc_length
        contours = [cv2.approxPolyDP(contours[0], epsilon, True)]

    if contours[0].shape[0] != 4:
        print("[ERROR] border detection failed")
        return None, None

    # draw debug image
    contours = list(
        filter(lambda x: cv2.contourArea(x) > int(sum(image.shape[:2]) / 2), contours)
    )
    # # draw border
    result_image = image.copy()
    result_image = cv2.drawContours(
        result_image, contours, -1, color=(0, 255, 255), thickness=2
    )
    # # draw bounding box
    bbox_x_left = int(np.min(contours[0][..., 0]))
    bbox_y_top = int(np.min(contours[0][..., 1]))
    bbox_x_right = int(np.max(contours[0][..., 0]))
    bbox_y_bottom = int(np.max(contours[0][..., 1]))
    bbox_width = bbox_x_right - bbox_x_left
    bbox_height = bbox_y_bottom - bbox_y_top
    cv2.rectangle(
        result_image,
        (bbox_x_left, bbox_y_top),
        (bbox_x_right, bbox_y_bottom),
        (0, 255, 0),
        thickness=2,
    )

    # create result data
    x_center = np.average(contours[0][..., 0])
    y_center = np.average(contours[0][..., 1])

    orthants = [
        calcOrthant(x_center, y_center, contours[0][i, 0, 0], contours[0][i, 0, 1])
        for i in range(4)
    ]
    boundary_x_right_top, boundary_y_right_top = contours[0][
        orthants.index(1), 0
    ].tolist()
    boundary_x_left_top, boundary_y_left_top = contours[0][
        orthants.index(2), 0
    ].tolist()
    boundary_x_left_bottom, boundary_y_left_bottom = contours[0][
        orthants.index(3), 0
    ].tolist()
    boundary_x_right_bottom, boundary_y_right_bottom = contours[0][
        orthants.index(4), 0
    ].tolist()

    result_data = {
        "boundary_x_left_top": str(boundary_x_left_top),
        "boundary_y_left_top": str(boundary_y_left_top),
        "boundary_x_right_top": str(boundary_x_right_top),
        "boundary_y_right_top": str(boundary_y_right_top),
        "boundary_x_right_bottom": str(boundary_x_right_bottom),
        "boundary_y_right_bottom": str(boundary_y_right_bottom),
        "boundary_x_left_bottom": str(boundary_x_left_bottom),
        "boundary_y_left_bottom": str(boundary_y_left_bottom),
        "bbox_x_left": str(bbox_x_left),
        "bbox_y_top": str(bbox_y_top),
        "bbox_width": str(bbox_width),
        "bbox_height": str(bbox_height),
    }

    # draw position info
    cv2.putText(
        result_image,
        "Boundary Points(x,y)="
        + f"({boundary_x_left_top},{boundary_y_left_top})"
        + f"({boundary_x_right_top},{boundary_y_right_top})"
        + f"({boundary_x_right_bottom},{boundary_y_right_bottom})"
        + f"({boundary_x_left_bottom},{boundary_y_left_bottom})",
        org=(0, 40),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=0.6,
        color=(0, 128, 128),
        thickness=1,
        lineType=8,
        bottomLeftOrigin=False,
    )
    cv2.putText(
        result_image,
        "Bounding Box(x,y,w,h)="
        + f"({bbox_x_left},{bbox_y_top},{bbox_width},{bbox_height})",
        org=(0, 80),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=0.6,
        color=(128, 128, 0),
        thickness=1,
        lineType=8,
        bottomLeftOrigin=False,
    )

    # crop image
    cropped_image = image[bbox_y_top:bbox_y_bottom, bbox_x_left:bbox_x_right].copy()

    return (result_image, cropped_image), result_data
