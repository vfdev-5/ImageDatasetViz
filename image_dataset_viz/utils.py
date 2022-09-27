import numpy as np


def bbox_to_points(bbox_xyxy):
    """Helper method to convert bounding box as list/tuple of [x1, y1, x2, y2] into points `ndarray` of shape (N, 2)

    Args:
        bbox_xyxy (list or tuple or ndarray): bounding box as list/tuple of [x1, y1, x2, y2]

    Returns:
        ndarray of points

    """
    assert (
        isinstance(bbox_xyxy, (list, tuple, np.ndarray)) and len(bbox_xyxy) == 4
    ), "Argument bbox_xyxy should be a list/tuple of [x1, y1, x2, y2]"

    return np.array(
        [
            [bbox_xyxy[0], bbox_xyxy[1]],
            [bbox_xyxy[2], bbox_xyxy[1]],
            [bbox_xyxy[2], bbox_xyxy[3]],
            [bbox_xyxy[0], bbox_xyxy[3]],
        ]
    )


def xywh_to_xyxy(xywh):
    """Helper method to transform bounding box of type [x1, y1, width, height] into [x1, y1, x2, y2]

    Args:
        xywh (list or tuple): bounding box of type [x1, y1, width, height]

    Returns:
        list [x1, y1, x2, y2]

    """
    assert (
        isinstance(xywh, (list, tuple, np.ndarray)) and len(xywh) == 4
    ), "Argument xywh should be a list/tuple of [x1, y1, width, height]"
    x1, y1 = xywh[0], xywh[1]
    x2 = x1 + max(0, xywh[2] - 1)
    y2 = y1 + max(0, xywh[3] - 1)
    return [x1, y1, x2, y2]


def xyxy_to_xywh(xyxy):
    """Helper method to transform bounding box of type [x1, y1, x2, y2] into [x1, y1, width, height]

    Args:
        xyxy (list or tuple): bounding box of type [x1, y1, x2, y2]

    Returns:
        list [x1, y1, width, height]

    """
    assert (
        isinstance(xyxy, (list, tuple, np.ndarray)) and len(xyxy) == 4
    ), "Argument xyxy should be a list/tuple of [x1, y1, x2, y2]"
    x1, y1 = xyxy[0], xyxy[1]
    w = xyxy[2] - x1 + 1
    h = xyxy[3] - y1 + 1
    return [x1, y1, w, h]
