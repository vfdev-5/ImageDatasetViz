from unittest import TestCase, main

import numpy as np

from image_dataset_viz import bbox_to_points, xywh_to_xyxy, xyxy_to_xywh


class TestHelperMethods(TestCase):

    def test_bbox_to_points(self):

        bbox = (10, 12, 34, 45)
        true_points = np.array([
            [10, 12],
            [34, 12],
            [34, 45],
            [10, 45]
        ])
        points = bbox_to_points(bbox)

        self.assertTrue((true_points == points).all())

    def test_xywh_to_xyxy(self):

        xywh = (10, 12, 34, 45)
        true_xyxy = [10, 12, 43, 56]
        xyxy = xywh_to_xyxy(xywh)
        self.assertEqual(true_xyxy, xyxy)

    def test_xyxy_to_xywh(self):

        xyxy = [10, 12, 43, 56]
        true_xywh = [10, 12, 34, 45]
        xywh = xyxy_to_xywh(xyxy)
        self.assertEqual(true_xywh, xywh)


if __name__ == "__main__":
    main()
