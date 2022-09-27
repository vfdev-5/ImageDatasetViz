from unittest import TestCase, main

import numpy as np

from image_dataset_viz import bbox_to_points, xywh_to_xyxy, xyxy_to_xywh


class TestHelperMethods(TestCase):
    def _test_func(self, input_, true_output, func):

        output = func(list(input_))
        self.assertTrue(np.equal(true_output, output).all())

        output = func(tuple(input_))
        self.assertTrue(np.equal(true_output, output).all())

        output = func(np.array(input_))
        self.assertTrue(np.equal(true_output, output).all())

        with self.assertRaises(AssertionError):
            func("1234")

        with self.assertRaises(AssertionError):
            func([1, 2])

        with self.assertRaises(AssertionError):
            func([1, 2, 3, 4, 5])

    def test_bbox_to_points(self):
        bbox = (10, 12, 34, 45)
        true_points = np.array([[10, 12], [34, 12], [34, 45], [10, 45]])
        self._test_func(bbox, true_points, bbox_to_points)

    def test_xywh_to_xyxy(self):
        xywh = (10, 12, 34, 45)
        true_xyxy = [10, 12, 43, 56]
        self._test_func(xywh, true_xyxy, xywh_to_xyxy)

    def test_xyxy_to_xywh(self):
        xyxy = [10, 12, 43, 56]
        true_xywh = [10, 12, 34, 45]
        self._test_func(xyxy, true_xywh, xyxy_to_xywh)


if __name__ == "__main__":
    main()
