
from unittest import TestCase, main

from PIL import Image

from image_dataset_viz.dataset_exporter import resize_image


class TestDatasetExporter(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_resize_image(self):
        size = (320, 300)
        large_img = Image.new('RGB', size=size)
        max_dims = (256, 256)
        img, _ = resize_image(large_img, max_dims)

        self.assertLessEqual(img.size[0], max_dims[0])
        self.assertLessEqual(img.size[1], max_dims[1])


if __name__ == "__main__":
    main()
