
import sys

if sys.version_info[0] < 3:
    from backports import tempfile
    from pathlib2 import Path
else:
    import tempfile
    from pathlib import Path

from unittest import TestCase, main

import numpy as np
from PIL import Image, ImageFont

from image_dataset_viz.dataset_exporter import resize_image, DatasetExporter, \
    get_default_font, to_pil, render_datapoint


class TestDatasetExporter(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_default_font(self):
        font = get_default_font(10)
        assert isinstance(font, ImageFont.FreeTypeFont)

    def test_resize_image(self):
        size = (320, 300)
        large_img = Image.new('RGB', size=size)
        max_dims = (256, 256)
        img, _ = resize_image(large_img, max_dims)

        self.assertLessEqual(img.size[0], max_dims[0])
        self.assertLessEqual(img.size[1], max_dims[1])

    def test_asserts(self):

        with self.assertRaises(TypeError):
            de = DatasetExporter()
            de.export_datapoint(0, "test", "test.png")

        with self.assertRaises(TypeError):
            def read_img(i):
                img = Image.new(mode='RGB', size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
                return img

            de = DatasetExporter(read_img_fn=read_img)
            de.export_datapoint(0, 0, "test.png")

    def test_render_datapoint(self):

        img = np.ones((100, 120, 3), dtype=np.uint8)
        res = render_datapoint(img, "test label", text_color=(0, 255, 0), text_size=10)
        assert isinstance(res, Image.Image)

        target = Image.fromarray(np.ones((100, 120, 3), dtype=np.uint8))
        res = render_datapoint(img, target, text_color=(0, 255, 0), text_size=10)
        assert isinstance(res, Image.Image)

        target = np.array([[10, 10], [55, 10], [55, 77], [10, 77]])
        res = render_datapoint(img, target, geom_color=(255, 0, 0))
        assert isinstance(res, Image.Image)

    def test_export_datapoint(self):

        def read_img(i):
            img = Image.new(mode='RGB', size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        de = DatasetExporter(read_img_fn=read_img, img_id_fn=lambda x: str(x))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            de.export_datapoint(0, "test", path.as_posix())
            self.assertTrue(path.exists())

    def test_export_datapoint_target_none(self):

        def read_img(i):
            img = Image.new(mode='RGB', size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        de = DatasetExporter(read_img_fn=read_img, img_id_fn=lambda x: str(x))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            de.export_datapoint(0, None, path.as_posix())
            self.assertTrue(path.exists())

    def test_integration(self):

        def read_img(i):
            img = Image.new(mode='RGB', size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        def read_target(i):
            return "label_{}".format(i)

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(read_img_fn=read_img, read_target_fn=read_target,
                             img_id_fn=lambda x: str(x),
                             max_output_img_size=(s, s), margins=(m, m),
                             n_cols=n_cols, max_n_rows=max_n_rows)

        indices = [i for i in range(n)]

        with tempfile.TemporaryDirectory() as tmpdir:
            de.export(indices, indices, output_folder=tmpdir)
            path = Path(tmpdir)
            out_files = list(path.glob("*.png"))
            self.assertEqual(len(out_files), int(np.ceil(n / (n_cols * max_n_rows))))
            for fp in out_files:
                out_img = Image.open(fp)
                self.assertEqual(out_img.size, ((s + m) * n_cols, (s + m) * max_n_rows))

    def test_integration_targets_none(self):

        def read_img(i):
            img = Image.new(mode='RGB', size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(read_img_fn=read_img,
                             img_id_fn=lambda x: str(x),
                             max_output_img_size=(s, s), margins=(m, m),
                             n_cols=n_cols, max_n_rows=max_n_rows)

        indices = [i for i in range(n)]

        with tempfile.TemporaryDirectory() as tmpdir:
            de.export(indices, None, output_folder=tmpdir)
            path = Path(tmpdir)
            out_files = list(path.glob("*.png"))
            self.assertEqual(len(out_files), int(np.ceil(n / (n_cols * max_n_rows))))
            for fp in out_files:
                out_img = Image.open(fp)
                self.assertEqual(out_img.size, ((s + m) * n_cols, (s + m) * max_n_rows))

    def test_to_pil(self):
        img = np.ones((100, 120, 3), dtype=np.uint8)
        pil_img = to_pil(img)
        assert isinstance(pil_img, Image.Image)

        img = Image.fromarray(np.ones((100, 120, 3), dtype=np.uint8))
        pil_img = to_pil(img)
        assert isinstance(pil_img, Image.Image)


if __name__ == "__main__":
    main()
