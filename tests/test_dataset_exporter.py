
import tempfile
from pathlib import Path

from unittest import TestCase, main

import numpy as np
from PIL import Image

from image_dataset_viz.dataset_exporter import resize_image, DatasetExporter


def read_img(i):
    img = Image.new(mode='RGB', size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
    return img


def read_target(i):
    return "label_{}".format(i)


def img_id_fn(x):
    return str(x)


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

    def test_asserts(self):

        with self.assertRaises(TypeError):
            de = DatasetExporter()
            de.export_datapoint(0, "test", "test.png")

        with self.assertRaises(TypeError):

            de = DatasetExporter(read_img_fn=read_img)
            de.export_datapoint(0, 0, "test.png")

    def test_export_datapoint(self):

        de = DatasetExporter(read_img_fn=read_img, img_id_fn=img_id_fn)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            de.export_datapoint(0, "test", path.as_posix())
            self.assertTrue(path.exists())

    def test_export_datapoint_target_none(self):

        de = DatasetExporter(read_img_fn=read_img, img_id_fn=img_id_fn)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            de.export_datapoint(0, None, path.as_posix())
            self.assertTrue(path.exists())

    def test_integration(self):

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(read_img_fn=read_img, read_target_fn=read_target,
                             img_id_fn=img_id_fn,
                             max_output_img_size=(s, s), margins=(m, m),
                             n_cols=n_cols, max_n_rows=max_n_rows, n_workers=1)

        indices = [i for i in range(n)]

        with tempfile.TemporaryDirectory() as tmpdir:
            de.export(indices, indices, output_folder=tmpdir)
            path = Path(tmpdir)
            out_files = list(path.glob("*.png"))
            self.assertEqual(len(out_files), int(np.ceil(n / (n_cols * max_n_rows))))
            for fp in out_files:
                out_img = Image.open(fp)
                self.assertEqual(out_img.size, ((s + m) * n_cols, (s + m) * max_n_rows))

    def test_integration_mp(self):

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(read_img_fn=read_img, read_target_fn=read_target,
                             img_id_fn=img_id_fn,
                             max_output_img_size=(s, s), margins=(m, m),
                             n_cols=n_cols, max_n_rows=max_n_rows, n_workers=4)

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

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(read_img_fn=read_img,
                             img_id_fn=img_id_fn,
                             max_output_img_size=(s, s), margins=(m, m),
                             n_cols=n_cols, max_n_rows=max_n_rows, n_workers=1)

        indices = [i for i in range(n)]

        with tempfile.TemporaryDirectory() as tmpdir:
            de.export(indices, None, output_folder=tmpdir)
            path = Path(tmpdir)
            out_files = list(path.glob("*.png"))
            self.assertEqual(len(out_files), int(np.ceil(n / (n_cols * max_n_rows))))
            for fp in out_files:
                out_img = Image.open(fp)
                self.assertEqual(out_img.size, ((s + m) * n_cols, (s + m) * max_n_rows))

    def test_integration_targets_none_mp(self):

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(read_img_fn=read_img,
                             img_id_fn=img_id_fn,
                             max_output_img_size=(s, s), margins=(m, m),
                             n_cols=n_cols, max_n_rows=max_n_rows, n_workers=4)

        indices = [i for i in range(n)]

        with tempfile.TemporaryDirectory() as tmpdir:
            de.export(indices, None, output_folder=tmpdir)
            path = Path(tmpdir)
            out_files = list(path.glob("*.png"))
            self.assertEqual(len(out_files), int(np.ceil(n / (n_cols * max_n_rows))))
            for fp in out_files:
                out_img = Image.open(fp)
                self.assertEqual(out_img.size, ((s + m) * n_cols, (s + m) * max_n_rows))


if __name__ == "__main__":
    main()
