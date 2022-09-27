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

from image_dataset_viz import bbox_to_points
from image_dataset_viz.dataset_exporter import (
    resize_image,
    DatasetExporter,
    get_default_font,
    to_pil,
    render_datapoint,
)


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
        large_img = Image.new("RGB", size=size)
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
                img = Image.new(
                    mode="RGB",
                    size=(64, 64),
                    color=(i % 255, (i * 2) % 255, (i * 3) % 255),
                )
                return img

            de = DatasetExporter(read_img_fn=read_img)
            de.export_datapoint(0, 0, "test.png")

    def test_render_datapoint(self):

        img = ((0, 0, 210) * np.ones((256, 256, 3))).astype(np.uint8)
        target1 = "test label"
        res = render_datapoint(img, target1, text_color=(0, 123, 0), text_size=10)
        assert isinstance(res, Image.Image)
        np_res = np.asarray(res)
        unique_pixels = np_res.reshape(-1, 3).tolist()
        unique_pixels = set([tuple(p) for p in unique_pixels])
        assert (0, 0, 210) in unique_pixels
        assert (0, 123, 0) in unique_pixels

        img = ((0, 0, 210) * np.ones((256, 256, 3))).astype(np.uint8)
        target2 = np.zeros((256, 256, 3), dtype=np.uint8)
        target2[34:145, 56:123, :] = 255
        alpha = 0.5
        res = render_datapoint(img, target2, blend_alpha=alpha)
        assert isinstance(res, Image.Image)
        np_res = np.asarray(res)
        unique_pixels = np_res.reshape(-1, 3).tolist()
        unique_pixels = set([tuple(p) for p in unique_pixels])
        assert (
            int(255 * alpha),
            int(255 * alpha),
            int(210 * (1.0 - alpha) + 255 * alpha),
        ) in unique_pixels
        assert (0, 0, 210) in unique_pixels

        img = ((0, 0, 255) * np.ones((256, 256, 3))).astype(np.uint8)
        target3 = np.array([[10, 10], [55, 10], [55, 77], [10, 77]])
        res = render_datapoint(img, target3, geom_color=(255, 0, 0))
        assert isinstance(res, Image.Image)
        np_res = np.asarray(res)
        unique_pixels = np_res.reshape(-1, 3).tolist()
        unique_pixels = set([tuple(p) for p in unique_pixels])
        assert (0, 0, 255) in unique_pixels
        assert (255, 0, 0) in unique_pixels

        img = ((0, 0, 210) * np.ones((256, 256, 3))).astype(np.uint8)
        target4 = (np.array([[10, 10], [55, 10], [55, 77], [10, 77]]), "test")
        res = render_datapoint(img, target4, geom_color=(123, 0, 0))
        assert isinstance(res, Image.Image)
        np_res = np.asarray(res)
        unique_pixels = np_res.reshape(-1, 3).tolist()
        unique_pixels = set([tuple(p) for p in unique_pixels])
        assert (0, 0, 210) in unique_pixels
        assert (123, 0, 0) in unique_pixels
        assert (255, 255, 255) in unique_pixels
        assert (0, 0, 0) in unique_pixels

        res = render_datapoint(
            img,
            [target2, target1, target3, target4],
            text_color=(0, 123, 0),
            text_size=10,
            geom_color=(234, 0, 0),
            blend_alpha=alpha,
        )
        assert isinstance(res, Image.Image)
        np_res = np.asarray(res)
        unique_pixels = np_res.reshape(-1, 3).tolist()
        unique_pixels = set([tuple(p) for p in unique_pixels])
        # Check target2
        assert (
            int(255 * alpha),
            int(255 * alpha),
            int(210 * (1.0 - alpha) + 255 * alpha),
        ) in unique_pixels
        assert (0, 0, 210) in unique_pixels
        # Check target1
        assert (0, 123, 0) in unique_pixels
        # Check target3
        assert (234, 0, 0) in unique_pixels
        # Check label colors
        assert (255, 255, 255) in unique_pixels
        assert (0, 0, 0) in unique_pixels

    def test_render_datapoint_with_kwargs(self):
        img = ((123, 234, 220) * np.ones((256, 256, 3))).astype(np.uint8)
        mask = np.zeros((256, 256, 3), dtype=np.uint8)
        mask[34:145, 56:123, :] = (255, 255, 0)

        alpha = 0.7
        targets = (
            (mask, {"blend_alpha": alpha}),
            (
                (bbox_to_points((10, 12, 145, 156)), "A"),
                (bbox_to_points((109, 120, 215, 236)), "B"),
                {"geom_color": (255, 255, 0)},
            ),
            (bbox_to_points((129, 140, 175, 186)), "C"),
        )
        res = render_datapoint(img, targets, blend_alpha=0.5)
        assert isinstance(res, Image.Image)
        np_res = np.asarray(res)
        unique_pixels = np_res.reshape(-1, 3).tolist()
        unique_pixels = set([tuple(p) for p in unique_pixels])
        assert (
            int(123 * (1.0 - alpha) + alpha * 255),
            int(234 * (1.0 - alpha) + alpha * 255),
            int(220 * (1.0 - alpha) + alpha * 0),
        ) in unique_pixels
        assert (255, 255, 0) in unique_pixels
        assert (0, 255, 0) in unique_pixels
        assert (255, 255, 255) in unique_pixels
        assert (0, 0, 0) in unique_pixels

    def test_render_datapoint_with_several_masks(self):

        img = ((123, 234, 220) * np.ones((256, 256, 3))).astype(np.uint8)

        mask1 = np.zeros((256, 256, 3), dtype=np.uint8)
        mask1[34:145, 56:123, :] = (255, 0, 0)

        mask2 = np.zeros((256, 256, 3), dtype=np.uint8)
        mask2[134:245, 156:223, :] = (255, 255, 0)

        alpha1 = 0.7
        alpha2 = 0.7
        targets = (
            (mask1, {"blend_alpha": alpha1}),
            (mask2, {"blend_alpha": alpha2}),
            (
                (bbox_to_points((10, 12, 145, 156)), "A"),
                (bbox_to_points((109, 120, 215, 236)), "B"),
                {"geom_color": (255, 255, 0)},
            ),
            (bbox_to_points((129, 140, 175, 186)), "C"),
        )

        res = render_datapoint(img, targets, blend_alpha=0.5)
        assert isinstance(res, Image.Image)
        np_res = np.asarray(res)
        unique_pixels = np_res.reshape(-1, 3).tolist()
        unique_pixels = set([tuple(p) for p in unique_pixels])
        # image
        assert (123, 234, 220) in unique_pixels
        # mask1
        assert (
            int(123 * (1.0 - alpha1) + alpha1 * 255),
            int(234 * (1.0 - alpha1) + alpha1 * 0),
            int(220 * (1.0 - alpha1) + alpha1 * 0),
        ) in unique_pixels
        # mask2
        assert (
            int(123 * (1.0 - alpha2) + alpha2 * 255),
            int(234 * (1.0 - alpha2) + alpha2 * 255),
            int(220 * (1.0 - alpha2) + alpha2 * 0),
        ) in unique_pixels
        # geoms
        assert (255, 255, 0) in unique_pixels
        assert (0, 255, 0) in unique_pixels
        # labels
        assert (255, 255, 255) in unique_pixels
        assert (0, 0, 0) in unique_pixels

    def test_export_datapoint(self):
        def read_img(i):
            img = Image.new(mode="RGB", size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        de = DatasetExporter(read_img_fn=read_img, img_id_fn=lambda x: str(x))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            de.export_datapoint(0, "test", path.as_posix())
            self.assertTrue(path.exists())

    def test_export_datapoint_non_png(self):
        def read_img(i):
            img = Image.new(mode="RGB", size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        de = DatasetExporter(read_img_fn=read_img, img_id_fn=lambda x: str(x))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jpg"
            output_path = Path(tmpdir) / "test.jpg.png"
            de.export_datapoint(0, "test", path.as_posix())
            self.assertTrue(output_path.exists())

    def test_export_datapoint_target_none(self):
        def read_img(i):
            img = Image.new(mode="RGB", size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        de = DatasetExporter(read_img_fn=read_img, img_id_fn=lambda x: str(x))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            de.export_datapoint(0, None, path.as_posix())
            self.assertTrue(path.exists())

    def test_integration(self):
        def read_img(i):
            img = Image.new(mode="RGB", size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        def read_target(i):
            return "label_{}".format(i)

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(
            read_img_fn=read_img,
            read_target_fn=read_target,
            img_id_fn=lambda x: str(x),
            max_output_img_size=(s, s),
            margins=(m, m),
            n_cols=n_cols,
            max_n_rows=max_n_rows,
        )

        indices = [i for i in range(n)]

        with tempfile.TemporaryDirectory() as tmpdir:
            de.export(indices, indices, output_folder=tmpdir)
            path = Path(tmpdir)
            out_files = list(path.glob("*.png"))
            self.assertEqual(len(out_files), int(np.ceil(n / (n_cols * max_n_rows))))
            for fp in out_files:
                out_img = Image.open(fp)
                self.assertEqual(out_img.size, ((s + m) * n_cols, (s + m) * max_n_rows))

    def test_integration_targets_as_poly(self):
        def read_img(i):
            img = Image.new(mode="RGB", size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        def read_target(i):
            return np.array([[10 + i, 10], [55 + i, 10], [55, 77], [10, 77]]), "label_{}".format(i)

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(
            read_img_fn=read_img,
            read_target_fn=read_target,
            img_id_fn=lambda x: str(x),
            max_output_img_size=(s, s),
            margins=(m, m),
            n_cols=n_cols,
            max_n_rows=max_n_rows,
        )

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
            img = Image.new(mode="RGB", size=(64, 64), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            return img

        n = 100
        s = 32
        m = 5
        max_n_rows = 5
        n_cols = 10
        de = DatasetExporter(
            read_img_fn=read_img,
            img_id_fn=lambda x: str(x),
            max_output_img_size=(s, s),
            margins=(m, m),
            n_cols=n_cols,
            max_n_rows=max_n_rows,
        )

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
