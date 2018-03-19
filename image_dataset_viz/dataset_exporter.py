import sys
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont, Image

from tqdm import tqdm


def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.
    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
    )

    f = kwargs.get('file', sys.stderr)
    isatty = f.isatty()
    # NOTE when run under mpirun/slurm, isatty is always False
    # Jupyter notebook should be recognized as tty.
    # Wait for https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream
        if isinstance(f, iostream.OutStream):
            isatty = True
    except ImportError:
        pass

    if isatty:
        default['mininterval'] = 0.5
    else:
        # If not a tty, don't refresh progress bar that often
        default['mininterval'] = 180
    default.update(kwargs)
    return default


def get_tqdm(**kwargs):
    """ Similar to :func:`get_tqdm_kwargs`,
    but returns the tqdm object directly. """
    return tqdm(**get_tqdm_kwargs(**kwargs))


def imread_pillow(fp):
    return Image.open(fp)


def identity(y):
    return y


def default_img_id(fp):
    return Path(fp).stem[:50]


def resize_image(img, max_output_img_size):
    w, h = img.size
    f = 1.0
    if w > max_output_img_size[0] or h > max_output_img_size[1]:
        f = max(w / max_output_img_size[0], h / max_output_img_size[1])
        ow = int(w / f)
        oh = int(h / f)
        img = img.resize((ow, oh))
    return img, f


def update_pos(pos, text_size, w, h):
    pos = list(pos)
    if pos[0] + text_size[0] > w:
        pos[0] = w - text_size[0]
    if pos[1] + text_size[1] > h:
        pos[1] = h - text_size[1]
    return pos


def write_text(img, text, pos, color, font, max_line_length=30):
    draw = ImageDraw.Draw(img)

    if len(text) > max_line_length:
        ll = len(text)
        mltext = [text[c:c + max_line_length] for c in range(0, ll, max_line_length)]
        text = "\n".join(mltext)

    text_size = draw.textsize(text, font=font)
    pos = update_pos(pos, text_size, *img.size)
    draw.multiline_text(pos, text, color, font=font)


def draw_poly(img, points, color=(0, 255, 0)):
    draw = ImageDraw.Draw(img)
    draw.polygon(points, outline=color)


def write_obj_label(img, pos, label, font):
    draw = ImageDraw.Draw(img)
    text_size = draw.textsize(label, font=font)
    pos = update_pos(pos, text_size, *img.size)
    rect = (tuple(pos), (pos[0] + text_size[0], pos[1] + text_size[1]))
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text(tuple(pos), label, font=font, fill=(255, 255, 255))


def is_ndarray_image(img):
    return isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def is_pil_image(img):
    return isinstance(img, Image.Image) and img.mode in ('RGB', )


def is_points(target):
    return isinstance(target, np.ndarray) and target.ndim == 2 and target.shape[1] == 2


def is_list_of_points(target):
    return isinstance(target, (list, tuple)) and all([is_points(t) for t in target])


def is_points_with_labels(target):
    return isinstance(target, (tuple, list)) and len(target) == 2 and \
        isinstance(target[1], str) and is_points(target[0])


def is_list_of_points_with_labels(target):
    return isinstance(target, (list, tuple)) and all([is_points_with_labels(t) for t in target])


def check_image_type(img):
    assert is_ndarray_image(img) or \
        is_pil_image(img), "Image should be `ndarray` of shape (h, w, 3), type `uint8` or" + \
                           "`PIL.Image.Image` with mode 'RBG', but given {}".format(type(img))


def check_target_type(target):
    assert isinstance(target, str) or \
        is_points(target) or \
        is_list_of_points(target) or \
        is_points_with_labels(target) or \
        is_list_of_points_with_labels(target) or \
        is_ndarray_image(target) or \
        is_pil_image(target), "Target should be a text as `str`, points as list of `ndarray`s of shape (N, 2)," + \
                              "points with labels as list of pairs (`ndarray` of shape (N, 2), `str`)" + \
                              "or segmentation masks as `ndarray` of shape (h, w, 3), type `uint8` or " + \
                              "`PIL.Image.Image` with mode 'RGB', but given {}".format(type(target))


def to_pil(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    elif isinstance(img, Image.Image):
        return img


class DatasetExporter:
    """
    Helper class to export dataset of images/targets as a few images for better visualization.

    We assume
    """

    def __init__(self, read_img_fn=None, read_target_fn=None, img_id_fn=None,
                 max_output_img_size=(256, 256), margins=(5, 5),
                 n_cols=10, max_n_rows=50,
                 background_color=(127, 127, 120),
                 text_color=(255, 245, 235), text_size=11,
                 blend_alpha=0.75):
        # Initialize dataset exporter instance

        # read_img_fn should return image as `ndarray` of shape (h, w, 3), type `uint8` or `PIL.Image.Image` with
        # mode 'RBG'
        # read_target_fn can return text as `str`,
        #     list of polygons as list of `ndarray` of shape (N, 2),
        #     polygons points with labels as list of pairs (`ndarray` of shape (N, 2), `str`),
        #     or segmentation masks as `ndarray` of shape (h, w, 3), type `uint8` or `PIL.Image.Image` with mode 'RGB'

        self.read_img_fn = read_img_fn if read_img_fn is not None else imread_pillow
        self.read_target_fn = read_target_fn if read_target_fn is not None else identity
        self.img_id_fn = img_id_fn if img_id_fn is not None else default_img_id
        self.max_output_img_size = max_output_img_size
        self.margins = margins
        self.n_cols = n_cols
        self.max_n_rows = max_n_rows
        self.background_color = background_color
        self.text_color = text_color
        self.text_size = text_size
        self.blend_alpha = blend_alpha
        # Test create a font
        try:
            self.default_font = ImageFont.truetype(font="DejaVuSans.ttf", size=text_size)
        except OSError:
            font_path = Path(__file__).parent() / "fonts" / "DejaVuSans.ttf"
            self.default_font = ImageFont.truetype(font=font_path.as_posix(), size=text_size)

    def export_datapoint(self, img_file, target, output_filepath):
        img = self._render_datapoint(img_file, target)
        filepath = Path(output_filepath)
        if filepath.suffix != ".png":
            filepath += ".png"
        img.save(filepath)

    def _render_datapoint(self, f, t):
        # Open input image
        raw_img = self.read_img_fn(f)
        check_image_type(raw_img)

        img = to_pil(raw_img)
        img, scale = resize_image(img, self.max_output_img_size)

        # Open target
        target = self.read_target_fn(t)
        check_target_type(target)

        # Render target
        self._render_target(img, target, scale=scale)

        # Write image id
        image_id = self.img_id_fn(f)
        write_text(img, image_id, (1, img.size[1]),
                   color=self.text_color, font=self.default_font)
        return img

    def _render_target(self, img, target, scale=1.0):

        def _render_points(target):
            if scale > 1.0:
                target = [(int(p[0] / scale), int(p[1] / scale)) for p in target]
            draw_poly(img, target, color=(0, 255, 0))

        def _render_points_with_label(target):
            poly = target[0]
            if scale > 1.0:
                poly = [(int(p[0] / scale), int(p[1] / scale)) for p in poly]
            draw_poly(img, poly, color=(0, 255, 0))
            pos = np.max(poly, axis=0).tolist()
            write_obj_label(img, pos, label=target[1], font=self.default_font)

        if isinstance(target, str):
            write_text(img, target, (1, 1), color=self.text_color, font=self.default_font)
            return img
        elif is_points(target):
            _render_points(target)
        elif is_list_of_points(target):
            for t in target:
                _render_points(t)
        elif is_points_with_labels(target):
            _render_points_with_label(target)
        elif is_list_of_points_with_labels(target):
            for t in target:
                _render_points_with_label(t)
        elif is_ndarray_image(target) or is_pil_image(target):
            mask = to_pil(target)
            return Image.blend(img, mask, alpha=self.blend_alpha)

    def export(self, img_files, targets, output_folder, filename_prefix="dataset"):
        assert len(img_files) == len(targets), \
            "Number of input images should be equal to the number of input targets"
        output = Path(output_folder)
        if not output.exists():
            output.mkdir(parents=True)

        n_rows = max(min(len(img_files) // self.n_cols, self.max_n_rows), 1)
        total_width = (self.max_output_img_size[0] + self.margins[0]) * self.n_cols
        total_height = (self.max_output_img_size[0] + self.margins[0]) * n_rows
        size = (total_width, total_height)
        total_img = Image.new(mode='RGB', size=size, color=self.background_color)
        n_images = len(img_files)
        max_counter = n_rows * self.n_cols

        with get_tqdm(total=n_images) as bar:
            for c in range(0, n_images, max_counter):
                filepath = output / (filename_prefix + "_part_{}.png".format(c))
                for i, (f, t) in enumerate(zip(img_files[c:c + max_counter], targets[c:c + max_counter])):
                    iy, ix = np.unravel_index(i, (n_rows, self.n_cols))
                    x = ix * (self.max_output_img_size[0] + self.margins[0]) + self.margins[0] // 2
                    y = iy * (self.max_output_img_size[1] + self.margins[1]) + self.margins[1] // 2
                    img = self._render_datapoint(f, t)
                    total_img.paste(img, (x, y))
                total_img.save(filepath.as_posix())
                bar.update(max_counter)
