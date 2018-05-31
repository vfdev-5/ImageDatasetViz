from __future__ import division
import sys

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import numpy as np
from PIL import ImageDraw, ImageFont, Image

from tqdm import tqdm


def get_default_font(text_size):
    """Method to get a default font to render targets with text

    Args:
        text_size (int): text size in pixels

    Returns:
        PIL.ImageFont
    """
    assert isinstance(text_size, int) and text_size > 0, "Text size should be positive integer"
    try:
        return ImageFont.truetype(font="DejaVuSans.ttf", size=text_size)
    except OSError:
        font_path = Path(__file__).parent() / "fonts" / "DejaVuSans.ttf"
        return ImageFont.truetype(font=font_path.as_posix(), size=text_size)


def render_datapoint(img, target=None, image_id=None, output_size=None,
                     text_color=(0, 255, 0), text_size=10,
                     geom_color=(0, 255, 0), blend_alpha=0.7):
    """Method to render image and target as PIL image

    Args:
        img (PIL.Image.Image or np.ndarray): input image
        target (optional): text as `str`, list of polygons as list of `ndarray` of
                shape (N, 2), polygons points with labels as list of pairs (`ndarray` of shape (N, 2), `str`), or
                segmentation masks as `ndarray` of shape (h, w, 3), type `uint8` or `PIL.Image.Image` with mode 'RGB'.
        image_id (str, optional): Image id to write in the output image
        output_size (list, optional): output image maximum size. If input image height or width is larger
            than output_size, output image is rescaled with aspect ratio preserved.
        text_color (list, optional): text color (R, G, B) if target contains a text
        text_size (int, optional): text size if target contains a text.
        geom_color (int, optional): geometry color (R, G, B) if target contains polygons to draw
        blend_alpha (float): alpha for blending mask image into the input image

    Returns:
        PIL.Image.Image
    """
    # Open input image
    check_image_type(img)
    if target is not None:
        check_target_type(target)
        assert isinstance(text_color, (list, tuple)) and len(text_color) == 3, \
            "Text color should be a list of 3 integers"
        assert isinstance(text_size, int) and text_size > 0, \
            "Text size should be a positive integer"
        assert isinstance(blend_alpha, float) and 0.0 <= blend_alpha <= 1.0, \
            "Alpha should be a positive float between 0 and 1"
    if image_id is not None:
        assert isinstance(image_id, str), "Image id should be a string"

    img = to_pil(img)
    if output_size is not None:
        assert isinstance(output_size, (list, tuple)) and \
            len(output_size) == 2, "output_size should be a list of 2 integers".format(img.size)
        img, scale = resize_image(img, output_size)
    else:
        img = img.copy()
        scale = 1.0

    if target is not None:
        # Render target
        img = render_target(img, target, scale=scale, text_color=text_color,
                            text_size=text_size, geom_color=geom_color, blend_alpha=blend_alpha)

    # Write image id
    if image_id is not None:
        font = get_default_font(text_size)
        write_text(img, image_id, (1, img.size[1]), color=text_color, font=font)
    return img


class DatasetExporter:
    """
    Helper class to export dataset of images/targets as a few large images for better visualization.

    For example, we have a dataset of image files and annotations files (polygons with labels):
    ```
    img_files = [
        '/path/to/image_1.ext',
        '/path/to/image_2.ext',
        ...
        '/path/to/image_1000.ext',
    ]
    target_files = [
        '/path/to/target_1.ext2',
        '/path/to/target_2.ext2',
        ...
        '/path/to/target_1000.ext2',
    ]
    ```
    We can produce a single image composed of 20x50 small samples with targets to better visualize the whole dataset.
    Let's assume that we do need a particular processing to open the images in RGB 8bits format:
    ```
    from PIL import Image

    def read_img_fn(img_filepath):
        return Image.open(img_filepath).convert('RGB')
    ```
    and let's say the annotations are just lines with points and a label, e.g. `12 23 34 45 56 67 car`
    ```
    import numpy as np

    def read_target_fn(target_filepath):
        with Path(target_filepath).open('r') as handle:
            points_labels = []
            while True:
                line = handle.readline()
                if len(line) == 0:
                    break
                splt = line[:-1].split(' ')  # Split into points and labels
                label = splt[-1]
                points = np.array(splt[:-1]).reshape(-1, 2)
                points_labels.append((points, label))
        return points_labels
    ```
    Now we can export the dataset
    ```
    de = DatasetExporter(read_img_fn=read_img_fn, read_target_fn=read_target_fn,
                         img_id_fn=lambda fp: Path(fp).stem, n_cols=20)
    de.export(img_files, target_files, output_folder="dataset_viz")
    ```
    and thus we should obtain a single png image with composed of 20x50 small samples.
    """

    def __init__(self, read_img_fn=None, read_target_fn=None, img_id_fn=None,
                 max_output_img_size=(256, 256), margins=(5, 5),
                 n_cols=10, max_n_rows=50,
                 background_color=(127, 127, 120),
                 text_color=(255, 245, 235), text_size=11,
                 geom_color=(0, 255, 0),
                 blend_alpha=0.75):
        """
        Initialize dataset exporter instance

        Args:
            read_img_fn (callable): it specified, it should return image as `ndarray` of shape (h, w, 3), type `uint8`
                or `PIL.Image.Image` with mode 'RBG'. By default, `imread_pillow` function is used.
            read_target_fn (callable): if specified can return text as `str`, list of polygons as list of `ndarray` of
                shape (N, 2), polygons points with labels as list of pairs (`ndarray` of shape (N, 2), `str`), or
                segmentation masks as `ndarray` of shape (h, w, 3), type `uint8` or `PIL.Image.Image` with mode 'RGB'.
                By default, identity function is used. Provided target is rendered as a string value.
            img_id_fn (callable): optional, transforms image filepath to image id string displayed over each sample
            max_output_img_size (tuple of 2 integers): sample maximum size in the output image
            margins (tuple of 2 integers): margins between samples
            n_cols (int): number of columns in the output image
            max_n_rows (int): maximum number of rows of samples in the output image. If dataset is really big, thus we
                produce several output images instead of a single tower-like image.
            background_color (tuple or list): background color (R, G, B) if margin is not zero
            text_color (tuple or list): text color (R, G, B) used to draw image id and other labels
            text_size (int): text size
            geom_color (int, optional): geometry color (R, G, B) if target contains polygons to draw
            blend_alpha (float): alpha used to blend
        """
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
        self.geom_color = geom_color
        self.blend_alpha = blend_alpha
        # Test if can create a font
        self.default_font = get_default_font(text_size)

    def export_datapoint(self, img_file, target, output_filepath):
        """
        Export a single sample. This method can be used to test sample rendering

        Args:
            img_file: input image index, filepath, everything acceptable by `read_img_fn`
            target: input target index, filepath, everything acceptable by `read_target_fn`
                or None if not needed
            output_filepath: output filepath

        Returns:

        """
        check_img_id_fn(self.img_id_fn, img_file)
        raw_img = self.read_img_fn(img_file)
        image_id = self.img_id_fn(img_file)
        if target is not None:
            target = self.read_target_fn(target)

        img = render_datapoint(raw_img, target, image_id=image_id, output_size=self.max_output_img_size,
                               text_color=self.text_color, text_size=self.text_size,
                               geom_color=self.geom_color, blend_alpha=self.blend_alpha)

        filepath = Path(output_filepath)
        if filepath.suffix != ".png":
            filepath += ".png"
        img.save(filepath)

    def export(self, img_files, targets, output_folder, filename_prefix="dataset"):
        """
        Method to export the dataset represented by `img_files` and `targets`
        Args:
            img_files: list of image indices, paths, everything acceptable by `read_img_fn`
            targets: list of targets, indices, paths, everything acceptable by `read_target_fn`
                or None if not needed
            output_folder: (str) output folder path
            filename_prefix: (str) output large output image filename prefix. Output filename is
                `filename_prefix + "_part_x.png"`
        Returns:

        """
        assert isinstance(img_files, (list, tuple)), \
            "Arguments `img_files` should be lists or tuples"
        if targets is not None:
            assert isinstance(targets, (list, tuple)), \
                "Arguments `targets` should be lists or tuples"
            assert len(img_files) == len(targets), \
                "Number of input images should be equal to the number of input targets"
        else:
            targets = [None] * len(img_files)

        output = Path(output_folder)
        if not output.exists():
            output.mkdir(parents=True)

        n_rows = max(min(int(np.ceil(len(img_files) / self.n_cols)), self.max_n_rows), 1)
        total_width = (self.max_output_img_size[0] + self.margins[0]) * self.n_cols
        total_height = (self.max_output_img_size[0] + self.margins[0]) * n_rows
        size = (total_width, total_height)
        n_images = len(img_files)
        max_counter = n_rows * self.n_cols

        with get_tqdm(total=n_images) as bar:
            for c in range(0, n_images, max_counter):
                total_img = Image.new(mode='RGB', size=size, color=self.background_color)
                filepath = output / (filename_prefix + "_part_{}.png".format(c))
                for i, (f, t) in enumerate(zip(img_files[c:c + max_counter], targets[c:c + max_counter])):
                    iy, ix = np.unravel_index(i, (n_rows, self.n_cols))
                    x = ix * (self.max_output_img_size[0] + self.margins[0]) + self.margins[0] // 2
                    y = iy * (self.max_output_img_size[1] + self.margins[1]) + self.margins[1] // 2

                    raw_img = self.read_img_fn(f)
                    image_id = self.img_id_fn(f)
                    target = self.read_target_fn(t)
                    img = render_datapoint(raw_img, target, image_id=image_id, output_size=self.max_output_img_size,
                                           text_color=self.text_color, text_size=self.text_size,
                                           geom_color=self.geom_color, blend_alpha=self.blend_alpha)
                    total_img.paste(img, (x, y))
                    bar.update(1)
                total_img.save(filepath.as_posix())


def render_target(img, target, scale=1.0, text_color=(255, 255, 0), text_size=10,
                  geom_color=(0, 255, 0), blend_alpha=0.7):
    """Method to render target to the image

    Args:
        img (PIL.Image.Image or np.ndarray): input image
        target: text as `str`, list of polygons as list of `ndarray` of
                shape (N, 2), polygons points with labels as list of pairs (`ndarray` of shape (N, 2), `str`), or
                segmentation masks as `ndarray` of shape (h, w, 3), type `uint8` or `PIL.Image.Image` with mode 'RGB'.
        scale (float, optional): if scale is different of 1.0 then rescale the target
        text_color (list, optional): text color (R, G, B) if target contains a text
        text_size (int, optional): text size if target contains a text.
        geom_color (int, optional): geometry color (R, G, B) if target contains polygons to draw
        blend_alpha (float): alpha for blending mask image into the input image

    Returns:
        PIL.Image.Image
    """
    check_image_type(img)
    check_target_type(target)
    assert isinstance(text_color, (list, tuple)) and len(text_color) == 3, \
        "Text color should be a list of 3 integers"
    assert isinstance(text_size, int) and text_size > 0, \
        "Text size should be a positive integer"
    assert isinstance(geom_color, (list, tuple)) and len(geom_color) == 3, \
        "Geometry color should be a list of 3 integers"
    assert isinstance(blend_alpha, float) and 0.0 <= blend_alpha <= 1.0, \
        "Alpha should be a positive float between 0 and 1"
    assert isinstance(scale, float) and scale > 0.0, \
        "Scale should be a positive float"
    img = to_pil(img)

    def _render_points(target, color=(0, 255, 0)):
        target = [(int(p[0] / scale), int(p[1] / scale)) for p in target]
        draw_poly(img, target, color=color)

    def _render_points_with_label(target, font, color=(0, 255, 0)):
        poly = target[0]
        poly = [(int(p[0] / scale), int(p[1] / scale)) for p in poly]
        draw_poly(img, poly, color=color)
        pos = np.max(poly, axis=0).tolist()
        write_obj_label(img, pos, label=target[1], font=font)

    if isinstance(target, str):
        font = get_default_font(text_size)
        write_text(img, target, (1, 1), color=text_color, font=font)
    elif is_points(target):
        _render_points(target, geom_color)
    elif is_list_of_points(target):
        for t in target:
            _render_points(t, geom_color)
    elif is_points_with_labels(target):
        font = get_default_font(text_size)
        _render_points_with_label(target, font, geom_color)
    elif is_list_of_points_with_labels(target):
        font = get_default_font(text_size)
        for t in target:
            _render_points_with_label(t, font, geom_color)
    elif is_ndarray_image(target) or is_pil_image(target):
        mask = to_pil(target)
        if scale != 1.0:
            # Rescale mask
            mask = mask.resize(img.size)
        return Image.blend(img, mask, alpha=blend_alpha)
    return img


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
    """Similar to `get_tqdm_kwargs`, but returns the tqdm object directly."""
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


def check_img_id_fn(img_id_fn, f):
    try:
        ret = img_id_fn(f)
    except Exception as e:
        raise TypeError("There is a problem with provided `img_id_fn`: {}".format(e))
    assert isinstance(ret, str), "Output of `img_id_fn` should be a string"
