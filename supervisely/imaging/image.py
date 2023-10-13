# coding: utf-8

import io
import os.path
from pkg_resources import parse_version
import base64
from typing import List, Tuple, Optional, Union
import cv2
from PIL import ImageDraw, ImageFile, ImageFont, Image as PILImage
import numpy as np
from enum import Enum
import nrrd

from supervisely.io.fs import ensure_base_path, get_file_ext, silent_remove
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.imaging.font import get_font
from supervisely._utils import get_bytes_hash, is_development, abs_url, rand_str

ImageFile.LOAD_TRUNCATED_IMAGES = True

# @TODO: refactoring image->img
KEEP_ASPECT_RATIO = -1  # TODO: need move it to best place

# Do NOT use directly for image extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
SUPPORTED_IMG_EXTS = [".jpg", ".jpeg", ".mpo", ".bmp", ".png", ".webp", ".tiff", ".tif", ".nrrd"]
DEFAULT_IMG_EXT = ".png"


class CornerAnchorMode:
    """ """

    TOP_LEFT = "tl"
    TOP_RIGHT = "tr"
    BOTTOM_LEFT = "bl"
    BOTTOM_RIGHT = "br"


class RotateMode(Enum):
    """ """

    KEEP_BLACK = 0
    """"""
    CROP_BLACK = 1
    """"""
    SAVE_ORIGINAL_SIZE = 2
    """"""


class ImageExtensionError(Exception):
    """ """

    pass


class UnsupportedImageFormat(Exception):
    """ """

    pass


class ImageReadException(Exception):
    """ """

    pass


def is_valid_ext(ext: str) -> bool:
    """
    Checks file extension for list of supported images extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp', '.tiff', '.tif', '.nrrd').

    :param ext: Image extention.
    :type ext: str
    :return: True if image extention in list of supported images extensions, False - in otherwise
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        sly.image.is_valid_ext('.png') # True
        sly.image.is_valid_ext('.py') # False
    """
    return ext.lower() in SUPPORTED_IMG_EXTS


def has_valid_ext(path: str) -> bool:
    """
    Checks if a given file has a supported extension('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp', '.tiff', '.tif', '.nrrd').

    :param path: Path to file.
    :type path: str
    :return: True if file extention in list of supported images extensions, False - in otherwise
    :rtype: :class:`bool`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        sly.image.has_valid_ext('/home/admin/work/docs/new_image.jpeg') # True
        sly.image.has_valid_ext('/home/admin/work/docs/016_img.py') # False
    """
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(path: str) -> None:
    """
    Generate exception error if file extention is not in list of supported images extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp', '.tiff', '.tif', '.nrrd').

    :param path: Path to file.
    :type path: str
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        print(sly.image.validate_ext('/home/admin/work/docs/new_image.jpeg'))
        # Output: None

        try:
            print(sly.image.validate_ext('/home/admin/work/docs/016_img.py'))
        except ImageExtensionError as error:
            print(error)

        # Output: Unsupported image extension: '.py' for file '/home/admin/work/docs/016_img.py'. Only the following extensions are supported: .jpg, .jpeg, .mpo, .bmp, .png, .webp.
    """
    _, ext = os.path.splitext(path)
    if not is_valid_ext(ext):
        raise ImageExtensionError(
            "Unsupported image extension: {!r} for file {!r}. Only the following extensions are supported: {}.".format(
                ext, path, ", ".join(SUPPORTED_IMG_EXTS)
            )
        )


def validate_format(path: str) -> None:
    """
    Validate input file format, if file extension is not supported raise ImageExtensionError.

    :param path: Path to file.
    :type path: str
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        print(sly.image.validate_format('/home/admin/work/docs/new_image.jpeg'))
        # Output: None

        try:
            print(sly.image.validate_format('/home/admin/work/docs/016_img.py'))
        except ImageReadException as error:
            print(error)

        # Output: Error has occured trying to read image '/home/admin/work/docs/016_img.py'. Original exception message: "cannot identify image file '/home/admin/work/docs/016_img.py'"
    """
    ext = get_file_ext(path)
    if ext == ".nrrd":
        data, header = nrrd.read(path, index_order='C')
        return
    try:
        pil_img = PILImage.open(path)
        pil_img.load()  # Validate image data. Because 'open' is lazy method.
    except OSError as e:
        raise ImageReadException(
            "Error has occured trying to read image {!r}. Original exception message: {!r}".format(
                path, str(e)
            )
        )

    img_format = pil_img.format
    img_ext = f".{img_format}"
    if not is_valid_ext(f".{img_format}"):
        raise UnsupportedImageFormat(
            "Unsupported image format {!r} for file {!r}. Only the following formats are supported: {}".format(
                img_ext, path, ", ".join(SUPPORTED_IMG_EXTS)
            )
        )


def read(path: str, remove_alpha_channel: Optional[bool] = True) -> np.ndarray:
    """
    Loads an image from the specified file and returns it in RGB format.

    :param path: Path to file.
    :type path: str
    :param remove_alpha_channel: Define remove alpha channel when reading file or not.
    :type remove_alpha_channel: bool, optional
    :return: Numpy array
    :rtype: :class:`np.ndarray`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        im = sly.image.read('/home/admin/work/docs/image.jpeg')
    """
    ext = get_file_ext(path)
    if ext == ".nrrd":
        data, header = nrrd.read(path, index_order='C')
        return data

    validate_format(path)
    if remove_alpha_channel is True:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("OpenCV can not open the file {!r}".format(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("OpenCV can not open the file {!r}".format(path))
        cnt_channels = img.shape[2]
        if cnt_channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif cnt_channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cnt_channels == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(
                "image has {} channels. Please, contact support...".format(cnt_channels)
            )


def read_bytes(image_bytes: str, keep_alpha: Optional[bool] = False) -> np.ndarray:
    """
    Loads an byte image and returns it in RGB format.

    :param image_bytes: Path to file.
    :type image_bytes: str
    :param keep_alpha: Define consider alpha channel when reading bytes or not.
    :type keep_alpha: bool, optional
    :return: Numpy array
    :rtype: :class:`np.ndarray`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        im_bytes = '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\...\xd9'
        im = sly.image.read_bytes(im_bytes)
    """
    if image_bytes.startswith(b"NRRD"):
        file_like = io.BytesIO(image_bytes)
        header = nrrd.read_header(file_like)
        data = nrrd.read_data(header, file_like, index_order='C')
        return data

    image_np_arr = np.asarray(bytearray(image_bytes), dtype="uint8")
    if keep_alpha is True:
        img = cv2.imdecode(image_np_arr, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        cnt_channels = img.shape[2]
        if cnt_channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif cnt_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cnt_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    else:
        img = cv2.imdecode(
            image_np_arr, cv2.IMREAD_COLOR
        )  # cv2.imdecode returns BGR always
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write(
    path: str, img: np.ndarray, remove_alpha_channel: Optional[bool] = True
) -> None:
    """
    Saves the image to the specified file. It create directory from path if the directory for this path does not exist.

    :param path: Path to file.
    :type path: str
    :param img: Image in numpy array(RGB format).
    :type img: np.ndarray
    :param remove_alpha_channel: Define remove alpha channel when writing file or not.
    :type remove_alpha_channel: bool, optional
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = '/home/admin/work/docs/new_image.jpeg'
        sly.image.write(path, image_np)
    """
    ensure_base_path(path)
    validate_ext(path)

    ext = get_file_ext(path)
    if ext == ".nrrd":
        return nrrd.write(path, img, index_order='C')

    res_img = img.copy()
    if len(img.shape) == 2:
        res_img = np.expand_dims(img, 2)
    cnt_channels = res_img.shape[2]
    if cnt_channels == 4:
        if remove_alpha_channel is True:
            res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        else:
            res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    elif cnt_channels == 3:
        res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    elif cnt_channels == 1:
        res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.imwrite(path, res_img)


def draw_text_sequence(
    bitmap: np.ndarray,
    texts: List[str],
    anchor_point: Tuple[int, int],
    corner_snap: Optional[CornerAnchorMode] = CornerAnchorMode.TOP_LEFT,
    col_space: Optional[int] = 12,
    font: Optional[ImageFont.FreeTypeFont] = None,
    fill_background: Optional[bool] = True,
) -> None:
    """
    Draws text labels on bitmap from left to right with col_space spacing between labels.

    :param bitmap: Image to draw texts in numpy format.
    :type bitmap: np.ndarray
    :param texts: List of texts to draw on image.
    :type texts: List[str]
    :param anchor_point: Coordinates of the place on the image where the text will be displayed(row, column).
    :type anchor_point: Tuple[int, int]
    :param corner_snap: Corner of image to draw texts.
    :type corner_snap: CornerAnchorMode, optional
    :param col_space: Distance between texts.
    :type col_space: int, optional
    :param font: Type of text font.
    :type font: ImageFont.FreeTypeFont, optional
    :param fill_background: Define fill text background or not.
    :type fill_background: bool, optional
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        sly.image.draw_text_sequence(image, ['some_text', 'another_text'], (10, 10))

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/wIzrDuf.jpg

                   After
    """
    col_offset = 0
    for text in texts:
        position = anchor_point[0], anchor_point[1] + col_offset
        _, text_width = draw_text(
            bitmap, text, position, corner_snap, font, fill_background
        )
        col_offset += text_width + col_space


def draw_text(
    bitmap: np.ndarray,
    text: str,
    anchor_point: Tuple[int, int],
    corner_snap: Optional[CornerAnchorMode] = CornerAnchorMode.TOP_LEFT,
    font: Optional[ImageFont.FreeTypeFont] = None,
    fill_background: Optional[bool] = True,
    color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int]]] = (0, 0, 0, 255),
) -> tuple:
    """
    Draws given text on bitmap image.

    :param bitmap: Image to draw texts in numpy format.
    :type bitmap: np.ndarray
    :param texts: Text to draw on image.
    :type texts: str
    :param anchor_point: Coordinates of the place on the image where the text will be displayed(row, column).
    :type anchor_point: Tuple[int, int]
    :param corner_snap: Corner of image to draw texts.
    :type corner_snap: CornerAnchorMode, optional
    :param font: Type of text font.
    :type font: ImageFont.FreeTypeFont, optional
    :param fill_background: Define fill text background or not.
    :type fill_background: bool, optional
    :param color: Text color as a tuple of three or four integers (red, green, blue, alpha)
                  ranging from 0 to 255.
                  If alpha is not provided, it defaults to 255 (fully opaque).
    :type color: Union[Tuple[int, int, int, int], Tuple[int, int, int]], optional
    :return: Height and width of text
    :rtype: :class:`Tuple[int, int]`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        sly.image.draw_text(image, 'your text', (100, 50), color=(0, 0, 0, 255))

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/6Ptz8Tf.jpg

                   After
    """
    if not (isinstance(color, (list, tuple)) and len(color) in [3, 4]):
        raise TypeError("Color must be list or tuple of three or four elements")

    if font is None:
        font = get_font()

    source_img = PILImage.fromarray(bitmap)
    source_img = source_img.convert("RGBA")

    canvas = PILImage.new("RGBA", source_img.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(canvas, "RGBA")
    text_width, text_height = drawer.textsize(text, font=font)
    rect_top, rect_left = anchor_point

    if corner_snap == CornerAnchorMode.TOP_LEFT:
        pass  # Do nothing
    elif corner_snap == CornerAnchorMode.TOP_RIGHT:
        rect_left -= text_width
    elif corner_snap == CornerAnchorMode.BOTTOM_LEFT:
        rect_top -= text_height
    elif corner_snap == CornerAnchorMode.BOTTOM_RIGHT:
        rect_top -= text_height
        rect_left -= text_width

    if fill_background:
        rect_right = rect_left + text_width
        rect_bottom = rect_top + text_height
        drawer.rectangle(
            ((rect_left, rect_top), (rect_right + 1, rect_bottom)),
            fill=(255, 255, 255, 128),
        )
    drawer.text((rect_left + 1, rect_top), text, fill=color, font=font)

    source_img = PILImage.alpha_composite(source_img, canvas)
    source_img = source_img.convert("RGB")
    bitmap[:, :, :] = np.array(source_img, dtype=np.uint8)

    return (text_height, text_width)


# @TODO: not working with alpha channel
def write_bytes(img: np.ndarray, ext: str) -> bytes:
    """
    Compresses the image and stores it in the byte object.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param ext: File extension that defines the output format.
    :type ext: str
    :return: Bytes object
    :rtype: :class:`bytes`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        bytes = sly.image.write_bytes(image_np, 'jpeg')
        print(type(bytes))
        # Output: <class 'bytes'>
    """
    ext = ("." + ext).replace("..", ".")
    if not is_valid_ext(ext):
        raise UnsupportedImageFormat(
            "Unsupported image format {!r}. Only the following formats are supported: {}".format(
                ext, ", ".join(SUPPORTED_IMG_EXTS)
            )
        )
    if ext == ".nrrd":
        nrrd_bytes = None
        _filename = f"./sly-nrrd-data-bytes-{rand_str(10)}{ext}"
        nrrd.write(_filename, img, index_order='C')
        with open(_filename, "rb") as nrrd_file:
            nrrd_bytes = nrrd_file.read()
        silent_remove(_filename)
        return nrrd_bytes

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    encode_status, img_array = cv2.imencode(ext, img)
    if encode_status is True:
        return img_array.tobytes()
    raise RuntimeError("Can not encode input image")


# @TODO: not working with alpha channel
def get_hash(img: np.ndarray, ext: str) -> str:
    """
    Hash input image with sha256 algoritm and encode result by using Base64.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param ext: File extension that defines the output format.
    :type ext: str
    :return: Hash string
    :rtype: :class:`str`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        hash = sly.image.get_hash(im, 'jpeg')
        print(hash)
        # Output: fTec3RD7Zxg0aYc0ooa5phPfBrzDe01urlFsgi5IzIQ=
    """
    return get_bytes_hash(write_bytes(img, ext))


def crop(img: np.ndarray, rect: Rectangle) -> np.ndarray:
    """
    Crop part of the image with rectangle size. If rectangle for crop is out of image area it generates ValueError.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param rect: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>` object for crop.
    :type rect: Rectangle
    :return: Cropped image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # If size of rectangle is more then image shape raise ValueError:
        try:
            crop_image = sly.image.crop(image_np, sly.Rectangle(0, 0, 5000, 6000))
        except ValueError as error:
            print(error)
        # Output: Rectangle for crop out of image area!

        crop_im = sly.image.crop(image_np, sly.Rectangle(0, 0, 500, 600))

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/4tNm2GS.jpg

                   After
    """
    img_rect = Rectangle.from_array(img)
    if not img_rect.contains(rect):
        raise ValueError("Rectangle for crop out of image area!")
    return rect.get_cropped_numpy_slice(img)


def crop_with_padding(img: np.ndarray, rect: Rectangle) -> np.ndarray:
    """
    Crop part of the image with rectangle size. If rectangle for crop is out of image area it generates additional padding.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param rect: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>` object for crop.
    :type rect: Rectangle
    :return: Cropped image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        crop_with_padding_image = sly.image.crop_with_padding(image_np, sly.Rectangle(0, 0, 1000, 1200))

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/Nv1UinH.jpg

                   After
    """
    img_rect = Rectangle.from_array(img)
    if not img_rect.contains(rect):
        row, col = img.shape[:2]
        new_img = cv2.copyMakeBorder(
            img,
            top=rect.height,
            bottom=rect.height,
            left=rect.width,
            right=rect.width,
            borderType=cv2.BORDER_CONSTANT,
        )
        new_rect = rect.translate(drow=rect.height, dcol=rect.width)
        return new_rect.get_cropped_numpy_slice(new_img)

    else:
        return rect.get_cropped_numpy_slice(img)


def restore_proportional_size(
    in_size: Tuple[int, int],
    out_size: Optional[Tuple[int, int]] = None,
    frow: Optional[float] = None,
    fcol: Optional[float] = None,
    f: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Calculate new size of the image.

    :param in_size: Size of input image (height, width).
    :type in_size: Tuple[int, int]
    :param out_size: New image size (height, width).
    :type out_size: Tuple[int, int], optional
    :param frow: Length of output image.
    :type frow: float, optional
    :param fcol: Height of output image.
    :type fcol: float, optional
    :param f: Positive non zero scale factor.
    :type f: float, optional
    :return: Height and width of image
    :rtype: :class:`Tuple[int, int]`
    """
    if out_size is not None and (frow is not None or fcol is not None) and f is None:
        raise ValueError(
            "Must be specified output size or scale factors not both of them."
        )

    if out_size is not None:
        if out_size[0] == KEEP_ASPECT_RATIO and out_size[1] == KEEP_ASPECT_RATIO:
            raise ValueError("Must be specified at least 1 dimension of size!")

        if (out_size[0] <= 0 and out_size[0] != KEEP_ASPECT_RATIO) or (
            out_size[1] <= 0 and out_size[1] != KEEP_ASPECT_RATIO
        ):
            raise ValueError("Size dimensions must be greater than 0.")

        result_row = (
            out_size[0]
            if out_size[0] > 0
            else max(1, round(out_size[1] / in_size[1] * in_size[0]))
        )
        result_col = (
            out_size[1]
            if out_size[1] > 0
            else max(1, round(out_size[0] / in_size[0] * in_size[1]))
        )
    else:
        if f is not None:
            if f < 0:
                raise ValueError('"f" argument must be positive!')
            frow = fcol = f

        if (frow < 0 or fcol < 0) or (frow is None or fcol is None):
            raise ValueError('Specify "f" argument for single scale!')

        result_col = round(fcol * in_size[1])
        result_row = round(frow * in_size[0])
    return result_row, result_col


# @TODO: reimplement, to be more convenient
def resize(
    img: np.ndarray,
    out_size: Optional[Tuple[int, int]] = None,
    frow: Optional[float] = None,
    fcol: Optional[float] = None,
) -> np.ndarray:
    """
    Resize the image to the specified size.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param out_size: New image size (height, width).
    :type out_size: Tuple[int, int], optional
    :param frow: Length of output image.
    :type frow: float, optional
    :param fcol: Height of output image.
    :type fcol: float, optional
    :return: Resize image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        resize_image = sly.image.resize(image_np, (300, 500))

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/Xya4yz0.jpg

                   After
    """
    result_height, result_width = restore_proportional_size(
        img.shape[:2], out_size, frow, fcol
    )
    return cv2.resize(img, (result_width, result_height), interpolation=cv2.INTER_CUBIC)


def resize_inter_nearest(
    img: np.ndarray,
    out_size: Optional[Tuple[int, int]] = None,
    frow: Optional[float] = None,
    fcol: Optional[float] = None,
) -> np.ndarray:
    """
    Resize image to match a certain size. Performs interpolation to up-size or down-size images.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param out_size: New image size (height, width).
    :type out_size: Tuple[int, int], optional
    :param frow: Length of output image.
    :type frow: float, optional
    :param fcol: Height of output image.
    :type fcol: float, optional
    :return: Resize image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        resize_image_nearest = sly.image.resize_inter_nearest(image_np, (300, 700))

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/0O6yMDH.jpg

                   After
    """
    import skimage.transform

    target_shape = restore_proportional_size(img.shape[:2], out_size, frow, fcol)
    resize_kv_args = dict(order=0, preserve_range=True, mode="constant")
    if parse_version(skimage.__version__) >= parse_version("0.14.0"):
        resize_kv_args["anti_aliasing"] = False
    return skimage.transform.resize(img, target_shape, **resize_kv_args).astype(
        img.dtype
    )


def scale(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Scales current image with the given factor.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param factor: Scale size.
    :type factor: float
    :return: Resize image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        scale_image = sly.image.scale(image_np, 0.3)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/NyP8tts.jpg

                   After
    """
    return resize(img, (round(img.shape[0] * factor), round(img.shape[1] * factor)))


def fliplr(img: np.ndarray) -> np.ndarray:
    """
    Flips the current image horizontally.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :return: Flip image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        fliplr_image = sly.image.fliplr(image_np)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/1mqnuZU.jpg

                   After
    """
    return np.flip(img, 1)


def flipud(img: np.ndarray) -> np.ndarray:
    """
    Flips the current image vertically.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :return: Flip image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        flipud_image = sly.image.flipud(image_np)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/LDwRDvm.jpg

                   After
    """
    return np.flip(img, 0)


def rotate(
    img: np.ndarray,
    degrees_angle: int,
    mode: Optional[RotateMode] = RotateMode.KEEP_BLACK,
) -> np.ndarray:
    """
    Rotates current image.

    :param img: Image in numpy format(RGB).
    :type img: np.ndarray
    :param degrees_angle: Angle in degrees for rotating.
    :type degrees_angle: int
    :param mode: One of RotateMode enum values.
    :type mode: RotateMode, optional
    :return: Rotate image in numpy format
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # keep_black mode
        rotate_im_keep_black = sly.image.rotate(image_np, 45)
        # crop_black mode
        rotate_im_crop_black = sly.image.rotate(image_np, 45, sly.image.RotateMode.CROP_BLACK)
        # origin_size mode
        rotate_im_origin_size = sly.image.rotate(image_np, 45, sly.image.RotateMode.SAVE_ORIGINAL_SIZE) * 255

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/VjiwV4O.jpg

                   After keep_black mode

          - .. figure:: https://i.imgur.com/Rs34eMa.jpg

                   After crop_black mode

          - .. figure:: https://i.imgur.com/ttDWfBE.jpg

                   After origin_size mode
    """
    import skimage.transform

    rotator = ImageRotator(imsize=img.shape[:2], angle_degrees_ccw=degrees_angle)
    if mode == RotateMode.KEEP_BLACK:
        return rotator.rotate_img(img, use_inter_nearest=False)  # @TODO: order = ???
    elif mode == RotateMode.CROP_BLACK:
        img_rotated = rotator.rotate_img(img, use_inter_nearest=False)
        return rotator.inner_crop.get_cropped_numpy_slice(img_rotated)
    elif mode == RotateMode.SAVE_ORIGINAL_SIZE:
        # TODO Implement this in rotator instead.
        return skimage.transform.rotate(img, degrees_angle, resize=False)
    else:
        raise NotImplementedError('Rotate mode "{0}" not supported!'.format(str(mode)))


# Color augmentations
def _check_contrast_brightness_inputs(min_value, max_value):
    """
    The function _check_contrast_brightness_inputs checks the input brightness or contrast for correctness.
    """
    if min_value < 0:
        raise ValueError("Minimum value must be greater than or equal to 0.")
    if min_value > max_value:
        raise ValueError(
            "Maximum value must be greater than or equal to minimum value."
        )


def random_contrast(
    image: np.ndarray, min_factor: float, max_factor: float
) -> np.ndarray:
    """
    Randomly changes contrast of the input image.

    :param image: Image in numpy format(RGB).
    :type image: np.ndarray
    :param min_factor: Lower bound of contrast range.
    :type min_factor: float
    :param max_factor: Upper bound of contrast range.
    :type max_factor: float
    :return: Image in numpy format with new contrast
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        rand_contrast_im = sly.image.random_contrast(image_np, 1.1, 1.8)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/4zSNuJU.jpg

                   After
    """
    _check_contrast_brightness_inputs(min_factor, max_factor)
    contrast_value = np.random.uniform(min_factor, max_factor)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_mean = round(image_gray.mean())
    image = image.astype(np.float32)
    image = contrast_value * (image - image_mean) + image_mean
    return np.clip(image, 0, 255).astype(np.uint8)


def random_brightness(
    image: np.ndarray, min_factor: float, max_factor: float
) -> np.ndarray:
    """
    Randomly changes brightness of the input image.

    :param image: Image in numpy format(RGB).
    :type image: np.ndarray
    :param min_factor: Lower bound of brightness range.
    :type min_factor: float
    :param max_factor: Upper bound of brightness range.
    :type max_factor: float
    :return: Image in numpy format with new brightness
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        rand_brightness_im = sly.image.random_brightness(image_np, 1.5, 8.5)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/bOYwwYH.jpg

                   After
    """
    _check_contrast_brightness_inputs(min_factor, max_factor)
    brightness_value = np.random.uniform(min_factor, max_factor)
    image = image.astype(np.float32)
    image = image * brightness_value
    return np.clip(image, 0, 255).astype(np.uint8)


def random_noise(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Adds random Gaussian noise to the input image.

    :param image: Image in numpy format(RGB).
    :type image: np.ndarray
    :param mean: The mean value of noise distribution.
    :type mean: float
    :param std: The standard deviation of noise distribution.
    :type std: float
    :return: Image in numpy format with random noise
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        random_noise_im = sly.image.random_noise(image_np, 25, 19)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/EzyEHeM.jpg

                   After
    """
    image = image.astype(np.float32)
    image += np.random.normal(mean, std, image.shape)

    return np.clip(image, 0, 255).astype(np.uint8)


def random_color_scale(
    image: np.ndarray, min_factor: float, max_factor: float
) -> np.ndarray:
    """
    Changes image colors by randomly scaling each of RGB components. The scaling factors are sampled uniformly from the given range.

    :param image: Image in numpy format(RGB).
    :type image: np.ndarray
    :param min_factor: Minimum scale factor.
    :type min_factor: float
    :param max_factor: Maximum scale factor.
    :type max_factor: float
    :return: Image in numpy format with random color scale
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        random_color_scale_im = sly.image.random_color_scale(image_np, 0.5, 0.9)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/GGUZqlA.jpg

                   After
    """
    image_float = image.astype(np.float64)
    scales = np.random.uniform(
        low=min_factor, high=max_factor, size=(1, 1, image.shape[2])
    )
    res_image = image_float * scales
    return np.clip(res_image, 0, 255).astype(np.uint8)


# Blurs
def blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blurs an image using the normalized box filter.

    :param image: Image in numpy format(RGB).
    :type image: np.ndarray
    :param kernel_size: Blurring kernel size.
    :type kernel_size: int
    :return: Image in numpy format with blur
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        blur_im = sly.image.blur(image_np, 7)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/wFnBaC6.jpg

                   After
    """
    return cv2.blur(image, (kernel_size, kernel_size))


def median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blurs an image using the median filter.

    :param image: Image in numpy format(RGB).
    :type image: np.ndarray
    :param kernel_size: Blurring kernel size(must be odd and greater than 1, for example: 3, 5, 7).
    :type kernel_size: int
    :return: Image in numpy format with median blur
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        median_blur_im = sly.image.median_blur(image_np, 5)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/FQ977ON.jpg

                   After
    """
    return cv2.medianBlur(image, kernel_size)


def gaussian_blur(image: np.ndarray, sigma_min: float, sigma_max: float) -> np.ndarray:
    """
    Blurs an image using a Gaussian filter.

    :param image: Image in numpy format(RGB).
    :type image: np.ndarray
    :param sigma_min: Lower bound of Gaussian kernel standard deviation range.
    :type sigma_min: float
    :param sigma_min: Upper bound of Gaussian kernel standard deviation range.
    :type sigma_max: float
    :return: Image in numpy format with gaussian blur
    :rtype: :class:`np.ndarray`

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        gaussian_blur_im = sly.image.gaussian_blur(image_np, 3.3, 7.5)

    .. list-table::

        * - .. figure:: https://i.imgur.com/BHUALdv.jpg

                   Before

          - .. figure:: https://i.imgur.com/brs6Au0.jpg

                   After
    """
    sigma_value = np.random.uniform(sigma_min, sigma_max)
    return cv2.GaussianBlur(image, (0, 0), sigma_value)


def drop_image_alpha_channel(img: np.ndarray) -> np.ndarray:
    """
    Converts 4-channel image to 3-channel.

    :param img: Image in numpy format(RGBA).
    :type img: np.ndarray
    :return: Image in numpy format(RGB)
    :rtype: :class:`np.ndarray`
    """
    if img.shape[2] != 4:
        raise ValueError(
            "Only 4-channel RGBA images are supported for alpha channel removal. "
            + "Instead got {} channels.".format(img.shape[2])
        )
    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)


# @TODO: refactor from two separate methods to a single one
# bgra or bgr
def np_image_to_data_url(img: np.ndarray) -> str:
    """
    Convert image to url string.

    :param img: Image in numpy format(RGBA or RGB).
    :type img: np.ndarray
    :return: String with image url
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        data_url = sly.image.np_image_to_data_url(im)
        print(data_url)
        # Output: 'data:image/png;base64,iVBORw0K...'
    """
    encode_status, bgra_result_png = cv2.imencode(".png", img)
    img_png = bgra_result_png.tobytes()
    img_base64 = base64.b64encode(img_png)
    data_url = "data:image/png;base64,{}".format(str(img_base64, "utf-8"))
    return data_url


def data_url_to_numpy(data_url: str) -> np.ndarray:
    """
    Convert url string to numpy image(RGB).

    :param img: String with image url.
    :type img: str
    :return: Image in numpy format(RGBA)
    :rtype: :class:`np.ndarray`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        image_np = sly.image.data_url_to_numpy(data_url)
    """
    img_base64 = data_url[len("data:image/png;base64,") :]
    img_base64 = base64.b64decode(img_base64)
    image = read_bytes(img_base64)
    return image


# only rgb
def np_image_to_data_url_backup_rgb(img: np.ndarray) -> str:
    """
    Convert image to url string.

    :param img: Image in numpy format(only RGB).
    :type img: np.ndarray
    :return: String with image url
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        data_url = sly.image.np_image_to_data_url_backup_rgb(im)
        print(data_url)
        # Output: 'data:image/png;base64,iVBORw0K...'
    """
    img_base64 = base64.b64encode(write_bytes(img, "png"))
    data_url = "data:image/png;base64,{}".format(str(img_base64, "utf-8"))
    return data_url


def get_labeling_tool_url(team_id, workspace_id, project_id, dataset_id, image_id):
    res = f"/app/images/{team_id}/{workspace_id}/{project_id}/{dataset_id}#image-{image_id}"
    if is_development():
        res = abs_url(res)
    return res


def get_labeling_tool_link(url, name="open in labeling tool"):
    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{name}<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i></a>'
