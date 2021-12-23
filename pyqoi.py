"""Pure Python implementation of QOI - the Quite OK Image format."""
import abc
import argparse
import dataclasses
import enum
import io
import struct
from typing import Optional, Protocol, Sequence, runtime_checkable

__version__ = "0.0.2"


QOI_MAGIC = b"qoif"
"""Magic bytes identifying the QOI format."""


class PyqoiException(Exception):
    """Base Exception class for pyqoi errors."""

    def __init__(self, msg: str):
        """Specify a generic human-readable error message."""
        super().__init__(msg)


class PyqoiDecodeError(PyqoiException, ValueError):
    """For pyqoi errors during decoding QOI data."""

    def __init__(self, msg: str, header: bytes):
        """Specify a decoding error message and file header."""
        self.header = header
        super().__init__(msg)


@runtime_checkable
class SupportsRead(Protocol):
    """An ABC with one abstract method *read*."""

    @abc.abstractmethod
    def read(self, size: Optional[int] = -1) -> bytes:
        """Read and return up to *size* bytes.

        If the argument is omitted, `None`, or negative,
        data is read and returned until EOF is reached.
        """


class QOIChannels(enum.IntEnum):
    """Valid values for the "channels" field of a QOI header.

    The member values represent the integer values in the QOI header.
    """

    RGB = 3
    RGBA = 4


class QOIColorSpace(enum.IntEnum):
    """Valid values for the "colorspace" field of a QOI header.

    The member values represent the integer values in the QOI header.
    """

    SRGB = 0
    LINEAR = 1


@dataclasses.dataclass
class QOIHeader:
    """Metadata in the file header.

    Note: the "colorspace" and "channel" fields are purely informative.
    They do not change how an image is encoded or decoded.
    """

    width: int
    """Image width in pixels.

    Minimum value should be 0, maximum value should be (23 ** 2) - 1.
    """

    height: int
    """Image height in pixels.

    Minimum value should be 0, maximum value should be (23 ** 2) - 1.
    """

    channels: QOIChannels
    """The colour channels used in this image.

    Should be either 3 for RGB for 4 for RGBA.
    """

    colorspace: QOIColorSpace
    """The colour space this image is designed for.

    Should either be 0 for sRGB or 1 for linear.
    """


Pixel = tuple[int, int, int, int]
"""Data representing the red, green, blue, and alpha channels of one pixel."""


@dataclasses.dataclass
class QOIData:
    """Full data from a QOI file."""

    header: QOIHeader
    """File header containing metadata."""

    pixels: Sequence[Pixel]
    """Raw image data.

    This is a "flat" structure - one row of pixels leads on from the next.
    """


QOI_HEADER_STRUCT = struct.Struct(
    ">"  # big-endian
    "4s"  # magic uint32_t magic bytes "qoif"
    "I"  # width uint32_t image width in pixels
    "I"  # height uint32_t image height in pixels
    "B"  # channels uint8_t 3 = RGB, 4 = RGBA
    "B"  # colorspace uint8_t 0 = sRGB with linear alpha, 1 = all channels linear
)


def load_header_bytes(header: bytes) -> QOIHeader:
    """Read the header data of a QOI file.

    Raises a PyqoiDecodeError if the header is invalid.
    """
    # check header size
    if len(header) != QOI_HEADER_STRUCT.size:
        raise PyqoiDecodeError(
            "header is wrong size, "
            f"must be {QOI_HEADER_STRUCT.size} bytes, is {len(header)} bytes",
            header,
        )

    # parse header fields
    magic, width, height, channels, colour_space = QOI_HEADER_STRUCT.unpack(header)

    # check magic
    if magic != QOI_MAGIC:
        raise PyqoiDecodeError(
            f"magic bytes are incorrect, should be {QOI_MAGIC!r}, is {magic!r}", header
        )

    # width and height can't be incorrect due to parsing method

    # check channels field
    try:
        channels_enum = QOIChannels(channels)
    except ValueError:
        channels_values = set(c.value for c in QOIChannels)
        raise PyqoiDecodeError(
            "channels header field is incorrect, "
            f"must be one of {channels_values}, is {channels}",
            header,
        )

    # check colorspace field
    try:
        colour_space_enum = QOIColorSpace(colour_space)
    except ValueError:
        colour_space_values = set(c.value for c in QOIColorSpace)
        raise PyqoiDecodeError(
            "colorspace header field is incorrect, "
            f"must be one of {colour_space_values}, is {colour_space}",
            header,
        )

    return QOIHeader(width, height, channels_enum, colour_space_enum)


def load_header_file(f: SupportsRead) -> QOIHeader:
    """Read the header from a QOI file.

    Raises a PyqoiDecodeError if the header is invalid.
    """
    return load_header_bytes(f.read(QOI_HEADER_STRUCT.size))


def loads(data: bytes) -> QOIData:
    """Read a QOI file.

    Raises a PyqoiDecodeError if the image is invalid.
    """
    return load(io.BytesIO(data))


def load(f: SupportsRead) -> QOIData:
    """Read the data of a QOI file.

    Raises a PyqoiDecodeError if the image is invalid.
    """
    header = load_header_file(f)
    # todo
    return QOIData(header, [])


def __main__(argv: Optional[Sequence[str]] = None):
    """Command-line interface to pyqoi.

    pyqoi is a pure Python implementation of QOI - the Quite OK Image format.

    Reads the header of a QOI file and prints the metadata.
    """
    parser = argparse.ArgumentParser(description=__main__.__doc__)
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("infile", type=argparse.FileType("rb"))
    args = parser.parse_args(argv)
    print(load_header_file(args.infile))


if __name__ == "__main__":
    __main__()
