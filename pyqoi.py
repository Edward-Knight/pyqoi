"""Pure Python implementation of QOI - the Quite OK Image format."""
import abc
import argparse
import dataclasses
import enum
import struct
from typing import Optional, Protocol, Sequence, runtime_checkable

__version__ = "0.0.1"


@runtime_checkable
class SupportsRead(Protocol):
    """An ABC with one abstract method _read_."""

    __slots__ = ()

    @abc.abstractmethod
    def read(self, size: Optional[int] = -1) -> bytes:
        """Read and return up to _size_ bytes.

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


QOI_HEADER_STRUCT = struct.Struct(
    ">"  # big-endian
    "4s"  # magic uint32_t magic bytes "qoif"
    "I"  # width uint32_t image width in pixels
    "I"  # height uint32_t image height in pixels
    "B"  # channels uint8_t 3 = RGB, 4 = RGBA
    "B"  # colorspace uint8_t 0 = sRGB with linear alpha, 1 = all channels linear
)


def load(f: SupportsRead) -> QOIHeader:
    """Load a QOI file.

    Only loads and returns the header for now.
    """
    magic, width, height, channels, colour_space = QOI_HEADER_STRUCT.unpack(
        f.read(QOI_HEADER_STRUCT.size)
    )
    assert magic == b"qoif"
    return QOIHeader(width, height, QOIChannels(channels), QOIColorSpace(colour_space))


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
    print(load(args.infile))


if __name__ == "__main__":
    __main__()
