"""Pure Python implementation of QOI - the Quite OK Image format."""
import abc
import argparse
import dataclasses
import enum
import io
import struct
from typing import Optional, Protocol, Sequence, runtime_checkable

__version__ = "0.0.3"


QOI_MAGIC = b"qoif"
"""Magic bytes identifying the QOI format."""


class QOIOpTag(enum.IntEnum):
    """Bit patterns used for determining data chunk types."""

    QOI_OP_RGB = 0b11111110
    QOI_OP_RGBA = 0b11111111
    QOI_OP_INDEX = 0b00
    QOI_OP_DIFF = 0b01
    QOI_OP_LUMA = 0b10
    QOI_OP_RUN = 0b11


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

    def to_bytes(self) -> bytes:
        """Return the bytes representation of this header, with magic."""
        return QOI_HEADER_STRUCT.pack(
            QOI_MAGIC,
            self.width,
            self.height,
            self.channels.value,
            self.colorspace.value,
        )


class PyqoiDecodeEOFError(PyqoiDecodeError, EOFError):
    """Raise when we hit EOF trying to decode QOI data."""

    def __init__(self, header: QOIHeader):
        """Specify the file header."""
        super().__init__("EOF while decoding QOI data", header.to_bytes())


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


def hash_index(r: int, g: int, b: int, a: int) -> int:
    """Hash function to find the index in the previously seen pixel array.

    A running array[64] (zero-initialized) of previously seen pixel
    values is maintained by the encoder and decoder. Each pixel that is
    seen by the encoder and decoder is put into this array at the
    position formed by a hash function of the color value. In the
    encoder, if the pixel value at the index matches the current pixel,
    this index position is written to the stream as QOI_OP_INDEX. The
    hash function for the index is:

    index_position = (r * 3 + g * 5 + b * 7 + a * 11) % 64
    """
    return (r * 3 + g * 5 + b * 7 + a * 11) % 64


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
    data: list[Pixel] = []
    previous_pixels: list[Optional[Pixel]] = [(0, 0, 0, 0)] * 64
    last_pixel: Pixel = (0, 0, 0, 255)
    index_0_count = 0
    chunk_count = 0

    while True:
        tag_byte = f.read(1)
        tag_8 = int.from_bytes(tag_byte, "big")
        tag_2 = tag_8 >> 6
        data_6 = tag_8 & 0b00111111
        match tag_8:
            case QOIOpTag.QOI_OP_RGB:
                index_0_count = 0
                pixel = (*f.read(3), last_pixel[3])
                if len(pixel) < 4:
                    raise PyqoiDecodeEOFError(header)
            case QOIOpTag.QOI_OP_RGBA:
                index_0_count = 0
                pixel = (*f.read(4),)
                if len(pixel) < 4:
                    raise PyqoiDecodeEOFError(header)
            case _:
                match tag_2:
                    case QOIOpTag.QOI_OP_INDEX:
                        # data_6 is index into the previous_pixels array

                        # EOF check
                        # instead of peeking 8 bytes each loop we do the EOF check here
                        # after 7 consecutive QOI_OP_INDEX chunks with an index of 0
                        if data_6 == 0:
                            index_0_count += 1
                        else:
                            index_0_count = 0
                        if index_0_count == 7:
                            # next byte should be 0x01 indicating EOF
                            eof_byte = f.read(1)
                            if eof_byte == b"\x01":
                                # delete the last 6 pixels - these were not actually QOI_OP_INDEX chunks, but an EOF marker
                                data = data[:-6]
                                break
                            else:
                                raise PyqoiDecodeError(
                                    f"Malformed EOF marker, file has {index_0_count} consecutive QOI_OP_INDEX chunks with an index of 0",
                                    header=header.to_bytes(),
                                )
                        pixel = previous_pixels[data_6]
                    case QOIOpTag.QOI_OP_DIFF:
                        index_0_count = 0
                        dr = (data_6 >> 4) - 2
                        dg = ((data_6 & 0b001100) >> 2) - 2
                        db = (data_6 & 0b000011) - 2
                        pixel = (
                            (last_pixel[0] + dr) % 256,
                            (last_pixel[1] + dg) % 256,
                            (last_pixel[2] + db) % 256,
                            last_pixel[3],
                        )
                    case QOIOpTag.QOI_OP_LUMA:
                        index_0_count = 0
                        dg = data_6 - 32
                        second_byte = int.from_bytes(f.read(1), "big")
                        dr = (second_byte >> 4) - 8 + dg
                        db = (second_byte & 0b00001111) - 8 + dg
                        pixel = (
                            (last_pixel[0] + dr) % 256,
                            (last_pixel[1] + dg) % 256,
                            (last_pixel[2] + db) % 256,
                            last_pixel[3],
                        )
                    case QOIOpTag.QOI_OP_RUN:
                        index_0_count = 0
                        run_length = data_6 + 1
                        if run_length < 1:
                            raise PyqoiDecodeError(
                                f"QOI_OP_RUN data chunk is invalid, run length is {run_length}",
                                header=header.to_bytes(),
                            )
                        data.extend([last_pixel] * (run_length - 1))
                        pixel = last_pixel

        data.append(pixel)
        previous_pixels[hash_index(*pixel)] = pixel
        last_pixel = pixel
        chunk_count += 1

    return QOIData(header, data)


def __main__(argv: Optional[Sequence[str]] = None):
    """Command-line interface to pyqoi.

    pyqoi is a pure Python implementation of QOI - the Quite OK Image format.

    Displays a QOI file and prints the metadata.
    """
    parser = argparse.ArgumentParser(description=__main__.__doc__)
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("infile", type=argparse.FileType("rb"))
    args = parser.parse_args(argv)
    data = load(args.infile)

    import numpy as np
    from PIL import Image

    Image.fromarray(
        np.asarray(data.pixels, dtype="uint8").reshape(
            (data.header.height, data.header.width, 4)
        )
    ).show()
    print(data.header)


if __name__ == "__main__":
    __main__()
