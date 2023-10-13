import io
from collections import OrderedDict
from datetime import datetime
from nrrd.reader import _get_field_type
from nrrd.writer import (
    _TYPEMAP_NUMPY2NRRD,
    _NUMPY2NRRD_ENDIAN_MAP,
    _NRRD_FIELD_ORDER,
    _format_field_value,
    _write_data,
)


def encode(
    data, header=None, custom_field_map=None, compression_level=9, index_order="F"
):
    if header is None:
        header = {}

    # Infer a number of fields from the NumPy array and overwrite values in the header dictionary.
    # Get type string identifier from the NumPy datatype
    header["type"] = _TYPEMAP_NUMPY2NRRD[data.dtype.str[1:]]

    # If the datatype contains more than one byte and the encoding is not ASCII, then set the endian header value
    # based on the datatype's endianness. Otherwise, delete the endian field from the header if present
    if data.dtype.itemsize > 1 and header.get("encoding", "").lower() not in [
        "ascii",
        "text",
        "txt",
    ]:
        header["endian"] = _NUMPY2NRRD_ENDIAN_MAP[data.dtype.str[:1]]
    elif "endian" in header:
        del header["endian"]

    # If space is specified in the header, then space dimension can not. See
    # http://teem.sourceforge.net/nrrd/format.html#space
    if "space" in header.keys() and "space dimension" in header.keys():
        del header["space dimension"]

    # Update the dimension and sizes fields in the header based on the data. Since NRRD expects meta data to be in
    # Fortran order we are required to reverse the shape in the case of the array being in C order. E.g., data was read
    # using index_order='C'.
    header["dimension"] = data.ndim
    header["sizes"] = list(data.shape) if index_order == "F" else list(data.shape[::-1])

    # The default encoding is 'gzip'
    if "encoding" not in header:
        header["encoding"] = "gzip"

    # Remove detached data filename from the header
    if "datafile" in header:
        header.pop("datafile")

    if "data file" in header:
        header.pop("data file")

    with io.BytesIO() as fh:
        fh.write(b"NRRD0005\n")

        # Copy the options since dictionaries are mutable when passed as an argument
        # Thus, to prevent changes to the actual options, a copy is made
        # Empty ordered_options list is made (will be converted into dictionary)
        local_options = header.copy()
        ordered_options = []

        # Loop through field order and add the key/value if present
        # Remove the key/value from the local options so that we know not to add it again
        for field in _NRRD_FIELD_ORDER:
            if field in local_options:
                ordered_options.append((field, local_options[field]))
                del local_options[field]

        # Leftover items are assumed to be the custom field/value options
        # So get current size and any items past this index will be a custom value
        custom_field_start_index = len(ordered_options)

        # Add the leftover items to the end of the list and convert the options into a dictionary
        ordered_options.extend(local_options.items())
        ordered_options = OrderedDict(ordered_options)

        for x, (field, value) in enumerate(ordered_options.items()):
            # Get the field_type based on field and then get corresponding
            # value as a str using _format_field_value
            field_type = _get_field_type(field, custom_field_map)
            value_str = _format_field_value(value, field_type)

            # Custom fields are written as key/value pairs with a := instead of : delimeter
            if x >= custom_field_start_index:
                fh.write(("%s:=%s\n" % (field, value_str)).encode("ascii"))
            else:
                fh.write(("%s: %s\n" % (field, value_str)).encode("ascii"))

        # Write the closing extra newline
        fh.write(b"\n")

        _write_data(
            data,
            fh,
            header,
            compression_level=compression_level,
            index_order=index_order,
        )

        fh.seek(0)
        return fh.read()
