#!/usr/bin/env python
# coding: utf-8

"""A module for reading NRRD files [NRRD1]_, basically a wrapper for calls on the pynrrd library [NRRD2]_.

References
----------
.. [NRRD1] http://teem.sourceforge.net/nrrd/format.html (20180212)
.. [NRRD2] https://github.com/mhe/pynrrd (20180212).
"""

import nrrd
import numpy as np

from loaders.volume import Volume


def open_image(path, verbose=True):
    """
    Open a 3D NRRD image at the given path.

    Parameters
    ----------
    path : str
        The path of the file to be loaded.
    verbose : bool, optional
        If `True` (default), print some meta data of the loaded file to standard output.

    Returns
    -------
    Volume
        The resulting 3D image volume, with the ``src_object`` attribute set to the tuple `(data, header)` returned
        by pynrrd's ``nrrd.read`` (where `data` is a Numpy array and `header` is a dictionary) and the desired
        anatomical world coordinate system ``system`` set to "RAS".

    Raises
    ------
    IOError
        If something goes wrong.
    """
    try:
        src_object = (voxel_data, hdr) = nrrd.read(path)
    except Exception as e:
        raise IOError(e)

    if verbose:
        print("Loading image:", path)
        print("Meta data:")
        for k in sorted(hdr.keys(), key=str.lower):
            print("{}: {!r}".format(k, hdr[k]))

    __check_data_kinds_in(hdr)
    src_system = __world_coordinate_system_from(
        hdr
    )  # No fixed world coordinates for NRRD images!
    mat = __matrix_from(hdr)  # Voxels to world coordinates

    # Create new ``Volume`` instance
    volume = Volume(
        src_voxel_data=voxel_data,
        src_transformation=mat,
        src_system=src_system,
        system="RAS",
        src_object=src_object,
    )
    return volume


def save_image(path, data, transformation, system="RAS", kinds=None):
    """
    Save the given image data as a NRRD image file at the given path.

    Parameters
    ----------
    path : str
        The path for the file to be saved.
    data : array_like
        Three-dimensional array that contains the voxels to be saved.
    transformation : array_like
        :math:`4×4` transformation matrix that maps from ``data``'s voxel indices to the given ``system``'s anatomical
        world coordinate system.
    system : str, optional
        The world coordinate system to which ``transformation`` maps the voxel data. Either "RAS" (default), "LAS", or
        "LPS" (these are the ones supported by the NRRD format).
    kinds : str or sequence of strings, optional
        If given, the string(s) will be used to set the NRRD header's "kinds" field. If a single string is given, it
        will be used for all dimensions. If multiple strings are given, they will be used in the given order. If
        nothing is given (default), the "kinds" field will not be set. Note that all strings should either be "domain"
        or "space".

    """
    if data.ndim > 3:
        raise RuntimeError(
            "Currently, only supports saving NRRD files with scalar data only!"
        )

    # Create the header entries from the transformation
    space = system.upper()
    space_directions = transformation[:3, :3].T.tolist()
    space_origin = transformation[:3, 3].tolist()
    options = {
        "space": space,
        "space directions": space_directions,
        "space origin": space_origin,
    }
    if kinds is not None:
        kinds = (data.ndim * [kinds]) if isinstance(kinds, str) else list(kinds)
        options["kinds"] = kinds
    nrrd.write(filename=path, data=data, header=options)


def save_volume(path, volume, src_order=True, src_system=True, kinds=None):
    """
    Save the given ``Volume`` instance as a NRRD image file at the given path.

    Parameters
    ----------
    path : str
        The path for the file to be saved.
    volume : Volume
        The ``Volume`` instance containing the image data to be saved.
    src_order : bool, optional
        If `True` (default), order the saved voxels as in ``src_data``; if `False`, order the saved voxels as in
        ``aligned_data``. In any case, the correct transformation matrix will be chosen.
    src_system : bool, optional
        If `True` (default), try to use ``volume``'s ``src_system`` as the anatomical world coordinate system for
        saving; if `False`, try to use ``volume``'s ``system`` instead. In either case, this works if the system is
        either "RAS", "LAS", or "LPS" (these are the ones supported by the NRRD format). If a different system is
        given, use "RAS" instead.
    kinds : str or sequence of strings, optional
        If given, the string(s) will be used to set the NRRD header's "kinds" field. If a single string is given, it
        will be used for all dimensions. If multiple strings are given, they will be used in the given order. If
        nothing is given (default), the "kinds" field will not be set. Note that all strings should either be "domain"
        or "space".
    """
    if volume.aligned_data.ndim > 3:
        raise RuntimeError(
            "Currently, only supports saving NRRD files with scalar data only!"
        )

    system = volume.src_system if src_system else volume.system
    system = system if system in ["RAS", "LAS", "LPS"] else "RAS"

    if src_order:
        data = volume.src_data
        transformation = volume.get_src_transformation(system)
    else:
        data = volume.aligned_data
        transformation = volume.get_aligned_transformation(system)

    save_image(
        path, data=data, transformation=transformation, system=system, kinds=kinds
    )


def __check_data_kinds_in(header):
    """
    Sanity check on the header's "kinds" field: are all entries either "domain" or "space" (i.e. are we really dealing
    with scalar data on a spatial domain)?

    Parameters
    ----------
    header : dict
        A dictionary containing the NRRD header (as returned by ``nrrd.read``, for example).

    Returns
    -------
    None
        Simply return if everything is ok or the "kinds" field is not set.

    Raises
    ------
    IOError
        If the "kinds" field contains entries other than "domain" or "space".
    """
    kinds = header.get("kinds")
    if kinds is None:
        return

    for k in kinds:
        if k.lower() not in ["domain", "space"]:
            raise IOError("At least one data dimension contains non-spatial data!")


def __world_coordinate_system_from(header):
    """
    From the given NRRD header, determine the respective assumed anatomical world coordinate system.

    Parameters
    ----------
    header : dict
        A dictionary containing the NRRD header (as returned by ``nrrd.read``, for example).

    Returns
    -------
    str
        The three-character uppercase string determining the respective anatomical world coordinate system (such as
        "RAS" or "LPS").

    Raises
    ------
    IOError
        If the header is missing the "space" field or the "space" field's value does not determine an anatomical world
        coordinate system.
    """
    try:
        system_str = header["space"]
    except KeyError as e:
        raise IOError(
            "Need the header's \"space\" field to determine the image's anatomical coordinate system."
        )

    if len(system_str) == 3:
        # We are lucky: this is already the format that we need
        return system_str.upper()

    # We need to separate the string (such as "right-anterior-superior") at its dashes, then get the first character
    # of each component. We cannot handle 4D data nor data with scanner-based coordinates ("scanner-...") or
    # non-anatomical coordinates ("3D-...")
    system_components = system_str.split("-")
    if len(system_components) == 3 and not system_components[0].lower() in [
        "scanner",
        "3d",
    ]:
        system_str = "".join(c[0].upper() for c in system_components)
        return system_str

    raise IOError('Cannot handle "space" value {}'.format(system_str))


def __matrix_from(header):
    """
    Calculate the transformation matrix from voxel coordinates to the header's anatomical world coordinate system.

    Parameters
    ----------
    header : dict
        A dictionary containing the NRRD header (as returned by ``nrrd.read``, for example).

    Returns
    -------
    numpy.ndarray
        The resulting :math:`4×4` transformation matrix.
    """
    try:
        space_directions = header["space directions"]
        space_origin = header["space origin"]
    except KeyError as e:
        raise IOError(
            'Need the header\'s "{}" field to determine the mapping from voxels to world coordinates.'.format(
                e
            )
        )

    # "... the space directions field gives, one column at a time, the mapping from image space to world space
    # coordinates ... [1]_" -> list of columns, needs to be transposed
    trans_3x3 = np.array(space_directions).T
    trans_4x4 = np.eye(4)
    trans_4x4[:3, :3] = trans_3x3
    trans_4x4[:3, 3] = space_origin
    return trans_4x4
