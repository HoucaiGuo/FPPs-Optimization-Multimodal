import rasterio
import numpy as np


def read_image(img_path, nodata=np.nan):
    """
    Read a TIFF image file.

    Parameters
    ----------
    img_path : str
        Image file's name.
    nodata : any
        Image's nodata value, which is numpy.nan by default. Users can specify a nodata value if
            there are nodata pixels in the image but not recorded in the TIFF's profile.

    Returns
    -------
    img_arr : numpy.array
        3D image array, (H, W, C) shaped, masked pixels denote nodata.
    profile : dict
        Image's profile.
    """
    reader = rasterio.open(img_path, mode='r')
    profile = reader.profile
    # if the nodata value is recorded in the TIFF file, use it.
    # otherwise use the "nodata" parameter
    if profile["nodata"] is not None:
        nodata = profile["nodata"]
    # record the nodata value in the profile
    else:
        profile["nodata"] = nodata
    val = reader.read()
    # use numpy.isnan() to check if the nodata value is numpy.nan
    img = np.ma.array(val,
                      mask=np.where(np.isnan(val) if np.isnan(nodata) else val == nodata, True, False),
                      dtype=np.dtype(profile["dtype"]))
    # (C, H, W) to (H, W, C)
    img = np.ma.transpose(img, (1, 2, 0))

    return img, profile


def check_dim(img):
    """
    Check if the image is (H, W, C) shaped. If not, expand it's dimension from (H, W) to (H, W, 1).

    Parameters
    ----------
    img : numpy.array
        2D or 3D image array, (H, W) or (H, W, C) shaped.

    Returns
    -------
    img : numpy.array
        3D image array, (H, W, C) shaped.
    """
    if len(img.shape) == 2:
        img = np.ma.expand_dims(img, axis=2)
    return img


def normalize_band(band):
    """
    Normalize a band array.

    Parameters
    ----------
    band : numpy.array
        2D or 3D image array, (H, W) or (H, W, C) shaped, masked pixels denote nodata.

    Returns
    -------
    band_norm : numpy.array
        Normalized band array.
    """
    min_b = band.min()
    max_b = band.max()
    band_norm = (band - min_b) / (max_b - min_b)

    return band_norm


def normalize_image(img):
    """
    Normalize an image.

    Parameters
    ----------
    img : numpy.array
        3D image array, (H, W, C) shaped, masked pixels denote nodata.

    Returns
    -------
    img_norm : numpy.array
        Normalized image array.
    """
    img = check_dim(img)
    img_norm = np.ma.stack([normalize_band(img[:, :, i]) for i in range(img.shape[2])], axis=2)

    return img_norm


def save_image(img, img_path, profile={}):
    """
    Save an image to the disk.

    Parameters
    ----------
    img : numpy.array
        3D image array, (H, W, C) shaped, masked pixels denote nodata.
    img_path : str
        The path of which to save the image.
    profile : dict
        The profile of the image.
    """
    img = check_dim(img)
    profile["dtype"] = str(img.dtype)
    # nodata = img.data[img.mask][0]
    # if np.any(img == nodata):
    #     raise Exception(f"Image's data contains the nodata value {nodata}, please change the masked value in the image,"
    #                     f" e.g. img.data[img.mask] = new_nodata_val.")
    # profile["nodata"] = nodata
    profile["height"] = img.shape[0]
    profile["width"] = img.shape[1]
    profile["count"] = img.shape[2]

    # (H, W, C) to (C, H ,W)
    img = np.ma.transpose(img, (2, 0, 1))
    writer = rasterio.open(img_path, mode='w', **profile)
    writer.write(img.data)
