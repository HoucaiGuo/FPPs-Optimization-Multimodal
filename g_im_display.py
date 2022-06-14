import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import g_im_stats
from g_im_io import check_dim


def color_composite(img, bands_idx):
    """
    Color composite of an image, idx denotes the R, G, and B bands' index in the image.

    Parameters
    ----------
    img : numpy.array
        3D image array, (H, W, C) shaped, masked pixels denote nodata.
    bands_idx : list of int
        List of the RGB bands' index.

    Returns
    -------
    img_rgb : numpy.array
        Color composited image.
    """
    img_rgb = np.ma.stack([img[:, :, i] for i in bands_idx], axis=2)
    return img_rgb


def linear_pct_stretch(img, pct):
    """
    A linear percent stretch allows you to trim extreme values from both ends of the histogram using a specified
    percentage. For example, assume that the pixel values in an image range from a to b. If you select a 2% linear
    stretch, the lowest 2% of histogram values are less than c and the highest 2% are greater than d. Values less
    than c are set to the min value of this image, and values greater than d are set to the max value of this image.
    Values in between are distributed from [min, max].

    Parameters
    ----------
    img : numpy.array
        3D image array, (H, W, C) shaped, masked pixels denote nodata.
    pct : int
        The percentage you want to stretch, normally, use 2 or 5.

    Returns
    -------
    img_pct_strch : numpy.array
        The linear percent stretched image.
    """
    img = check_dim(img)
    bands = []
    for i in range(img.shape[2]):
        bins, counts = g_im_stats.img_hist(img[:, :, i])[0]
        freq_pct = np.cumsum(counts, dtype=np.longdouble) / np.sum(counts, dtype=np.float64) * 100

        left = np.argmax(freq_pct >= pct)
        right = np.argmax(freq_pct >= 100 - pct)
        # choose the value which is nearer to pct
        a = bins[left] if freq_pct[left] - pct <= pct - freq_pct[left - 1] else bins[left - 1]
        b = bins[right] if freq_pct[right] - (100 - pct) <= (100 - pct) - freq_pct[right - 1] else bins[right - 1]
        c = img[:, :, i].min()
        d = img[:, :, i].max()

        # print(f"black point: {a} {bins[left-2:left+3]} {freq_pct[left-2:left+3]}, \n\t"
        #       f"white point: {b} {bins[right-2:right+3]} {freq_pct[right-2:right+3]}")

        band_strch = np.ma.where(img[:, :, i] < a, c, img[:, :, i])
        band_strch = np.ma.where(band_strch > b, d, band_strch)
        band_strch = np.ma.where((band_strch >= a) & (band_strch <= b),
                                 (d - c) * (band_strch - a) / (b - a) + c,
                                 band_strch)
        bands.append(band_strch)

    img_pct_strch = np.ma.stack(bands, axis=2)

    return img_pct_strch


def optimized_lin_stretch(img):
    """


    Parameters
    ----------
    img :

    Returns
    -------

    """
    img = check_dim(img)
    bands = []
    min_pct = 0.025
    max_pct = 0.99
    min_adj_pct = 0.1
    max_adj_pct = 0.5
    for i in range(img.shape[2]):
        bins, counts = g_im_stats.img_hist(img[:, :, i])[0]
        freq_pct = np.cumsum(counts, dtype=np.longdouble) / np.sum(counts, dtype=np.float64)

        a = bins[np.argmax(freq_pct >= min_pct) - 1]
        b = bins[np.argmax(freq_pct >= max_pct)]
        c = a - min_adj_pct * (b - a)
        d = b + max_adj_pct * (b - a)
        band_min = np.nanmin(img[:, :, i])
        band_max = np.nanmax(img[:, :, i])

        band_strch = np.where((img[:, :, i] < c) & (~np.isnan(img[:, :, i])),
                              band_min, img[:, :, i])
        band_strch = np.where((img[:, :, i] > d) & (~np.isnan(band_strch)),
                              band_max, img[:, :, i])
        band_strch = np.where((band_strch >= band_min) & (band_strch <= band_max) & (~np.isnan(band_strch)),
                              (band_max - band_min) * (band_strch - c) / (d - c) + band_min,
                              band_strch)
        bands.append(band_strch)

    img_optim_lin_strch = np.ma.stack(bands, axis=2)

    return img_optim_lin_strch


def show_hist(img, axis, cumulative=False, density=False, title="Histogram"):
    """
    Show the histogram of an image.

    Parameters
    ----------
    img : numpy.array
        3D image array, (H, W, C) shaped, masked pixels denote nodata.
    axis : matplotlib.axis.Axis
        The axis of which to plot the histogram.
    density : bool
        If True, the Y label denotes the percentage of the corresponding pixel, else the count.
    cumulative : bool
        If True, draw a cumulative histogram of the image.
    """
    img = check_dim(img)
    # hists is a list of tuple, (bins, counts)
    hists = g_im_stats.img_hist(img)
    for i in range(img.shape[2]):
        bins, counts = hists[i]
        if density:
            # use percent
            counts = np.divide(counts, g_im_stats.img_pixel_count(img[:, :, i]), dtype=np.float64) * 100
        if cumulative:
            counts = np.cumsum(counts)
        axis.plot(bins, counts)
    axis.get_xaxis().get_major_formatter().set_scientific(False)
    axis.get_yaxis().get_major_formatter().set_scientific(False)
    axis.set_title(title)
    axis.set_xlabel("Pixel value")
    axis.set_ylabel("Percent" if density else "Count")
    axis.legend([f"Band {i + 1}" for i in range(img.shape[2])], loc="upper right")


def equalize_hist(img):
    img = check_dim(img)
    bands = []
    for i in range(img.shape[2]):
        band_eq_hist = np.ma.array(data=exposure.equalize_hist(img[:, :, i].data,
                                                               nbins=np.ma.unique(img[:, :, i]).shape[0],
                                                               mask=~img[:, :, i].mask),
                                   mask=img[:, :, i].mask, dtype=np.float32)
        bands.append(band_eq_hist)
    img_eq_hist = np.ma.stack(bands, axis=2)

    return img_eq_hist


def match_hist(img, ref):
    img = check_dim(img)
    ref = check_dim(ref)
    matched = np.ma.array(exposure.match_histograms(img, ref,
                                                    multichannel=True if img.shape[2] > 1 else False),
                          mask=img.mask, dtype=np.float32)

    return matched


def show_image(img, axis, cmap="gray", title="Image"):
    img = check_dim(img)

    axis.imshow(img, cmap=cmap)
    axis.set_title(title)








