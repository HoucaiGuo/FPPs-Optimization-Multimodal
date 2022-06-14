from scipy import stats
import numpy as np
from g_im_io import check_dim
from sklearn import metrics
from scipy.ndimage import standard_deviation
from skimage.exposure import histogram


def img_max(img, mode="image"):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.max(img, axis=2), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.max(img, axis=axis)


def img_mean(img, mode="image", dtype=np.longdouble):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.mean(img, axis=2, dtype=dtype), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.mean(img, axis=axis, dtype=dtype)


def img_median(img, mode="image"):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.median(img, axis=2), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.median(img, axis=axis)


def img_min(img, mode="image"):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.min(img, axis=2), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.min(img, axis=axis)


def img_pixel_count(img, mode="image"):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.count(img, axis=2), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.count(img, axis=axis)


def img_pixel_vals(img):
    img = check_dim(img)
    return [img[:, :, i].data[~img[:, :, i].mask] for i in range(img.shape[2])]


def img_stddev(img, mode="image", dtype=np.longdouble, ddof=0):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.std(img, axis=2, dtype=dtype, ddof=ddof), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.std(img, axis=axis, dtype=dtype, ddof=ddof)


def img_sum(img, mode="image", dtype=np.longdouble):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.sum(img, axis=2, dtype=dtype), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.sum(img, axis=axis, dtype=dtype)


def img_var(img, mode="image", dtype=np.longdouble, ddof=0):
    img = check_dim(img)
    axis = None
    if mode == "pixel":
        return np.ma.expand_dims(np.ma.var(img, axis=2, dtype=dtype, ddof=ddof), axis=2)
    elif mode == "band":
        axis = (0, 1)
    return np.ma.var(img, axis=axis, dtype=dtype, ddof=ddof)


def img_cov(img1, img2, mode="image", dtype=np.longdouble, ddof=0):
    img1 = check_dim(img1)
    img2 = check_dim(img2)
    img1_bar = img_mean(img1, mode=mode, dtype=dtype)
    img2_bar = img_mean(img2, mode=mode, dtype=dtype)
    cov = img_sum((img1 - img1_bar) * (img2 - img2_bar), mode=mode, dtype=dtype) / (
                img_pixel_count(img1, mode=mode) - ddof)

    return cov


def img_corrcoef(img1, img2, mode="image", dtype=np.longdouble, ddof=0):
    img1 = check_dim(img1)
    img2 = check_dim(img2)

    cov = img_cov(img1, img2, mode=mode, dtype=dtype, ddof=ddof)
    img1_stddev = img_stddev(img1, mode=mode, dtype=dtype, ddof=ddof)
    img2_stddev = img_stddev(img2, mode=mode, dtype=dtype, ddof=ddof)
    cc = cov / (img1_stddev * img2_stddev)

    # print(f"cov: {cov}, stddev1: {img1_stddev}, stddev2: {img2_stddev}")

    return cc


def img_mutual_info(img1, img2, mode="image", normalize=True):
    img1 = check_dim(img1)
    img2 = check_dim(img2)

    if mode == "image":
        val1 = img1.data[~img1.mask]
        val2 = img2.data[~img2.mask]
        if normalize:
            mi = metrics.normalized_mutual_info_score(val1, val2)
        else:
            mi = metrics.mutual_info_score(val1, val2)
    else:
        if normalize:
            mi = np.array([metrics.normalized_mutual_info_score(img1[:, :, i].data[img1[:, :, i].mask],
                                                                img2[:, :, i].data[img2[:, :, i].mask])
                           for i in range(img1.shape[2])])
        else:
            mi = np.array([metrics.mutual_info_score(img1[:, :, i].data[img1[:, :, i].mask],
                                                     img2[:, :, i].data[img2[:, :, i].mask])
                           for i in range(img1.shape[2])])

    return mi


def img_mode(img, mode="image"):
    img = check_dim(img)
    axis = None
    if mode == "band":
        axis = (0, 1)
    elif mode == "pixel":
        axis = 2
    return stats.mode(img, axis=axis)


def img_hist(img):
    img = check_dim(img)
    hists = []
    vals = img_pixel_vals(img)
    for i in range(img.shape[2]):
        # numpy.ma.unique() can not return count, use numpy.unique() instead
        bins, counts = np.unique(vals[i], return_counts=True)  # np.ma.unique(vals[i], return_counts=True)
        hist = (bins, counts)
        hists.append(hist)

    return hists
