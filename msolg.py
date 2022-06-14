from phasepack import phasecong
import numpy as np
from skimage.metrics import structural_similarity


def msolg(img, nscale=4, norient=6):
    """
    Extract multi-scale and multi-orientation Log-Gabor (MSOLG) features from image.

    Parameters
    ----------
    img : numpy.array
        The input image.
    nscale : int
        The number of scale.
    norient : int
        The number of orientation.

    Returns
    -------
    msolg_features : numpy.array
        The extracted MSOLG features.
    """
    eo = phasecong(img, nscale, norient)[5]
    msolg_features = np.zeros(shape=(img.shape[0], img.shape[1], norient), dtype=np.float32)

    for o in range(norient):
        mul_s = np.zeros(shape=(img.shape[0], img.shape[1], nscale), dtype=np.float32)
        for s in range(nscale):
            mul_s[:, :, s] = np.abs(eo[o][s]).astype(np.float32)

        # sum by scale, as Equation (5) in the letter
        msolg_features[:, :, o] = np.sum(mul_s, axis=2)

    return msolg_features


def ssim_of_msolg(img1, img2, nscale=4, norient=6, full=False):
    """
    Calculate the structural similarity of MSOLG features. This is a new similarity metric of multimodal remote
    sensing images.

    Parameters
    ----------
    img1 : numpy.array
        The fixed image.
    img2 : numpy.array
        The moving image.
    nscale : int
        The number of scale.
    norient : int
        The number of orientation.
    full : bool
        Return the full similarity map or not.

    Returns
    -------
    msolg_ssim : float
        The structural similarity of MSOLG features.
    """
    img1_msolg = msolg(img1, nscale, norient)
    img2_msolg = msolg(img2, nscale, norient)
    msolg_ssim = structural_similarity(img1_msolg, img2_msolg, gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=True, multichannel=True, full=full)

    return msolg_ssim
