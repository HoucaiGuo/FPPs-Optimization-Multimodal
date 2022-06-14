from g_im_io import *
from g_im_display import *
import matplotlib.pyplot as plt
import numpy as np
from functions import *
import matplotlib.patches as patches
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import matplotlib as mpl

mpl.rc('font', family='Times New Roman')

test = "optical-sar-2"

if __name__ == "__main__":
    f_img_path = f"final_data/{test}/f_img.tif"
    optim_img_path = f"final_data/{test}/ssim_of_msolg/ns2_no6_optim_img.jpg"

    f_img, prof = read_image(f_img_path)
    f_img = normalize_image(f_img)
    # f_img = rgb2gray(f_img)
    optim_img = read_image(optim_img_path)[0][:, :, 0]
    optim_img = normalize_image(optim_img)

    # f_img = np.concatenate([f_img, f_img, f_img], axis=2)
    # optim_img = np.concatenate([optim_img, optim_img, optim_img], axis=2)

    if test == "optical-infrared-2":
        step = 80
        f_img = np.concatenate([f_img, f_img, f_img], axis=2)
        optim_img = np.concatenate([optim_img, optim_img, optim_img], axis=2)
    elif test == "optical-lidar-2":
        step = 35
        optim_img = np.concatenate([optim_img, optim_img, optim_img], axis=2)
    else:
        # step = 40
        step = 50
        optim_img = np.concatenate([optim_img, optim_img, optim_img], axis=2)

    cb_img = checker_board(f_img, optim_img, step=step, hist_match=False)
    # cb_rgb = color_composite(linear_pct_stretch(cb_img, 1), [2, 1, 0])
    # cb_img = linear_pct_stretch(cb_img, 1)

    cm_2_inc = 1 / 2.54
    fig = plt.figure(figsize=[4.5 * cm_2_inc, 4.5 * cm_2_inc])
    rect = 0, 0, 1, 1
    axis = fig.add_axes(rect)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    plt.imshow(cb_img, cmap="gray")

    if test == "optical-infrared-2":
        rect_a = patches.Rectangle((160, 320), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
        rect_b = patches.Rectangle((560, 400), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
        axis.add_patch(rect_a)
        axis.text(170, 310, "a", color="red", fontsize=15)
        axis.add_patch(rect_b)
        axis.text(570, 390, "b", color="red", fontsize=15)
    elif test == "optical-lidar-2":
        rect_a = patches.Rectangle((210, 315), 70, 70, linewidth=1, edgecolor='r', facecolor='none')
        rect_b = patches.Rectangle((385, 105), 70, 70, linewidth=1, edgecolor='r', facecolor='none')
        axis.add_patch(rect_a)
        axis.text(220, 305, "a", color="red", fontsize=15)
        axis.add_patch(rect_b)
        axis.text(395, 95, "b", color="red", fontsize=15)
    else:
        rect_a = patches.Rectangle((100, 150), 100, 100, linewidth=1, edgecolor='r', facecolor='none')
        rect_b = patches.Rectangle((250, 200), 100, 100, linewidth=1, edgecolor='r', facecolor='none')
        axis.add_patch(rect_a)
        axis.text(110, 140, "a", color="red", fontsize=15)
        axis.add_patch(rect_b)
        axis.text(260, 190, "b", color="red", fontsize=15)

    plt.savefig(f"figures/cb/{test}-cb.jpg", dpi=900)
    plt.show()

    if test == "optical-infrared-2":
        cm_2_inc = 1 / 2.54
        fig = plt.figure(figsize=[4.5 * cm_2_inc, 4.5 * cm_2_inc])
        rect = 0, 0, 1, 1
        axis = fig.add_axes(rect)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.imshow(cb_img[320:480, 160:320], cmap="gray")
        # plt.savefig(f"figures/cb/{test}-cb-sub1.jpg", dpi=900)
        plt.show()

        cm_2_inc = 1 / 2.54
        fig = plt.figure(figsize=[4.5 * cm_2_inc, 4.5 * cm_2_inc])
        rect = 0, 0, 1, 1
        axis = fig.add_axes(rect)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.imshow(cb_img[400:560, 480:640], cmap="gray")
        # plt.savefig(f"figures/cb/{test}-cb-sub2.jpg", dpi=900)
        plt.show()
    elif test == "optical-lidar-2":
        cm_2_inc = 1 / 2.54
        fig = plt.figure(figsize=[4.5 * cm_2_inc, 4.5 * cm_2_inc])
        rect = 0, 0, 1, 1
        axis = fig.add_axes(rect)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.imshow(cb_img[315:385, 210:280], cmap="gray")
        # plt.savefig(f"figures/cb/{test}-cb-sub1.jpg", dpi=900)
        plt.show()

        cm_2_inc = 1 / 2.54
        fig = plt.figure(figsize=[4.5 * cm_2_inc, 4.5 * cm_2_inc])
        rect = 0, 0, 1, 1
        axis = fig.add_axes(rect)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.imshow(cb_img[105:175, 385:455], cmap="gray")
        plt.savefig(f"figures/cb/{test}-cb-sub2.jpg", dpi=900)
        plt.show()
    else:
        cm_2_inc = 1 / 2.54
        fig = plt.figure(figsize=[4.5 * cm_2_inc, 4.5 * cm_2_inc])
        rect = 0, 0, 1, 1
        axis = fig.add_axes(rect)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.imshow(cb_img[150:250, 100:200], cmap="gray")
        plt.savefig(f"figures/cb/{test}-cb-sub1.jpg", dpi=900)
        plt.show()

        cm_2_inc = 1 / 2.54
        fig = plt.figure(figsize=[4.5 * cm_2_inc, 4.5 * cm_2_inc])
        rect = 0, 0, 1, 1
        axis = fig.add_axes(rect)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.imshow(cb_img[200:300, 250:350], cmap="gray")
        plt.savefig(f"figures/cb/{test}-cb-sub2.jpg", dpi=900)
        plt.show()
