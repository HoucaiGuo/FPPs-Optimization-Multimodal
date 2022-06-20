from g_im_io import *
from g_im_display import *
from skimage.color import rgb2gray
from scipy.spatial import Delaunay

from optimization_of_feature_points import PointsOptimization

# test = "optical-sar-1"
# criterion = "mind"
#
# f_img_path = f"final_data/{test}/f_img.tif"
# m_img_path = f"final_data/{test}/m_img.tif"
# f_inliers_path = f"final_data/{test}/f_inliers.csv"
# m_inliers_path = f"final_data/{test}/m_inliers.csv"
# f_optim_path = f"final_data/{test}/{criterion}/f_optim_{criterion}.csv"
# m_optim_path = f"final_data/{test}/{criterion}/m_optim_{criterion}.csv"
# optim_img_path = f"final_data/{test}/{criterion}/{criterion}_optim_img.jpg"
# losses_path = f"final_data/{test}/{criterion}/losses.csv"

line_color = "#f9ca24"
f_pt_color = "#d63031"
m_pt_color = "#0984e3"

if __name__ == "__main__":
    for test in ["optical-infrared-1", "optical-infrared-2",
                 "optical-lidar-1", "optical-lidar-2", "optical-lidar-3",
                 "optical-sar-1", "optical-sar-2", "optical-sar-3"]:
        criterion = "ssim_of_msolg"
        f_img_path = f"final_data/{test}/f_img.tif"
        m_img_path = f"final_data/{test}/m_img.tif"
        f_inliers_path = f"final_data/{test}/f_inliers.csv"
        m_inliers_path = f"final_data/{test}/m_inliers.csv"
        f_optim_path = f"final_data/{test}/{criterion}/f_optim_{criterion}.csv"
        m_optim_path = f"final_data/{test}/{criterion}/m_optim_{criterion}.csv"
        optim_img_path = f"final_data/{test}/{criterion}/{criterion}_optim_img.jpg"
        losses_path = f"final_data/{test}/{criterion}/losses.csv"

        # region read, normalize, stretch and color composite image
        f_img, prof = read_image(f_img_path)
        print(prof)
        f_img = f_img.astype(np.float32)
        m_img = read_image(m_img_path)[0].astype(np.float32)
        f_norm = normalize_image(f_img)
        m_norm = normalize_image(m_img)

        if f_img.shape[2] == 1:
            f_norm = np.concatenate([f_norm, f_norm, f_norm], axis=2)
        if m_img.shape[2] == 1:
            m_norm = np.concatenate([m_norm, m_norm, m_norm], axis=2)

        f_norm = linear_pct_stretch(f_norm, 2)
        m_norm = linear_pct_stretch(m_norm, 2)

        f_rgb = color_composite(f_norm, [2, 1, 0])
        m_rgb = color_composite(m_norm, [2, 1, 0])

        f_gray = rgb2gray(f_rgb)
        m_gray = rgb2gray(m_rgb)
        # endregion

        # region read points
        f_file = open(f_inliers_path, mode="r")
        m_file = open(m_inliers_path, mode="r")
        f_lines = f_file.readlines()
        m_lines = m_file.readlines()
        f_inliers = np.empty(shape=(len(f_lines), 2))
        m_inliers = np.empty(shape=(len(m_lines), 2))
        for i in range(len(f_lines)):
            f_info = f_lines[i].split(sep=',')
            m_info = m_lines[i].split(sep=',')
            f_inliers[i, 0] = eval(f_info[0])
            f_inliers[i, 1] = eval(f_info[1])
            m_inliers[i, 0] = eval(m_info[0])
            m_inliers[i, 1] = eval(m_info[1])
        f_file.close()
        m_file.close()
        # endregion

        # region points optimization, image registration
        plt_optim = PointsOptimization(f_img, m_img, f_inliers, m_inliers, criterion=criterion, nscale=2, norient=6)
        plt_optim.optimize()
        f_optim = plt_optim.f_pts
        m_optim = plt_optim.m_pts
        print(f"Finished the optimization of FPPs, {f_optim.shape[0]} remain.")

        print(np.sum(plt_optim.del_acc_diffs))

        # region show optimal points and TIN
        # tin = Delaunay(f_optim)
        # fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        # axes[0].imshow(f_gray, cmap="gray")
        # axes[1].imshow(m_gray, cmap="gray")
        # axes[0].set_title("fixed image")
        # axes[1].set_title("moving image")
        # axes[0].triplot(f_optim[:, 1], f_optim[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        # axes[1].triplot(m_optim[:, 1], m_optim[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        # for i in range(f_optim.shape[0]):
        #     axes[0].plot(f_optim[i, 1], f_optim[i, 0], color=f_pt_color, marker=".", markersize=6)
        #     axes[1].plot(m_optim[i, 1], m_optim[i, 0], color=m_pt_color, marker=".", markersize=6)
        # plt.show()
        # endregion

        # region save optimized points and perform optimized PLT
        f_file = open(f_optim_path, mode="x")
        m_file = open(m_optim_path, mode="x")
        for i in range(f_optim.shape[0]):
            f_file.write(f"{f_optim[i, 0]},{f_optim[i, 1]}\n")
            m_file.write(f"{m_optim[i, 0]},{m_optim[i, 1]}\n")
        f_file.close()
        m_file.close()

        optim_img = plt_optim.transform()
        save_image(optim_img, optim_img_path, prof)

        # from skimage.io import imsave
        #
        # imsave(optim_img_path, optim_img)
        # endregion

        file = open(losses_path, mode="x")
        for i in range(len(plt_optim.del_losses)):
            file.write(f"{plt_optim.del_losses[i]}\n")
        file.close()


