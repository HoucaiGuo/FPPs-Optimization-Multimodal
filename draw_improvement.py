from skimage.transform import estimate_transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import matplotlib as mpl

fontsize = 15

mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rc('font', family='Times New Roman')

if __name__ == "__main__":
    pl_rmse = []
    optim_rmse = []

    for test in ["optical-sar-1", "optical-sar-2", "optical-sar-3",
                 "optical-lidar-1", "optical-lidar-2", "optical-lidar-3",
                 "optical-infrared-1", "optical-infrared-2"]:
        check_pts_path = f"final_data/{test}/check_points.csv"
        f_inliers_path = f"final_data/{test}/f_inliers.csv"
        m_inliers_path = f"final_data/{test}/m_inliers.csv"
        f_optim_path = f"final_data/{test}/ssim_of_msolg/ns2_no6_f_optim.csv"
        m_optim_path = f"final_data/{test}/ssim_of_msolg/ns2_no6_m_optim.csv"

        check_file = open(check_pts_path, mode="r", encoding="utf-8-sig")
        check_lines = check_file.readlines()
        f_check_pts = np.empty(shape=(len(check_lines), 2))
        m_check_pts = np.empty(shape=(len(check_lines), 2))
        for i in range(len(check_lines)):
            info = check_lines[i].split(sep=',')
            f_check_pts[i, 0] = eval(info[0])-1
            f_check_pts[i, 1] = eval(info[1])-1
            m_check_pts[i, 0] = eval(info[2])-1
            m_check_pts[i, 1] = eval(info[3])-1
        check_file.close()

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
        inliers_tform = estimate_transform("piecewise-affine", src=m_inliers, dst=f_inliers)
        inliers_warped_pts = inliers_tform(m_check_pts)
        inliers_residuals = [mean_squared_error(f_check_pts[i], inliers_warped_pts[i], squared=False)
                             for i in range(f_check_pts.shape[0])]
        pl_rmse.append(np.mean(inliers_residuals))

        f_file = open(f_optim_path, mode="r")
        m_file = open(m_optim_path, mode="r")
        f_lines = f_file.readlines()
        m_lines = m_file.readlines()
        f_optim_pts = np.empty(shape=(len(f_lines), 2))
        m_optim_pts = np.empty(shape=(len(m_lines), 2))
        for i in range(len(f_lines)):
            f_info = f_lines[i].split(sep=',')
            m_info = m_lines[i].split(sep=',')
            f_optim_pts[i, 0] = eval(f_info[0])
            f_optim_pts[i, 1] = eval(f_info[1])
            m_optim_pts[i, 0] = eval(m_info[0])
            m_optim_pts[i, 1] = eval(m_info[1])
        f_file.close()
        m_file.close()
        optim_tform = estimate_transform("piecewise-affine", src=m_optim_pts, dst=f_optim_pts)
        optim_warped_pts = optim_tform(m_check_pts)
        optim_residuals = [mean_squared_error(f_check_pts[i], optim_warped_pts[i], squared=False)
                           for i in range(f_check_pts.shape[0])]
        optim_rmse.append(np.mean(optim_residuals))

        print(f"{test}, feature points: {f_inliers.shape}, check points: {f_check_pts.shape}")

    print(pl_rmse)
    print(optim_rmse)

    fig, axis = plt.subplots(figsize=[6, 4])
    axis.set_yticks([0, 0.5, 1, 1.5])
    axis.set_yticklabels([0.0, 0.5, 1.0, 1.5])
    axis.set_xticks(range(1, 9, 1))
    axis.set_xticklabels(range(1, 9, 1))

    axis.set_ylabel("RMSE", weight="bold")
    axis.set_xlabel("Test No.", weight="bold")

    width = 0.4
    gap = 0.03

    pl_1 = axis.bar(1 - (width+gap) / 2, pl_rmse[0], width=width, color="#eb3b5a")
    optim_1 = axis.bar(1 + (width+gap) / 2, optim_rmse[0], width=width, color="#20bf6b")
    axis.bar_label(pl_1, fmt="%.2f")
    axis.bar_label(optim_1, fmt="%.2f")

    pl_2 = axis.bar(2 - (width+gap) / 2, pl_rmse[1], width=width, color="#eb3b5a")
    optim_2 = axis.bar(2 + (width+gap) / 2, optim_rmse[1], width=width, color="#20bf6b")
    axis.bar_label(pl_2, fmt="%.2f")
    axis.bar_label(optim_2, fmt="%.2f")

    pl_3 = axis.bar(3 - (width+gap) / 2, pl_rmse[2], width=width, color="#eb3b5a")
    optim_3 = axis.bar(3 + (width+gap) / 2, optim_rmse[2], width=width, color="#20bf6b")
    axis.bar_label(pl_3, fmt="%.2f")
    axis.bar_label(optim_3, fmt="%.2f")

    pl_4 = axis.bar(4 - (width+gap) / 2, pl_rmse[3], width=width, color="#eb3b5a")
    optim_4 = axis.bar(4 + (width+gap) / 2, optim_rmse[3], width=width, color="#20bf6b")
    axis.bar_label(pl_4, fmt="%.2f")
    axis.bar_label(optim_4, fmt="%.2f")

    pl_5 = axis.bar(5 - (width+gap) / 2, pl_rmse[4], width=width, color="#eb3b5a")
    optim_5 = axis.bar(5 + (width+gap) / 2, optim_rmse[4], width=width, color="#20bf6b")
    axis.bar_label(pl_5, fmt="%.2f")
    axis.bar_label(optim_5, fmt="%.2f")

    pl_6 = axis.bar(6 - (width+gap) / 2, pl_rmse[5], width=width, color="#eb3b5a")
    optim_6 = axis.bar(6 + (width+gap) / 2, optim_rmse[5], width=width, color="#20bf6b")
    axis.bar_label(pl_6, fmt="%.2f")
    axis.bar_label(optim_6, fmt="%.2f")

    pl_7 = axis.bar(7 - (width+gap) / 2, pl_rmse[6], width=width, color="#eb3b5a")
    optim_7 = axis.bar(7 + (width+gap) / 2, optim_rmse[6], width=width, color="#20bf6b")
    axis.bar_label(pl_7, fmt="%.2f")
    axis.bar_label(optim_7, fmt="%.2f")

    pl_8 = axis.bar(8 - (width+gap) / 2, pl_rmse[7], width=width, color="#eb3b5a", label="PL")
    # axis.bar_label(pl_8, fmt="%.2f")
    optim_8 = axis.bar(8 + (width+gap) / 2, optim_rmse[7], width=width, color="#20bf6b", label="Optimized PL")
    # axis.bar_label(optim_8, fmt="%.2f")
    axis.bar_label(pl_8, fmt="%.2f")
    axis.bar_label(optim_8, fmt="%.2f")

    legend = axis.legend(loc="best", fontsize=fontsize)
    legend.get_frame().set_edgecolor('black')
    plt.tight_layout()

    # plt.savefig(f"figures/improvement/improvement_fontsize_{fontsize}.tiff", dpi=900, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    plt.show()
