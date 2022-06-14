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
mpl.rcParams["mathtext.fontset"] = "stix"

if __name__ == "__main__":
    optim_rmse = []
    zncc_rmse = []
    nmi_rmse = []
    ssim_rmse = []
    rmi_rmse = []
    mind_rmse = []

    for test in ["optical-sar-1", "optical-sar-2", "optical-sar-3",
                 "optical-lidar-1", "optical-lidar-2", "optical-lidar-3",
                 "optical-infrared-1", "optical-infrared-2"]:
        f_optim_path = f"final_data/{test}/ssim_of_msolg/ns2_no6_f_optim.csv"
        m_optim_path = f"final_data/{test}/ssim_of_msolg/ns2_no6_m_optim.csv"
        f_zncc_pts_path = f"final_data/{test}/zncc/f_optim_zncc.csv"
        m_zncc_pts_path = f"final_data/{test}/zncc/m_optim_zncc.csv"
        f_ssim_pts_path = f"final_data/{test}/ssim/f_optim_ssim.csv"
        m_ssim_pts_path = f"final_data/{test}/ssim/m_optim_ssim.csv"
        f_nmi_pts_path = f"final_data/{test}/nmi/f_optim_nmi.csv"
        m_nmi_pts_path = f"final_data/{test}/nmi/m_optim_nmi.csv"
        f_rmi_pts_path = f"final_data/{test}/rmi/f_optim_rmi.csv"
        m_rmi_pts_path = f"final_data/{test}/rmi/m_optim_rmi.csv"
        f_mind_pts_path = f"final_data/{test}/mind/f_optim_mind.csv"
        m_mind_pts_path = f"final_data/{test}/mind/m_optim_mind.csv"
        check_pts_path = f"final_data/{test}/check_points.csv"

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

        f_file = open(f_zncc_pts_path, mode="r")
        m_file = open(m_zncc_pts_path, mode="r")
        f_lines = f_file.readlines()
        m_lines = m_file.readlines()
        f_zncc_pts = np.empty(shape=(len(f_lines), 2))
        m_zncc_pts = np.empty(shape=(len(m_lines), 2))
        for i in range(len(f_lines)):
            f_info = f_lines[i].split(sep=',')
            m_info = m_lines[i].split(sep=',')
            f_zncc_pts[i, 0] = eval(f_info[0])
            f_zncc_pts[i, 1] = eval(f_info[1])
            m_zncc_pts[i, 0] = eval(m_info[0])
            m_zncc_pts[i, 1] = eval(m_info[1])
        f_file.close()
        m_file.close()
        zncc_tform = estimate_transform("piecewise-affine", src=m_zncc_pts, dst=f_zncc_pts)
        zncc_warped_pts = zncc_tform(m_check_pts)
        zncc_residuals = [mean_squared_error(f_check_pts[i], zncc_warped_pts[i], squared=False)
                          for i in range(f_check_pts.shape[0])]
        zncc_rmse.append(np.mean(zncc_residuals))

        f_file = open(f_nmi_pts_path, mode="r")
        m_file = open(m_nmi_pts_path, mode="r")
        f_lines = f_file.readlines()
        m_lines = m_file.readlines()
        f_nmi_pts = np.empty(shape=(len(f_lines), 2))
        m_nmi_pts = np.empty(shape=(len(m_lines), 2))
        for i in range(len(f_lines)):
            f_info = f_lines[i].split(sep=',')
            m_info = m_lines[i].split(sep=',')
            f_nmi_pts[i, 0] = eval(f_info[0])
            f_nmi_pts[i, 1] = eval(f_info[1])
            m_nmi_pts[i, 0] = eval(m_info[0])
            m_nmi_pts[i, 1] = eval(m_info[1])
        f_file.close()
        m_file.close()
        nmi_tform = estimate_transform("piecewise-affine", src=m_nmi_pts, dst=f_nmi_pts)
        nmi_warped_pts = nmi_tform(m_check_pts)
        nmi_residuals = [mean_squared_error(f_check_pts[i], nmi_warped_pts[i], squared=False)
                         for i in range(f_check_pts.shape[0])]
        nmi_rmse.append(np.mean(nmi_residuals))

        f_file = open(f_ssim_pts_path, mode="r")
        m_file = open(m_ssim_pts_path, mode="r")
        f_lines = f_file.readlines()
        m_lines = m_file.readlines()
        f_ssim_pts = np.empty(shape=(len(f_lines), 2))
        m_ssim_pts = np.empty(shape=(len(m_lines), 2))
        for i in range(len(f_lines)):
            f_info = f_lines[i].split(sep=',')
            m_info = m_lines[i].split(sep=',')
            f_ssim_pts[i, 0] = eval(f_info[0])
            f_ssim_pts[i, 1] = eval(f_info[1])
            m_ssim_pts[i, 0] = eval(m_info[0])
            m_ssim_pts[i, 1] = eval(m_info[1])
        f_file.close()
        m_file.close()
        ssim_tform = estimate_transform("piecewise-affine", src=m_ssim_pts, dst=f_ssim_pts)
        ssim_warped_pts = ssim_tform(m_check_pts)
        ssim_residuals = [mean_squared_error(f_check_pts[i], ssim_warped_pts[i], squared=False)
                          for i in range(f_check_pts.shape[0])]
        ssim_rmse.append(np.mean(ssim_residuals))

        f_file = open(f_rmi_pts_path, mode="r")
        m_file = open(m_rmi_pts_path, mode="r")
        f_lines = f_file.readlines()
        m_lines = m_file.readlines()
        f_rmi_pts = np.empty(shape=(len(f_lines), 2))
        m_rmi_pts = np.empty(shape=(len(m_lines), 2))
        for i in range(len(f_lines)):
            f_info = f_lines[i].split(sep=',')
            m_info = m_lines[i].split(sep=',')
            f_rmi_pts[i, 0] = eval(f_info[0])
            f_rmi_pts[i, 1] = eval(f_info[1])
            m_rmi_pts[i, 0] = eval(m_info[0])
            m_rmi_pts[i, 1] = eval(m_info[1])
        f_file.close()
        m_file.close()
        rmi_tform = estimate_transform("piecewise-affine", src=m_rmi_pts, dst=f_rmi_pts)
        rmi_warped_pts = rmi_tform(m_check_pts)
        rmi_residuals = [mean_squared_error(f_check_pts[i], rmi_warped_pts[i], squared=False)
                         for i in range(f_check_pts.shape[0])]
        rmi_rmse.append(np.mean(rmi_residuals))

        f_file = open(f_mind_pts_path, mode="r")
        m_file = open(m_mind_pts_path, mode="r")
        f_lines = f_file.readlines()
        m_lines = m_file.readlines()
        f_mind_pts = np.empty(shape=(len(f_lines), 2))
        m_mind_pts = np.empty(shape=(len(m_lines), 2))
        for i in range(len(f_lines)):
            f_info = f_lines[i].split(sep=',')
            m_info = m_lines[i].split(sep=',')
            f_mind_pts[i, 0] = eval(f_info[0])
            f_mind_pts[i, 1] = eval(f_info[1])
            m_mind_pts[i, 0] = eval(m_info[0])
            m_mind_pts[i, 1] = eval(m_info[1])
        f_file.close()
        m_file.close()
        mind_tform = estimate_transform("piecewise-affine", src=m_mind_pts, dst=f_mind_pts)
        mind_warped_pts = mind_tform(m_check_pts)
        mind_residuals = [mean_squared_error(f_check_pts[i], mind_warped_pts[i], squared=False)
                          for i in range(f_check_pts.shape[0])]
        mind_rmse.append(np.mean(mind_residuals))

    print(optim_rmse)
    print(zncc_rmse)
    print(nmi_rmse)
    print(ssim_rmse)
    print(rmi_rmse)
    print(mind_rmse)

    fig, axis = plt.subplots(figsize=[6, 4])
    axis.set_yticks([0, 0.5, 1, 1.5])
    axis.set_yticklabels([0.0, 0.5, 1.0, 1.5])
    axis.set_xticks(range(1, 9, 1))
    axis.set_xticklabels(range(1, 9, 1))

    axis.set_ylabel("RMSE", weight="bold")
    axis.set_xlabel("Test No.", weight="bold")

    width = 0.125
    gap = 0.015

    optim = axis.bar(1 - (width + gap) * 5 / 2, optim_rmse[0], width=width, color="#20bf6b")
    zncc = axis.bar(1 - (width+gap) * 3 / 2, zncc_rmse[0], width=width, color='#fff200')
    nmi = axis.bar(1 - (width+gap) / 2, nmi_rmse[0], width=width, color="#fa8231")
    ssim = axis.bar(1 + (width+gap) / 2, ssim_rmse[0], width=width, color="#eb3b5a")
    rmi = axis.bar(1 + (width+gap) * 3 / 2, rmi_rmse[0], width=width, color="#3867d6")
    mind = axis.bar(1 + (width+gap) * 5 / 2, mind_rmse[0], width=width, color="#8854d0")

    optim = axis.bar(2 - (width+gap) * 5 / 2, optim_rmse[1], width=width, color="#20bf6b")
    zncc = axis.bar(2 - (width+gap) * 3 / 2, zncc_rmse[1], width=width, color='#fff200')
    nmi = axis.bar(2 - (width+gap) / 2, nmi_rmse[1], width=width, color="#fa8231")
    ssim = axis.bar(2 + (width+gap) / 2, ssim_rmse[1], width=width, color="#eb3b5a")
    rmi = axis.bar(2 + (width+gap) * 3 / 2, rmi_rmse[1], width=width, color="#3867d6")
    mind = axis.bar(2 + (width+gap) * 5 / 2, mind_rmse[1], width=width, color="#8854d0")

    optim = axis.bar(3 - (width+gap) * 5 / 2, optim_rmse[2], width=width, color="#20bf6b")
    zncc = axis.bar(3 - (width+gap) * 3 / 2, zncc_rmse[2], width=width, color='#fff200')
    nmi = axis.bar(3 - (width+gap) / 2, nmi_rmse[2], width=width, color="#fa8231")
    ssim = axis.bar(3 + (width+gap) / 2, ssim_rmse[2], width=width, color="#eb3b5a")
    rmi = axis.bar(3 + (width+gap) * 3 / 2, rmi_rmse[2], width=width, color="#3867d6")
    mind = axis.bar(3 + (width+gap) * 5 / 2, mind_rmse[2], width=width, color="#8854d0")

    optim = axis.bar(4 - (width+gap) * 5 / 2, optim_rmse[3], width=width, color="#20bf6b")
    zncc = axis.bar(4 - (width+gap) * 3 / 2, zncc_rmse[3], width=width, color='#fff200')
    nmi = axis.bar(4 - (width+gap) / 2, nmi_rmse[3], width=width, color="#fa8231")
    ssim = axis.bar(4 + (width+gap) / 2, ssim_rmse[3], width=width, color="#eb3b5a")
    rmi = axis.bar(4 + (width+gap) * 3 / 2, rmi_rmse[3], width=width, color="#3867d6")
    mind = axis.bar(4 + (width+gap) * 5 / 2, mind_rmse[3], width=width, color="#8854d0")

    optim = axis.bar(5 - (width+gap) * 5 / 2, optim_rmse[4], width=width, color="#20bf6b")
    zncc = axis.bar(5 - (width+gap) * 3 / 2, zncc_rmse[4], width=width, color='#fff200')
    nmi = axis.bar(5 - (width+gap) / 2, nmi_rmse[4], width=width, color="#fa8231")
    ssim = axis.bar(5 + (width+gap) / 2, ssim_rmse[4], width=width, color="#eb3b5a")
    rmi = axis.bar(5 + (width+gap) * 3 / 2, rmi_rmse[4], width=width, color="#3867d6")
    mind = axis.bar(5 + (width+gap) * 5 / 2, mind_rmse[4], width=width, color="#8854d0")

    optim = axis.bar(6 - (width+gap) * 5 / 2, optim_rmse[5], width=width, color="#20bf6b")
    zncc = axis.bar(6 - (width+gap) * 3 / 2, zncc_rmse[5], width=width, color='#fff200')
    nmi = axis.bar(6 - (width+gap) / 2, nmi_rmse[5], width=width, color="#fa8231")
    ssim = axis.bar(6 + (width+gap) / 2, ssim_rmse[5], width=width, color="#eb3b5a")
    rmi = axis.bar(6 + (width+gap) * 3 / 2, rmi_rmse[5], width=width, color="#3867d6")
    mind = axis.bar(6 + (width+gap) * 5 / 2, mind_rmse[5], width=width, color="#8854d0")

    optim = axis.bar(7 - (width+gap) * 5 / 2, optim_rmse[6], width=width, color="#20bf6b")
    zncc = axis.bar(7 - (width+gap) * 3 / 2, zncc_rmse[6], width=width, color='#fff200')
    nmi = axis.bar(7 - (width+gap) / 2, nmi_rmse[6], width=width, color="#fa8231")
    ssim = axis.bar(7 + (width+gap) / 2, ssim_rmse[6], width=width, color="#eb3b5a")
    rmi = axis.bar(7 + (width+gap) * 3 / 2, rmi_rmse[6], width=width, color="#3867d6")
    mind = axis.bar(7 + (width+gap) * 5 / 2, mind_rmse[6], width=width, color="#8854d0")

    optim = axis.bar(8 - (width+gap) * 5 / 2, optim_rmse[7], width=width, color="#20bf6b",
                     label="$\mathregular{MSOLG_{SSIM}}$")
    zncc = axis.bar(8 - (width+gap) * 3 / 2, zncc_rmse[7], width=width, color='#fff200', label="ZNCC")
    nmi = axis.bar(8 - (width+gap) / 2, nmi_rmse[7], width=width, color="#fa8231", label="NMI")
    ssim = axis.bar(8 + (width+gap) / 2, ssim_rmse[7], width=width, color="#eb3b5a", label="SSIM")
    rmi = axis.bar(8 + (width+gap) * 3 / 2, rmi_rmse[7], width=width, color="#3867d6", label="RMI")
    mind = axis.bar(8 + (width+gap) * 5 / 2, mind_rmse[7], width=width, color="#8854d0", label="MIND")

    legend = axis.legend(loc="best", fontsize=12, ncol=2)
    legend.get_frame().set_edgecolor('black')
    plt.tight_layout()

    # plt.savefig(f"figures/compare/compare_fontsize_{fontsize}.tiff", dpi=900, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    plt.show()
