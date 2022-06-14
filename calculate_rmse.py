from skimage.transform import estimate_transform
import numpy as np
from sklearn.metrics import mean_squared_error

test = "optical-lidar-3"

f_inliers_path = f"final_data/{test}/f_inliers.csv"
m_inliers_path = f"final_data/{test}/m_inliers.csv"
f_optim_pts_path = f"final_data/{test}/ssim_of_msolg/ns2_no6_f_optim.csv"
m_optim_pts_path = f"final_data/{test}/ssim_of_msolg/ns2_no6_m_optim.csv"
f_zncc_pts_path = f"final_data/{test}/zncc/f_optim_zncc.csv"
m_zncc_pts_path = f"final_data/{test}/zncc/m_optim_zncc.csv"
f_ssim_pts_path = f"final_data/{test}/ssim/f_optim_ssim.csv"
m_ssim_pts_path = f"final_data/{test}/ssim/m_optim_ssim.csv"
f_nmi_pts_path = f"final_data/{test}/nmi/f_optim_nmi.csv"
m_nmi_pts_path = f"final_data/{test}/nmi/m_optim_nmi.csv"
f_mind_pts_path = f"final_data/{test}/mind/f_optim_mind.csv"
m_mind_pts_path = f"final_data/{test}/mind/m_optim_mind.csv"
f_rmi_pts_path = f"final_data/{test}/rmi/f_optim_rmi.csv"
m_rmi_pts_path = f"final_data/{test}/rmi/m_optim_rmi.csv"
check_pts_path = f"final_data/{test}/refined_check_points.csv"

f_img_path = f"final_data/{test}/f_img.tif"
m_img_path = f"final_data/{test}/m_img.tif"
optim_img_path = f"final_data/{test}/ssim_of_msolg/ssim_of_msolg_optim_img.jpg"

if __name__ == "__main__":
    # region read check points
    check_file = open(check_pts_path, mode="r", encoding="utf-8-sig")
    check_lines = check_file.readlines()
    f_check_pts = np.empty(shape=(len(check_lines), 2))
    m_check_pts = np.empty(shape=(len(check_lines), 2))
    for i in range(len(check_lines)):
        info = check_lines[i].split(sep=',')
        f_check_pts[i, 0] = eval(info[0])
        f_check_pts[i, 1] = eval(info[1])
        m_check_pts[i, 0] = eval(info[2])
        m_check_pts[i, 1] = eval(info[3])
    check_file.close()
    # endregion

    # region read optimized points
    f_file = open(f_optim_pts_path, mode="r")
    m_file = open(m_optim_pts_path, mode="r")
    f_lines = f_file.readlines()
    m_lines = m_file.readlines()
    f_ours_pts = np.empty(shape=(len(f_lines), 2))
    m_ours_pts = np.empty(shape=(len(m_lines), 2))
    for i in range(len(f_lines)):
        f_info = f_lines[i].split(sep=',')
        m_info = m_lines[i].split(sep=',')
        f_ours_pts[i, 0] = eval(f_info[0])
        f_ours_pts[i, 1] = eval(f_info[1])
        m_ours_pts[i, 0] = eval(m_info[0])
        m_ours_pts[i, 1] = eval(m_info[1])
    f_file.close()
    m_file.close()

    ours_tform = estimate_transform("piecewise-affine", src=m_ours_pts, dst=f_ours_pts)
    ours_warped_pts = ours_tform(m_check_pts)
    ours_rmse = [mean_squared_error(f_check_pts[i], ours_warped_pts[i], squared=False)
                 for i in range(f_check_pts.shape[0])]
    print(f"msolg_ssim: {np.mean(ours_rmse):.3f}")
    # endregion

    # region read ssim optimized points
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
    ssim_rmse = [mean_squared_error(f_check_pts[i], ssim_warped_pts[i], squared=False)
                 for i in range(f_check_pts.shape[0])]
    print(f"ssim: {np.mean(ssim_rmse):.3f}")
    # endregion

    # region read zncc optimized points
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
    zncc_rmse = [mean_squared_error(f_check_pts[i], zncc_warped_pts[i], squared=False)
                 for i in range(f_check_pts.shape[0])]
    print(f"zncc: {np.mean(zncc_rmse):.3f}")
    # endregion

    # region read nmi optimized points
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
    nmi_rmse = [mean_squared_error(f_check_pts[i], nmi_warped_pts[i], squared=False)
                for i in range(f_check_pts.shape[0])]
    print(f"nmi: {np.mean(nmi_rmse):.3f}")
    # endregion

    # region read mind optimized points
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
    mind_rmse = [mean_squared_error(f_check_pts[i], mind_warped_pts[i], squared=False)
                 for i in range(f_check_pts.shape[0])]
    print(f"mind: {np.mean(mind_rmse):.3f}")
    # endregion

    # region read rmi optimized points
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
    rmi_rmse = [mean_squared_error(f_check_pts[i], rmi_warped_pts[i], squared=False)
                for i in range(f_check_pts.shape[0])]
    print(f"rmi: {np.mean(rmi_rmse):.3f}")
    # endregion

    # region read inliers
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

    pl_tform = estimate_transform("piecewise-affine", src=m_inliers, dst=f_inliers)
    pl_warped_pts = pl_tform(m_check_pts)
    pl_rmse = [mean_squared_error(f_check_pts[i], pl_warped_pts[i], squared=False)
               for i in range(f_check_pts.shape[0])]
    print(f"pl: {np.mean(pl_rmse):.3f}")


