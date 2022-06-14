from skimage.transform import estimate_transform
import numpy as np
from sklearn.metrics import mean_squared_error
from g_im_io import read_image, normalize_image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_rmse = np.zeros(shape=(6, 5), dtype=np.float32)

    for test in ["optical-infrared-1", "optical-lidar-1", "optical-sar-1"]:
        print(test)
        rmse = np.zeros(shape=(6, 5), dtype=np.float32)
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
        for ns in range(1, 7):
            for no in range(4, 9):
                f_optim_path = f"parameter_study_all/{test}/ns{ns}-no{no}/ns{ns}_no{no}_f_optim.csv"
                m_optim_path = f"parameter_study_all/{test}/ns{ns}-no{no}/ns{ns}_no{no}_m_optim.csv"
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
                optim_rmse = np.mean(optim_residuals)
                print(f"{test}, ns: {ns}, no: {no}, rmse: {optim_rmse}")

                rmse[ns - 1, no - 4] += optim_rmse
        all_rmse += rmse

        print(f"row: scale, 1-6, col: orient, 4-8, \n{rmse}")

    print("all")
    print(f"row: scale, 1-6, col: orient, 4-8, \n{all_rmse/3}")

