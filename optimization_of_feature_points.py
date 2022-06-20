import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from skimage.draw import polygon2mask
from skimage.transform import estimate_transform, warp
from functions import delete_point, delete_attribute, similarity_measure
from datetime import datetime
import shapely

np.set_printoptions(suppress=True)


class PointsOptimization:
    """
    Optimize the feature point pairs (FPPs) used in piecewise linear (PL) transformation
    by deleting one FPPs with negative influence on its local registration accuracy, iteratively.

    The method consists of two main procedures:
        1. Calculate the similarity loss of each FPP by applying a specific similarity measurement. The
        loss is defined as "S_del" minus "S_use", where "S_use" represents the similarity between
        the local patch in the fixed image and its correspondence in the registered image when
        using this FPP to estimate a PL transformation, and "S_del" is the similarity when discarding
        this FPP to estimate a PL transformation.

        2. Delete FPPs that have the maximum loss iteratively. For a positive loss value indicates this
        FPP has a negative influence on the registration accuracy of the local image patch, so all of the
        retained FPPs would have a positive impact on the registration accuracy.

    Parameters
    ----------
    f_img : numpy.array
        The fixed image, (Height, Width, Channel) shaped.
    m_img : numpy.array
        The moving image.
    f_pts : numpy.array
        FPPs detected from the fixed image, (n, 2) shaped.
    m_pts : numpy.array
        FPPs detected from the moving image.
    criterion : str, optional
        Similarity measure criterion to use. Default is "ssim_of_msolg".
        "zncc" : Zero mean normalized cross-correlation.
        "mi" : Mutual information.
        "nmi" : Normalized mutual information.
        "ssim" : Structural similarity.
        "ssim_of_msolg" : Structural Similarity of MSOLG features.
        "mind" : Modality independent neighborhood descriptor.
        "rmi" : Regional mutual information.
    order : int, optional
        The order of interpolation, default is 3. The order has to be in the range 0-5:
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic

    Attributes
    ----------
    f_img, m_img, f_pts, m_pts, criterion, order: same as Parameters.
    tin : scipy.Delaunay
        The triangulation of f_pts.
    convex_hull : scipy.ConvexHull
        The convex hull of f_pts.
    losses : numpy.array
        An array that records loss of each point.
    del_losses : list
        A list that records the maximum loss value in each iteration.
    del_f_pts : list
        List of deleted f_pt.
    del_m_pts : list
        List of deleted m_pt.
    """

    def __init__(self, f_img, m_img, f_pts, m_pts, criterion="ssim_of_msolg", order=3):
        self.f_img = f_img.astype(np.float32)
        self.m_img = m_img.astype(np.float32)
        self.f_pts = f_pts
        self.m_pts = m_pts
        self.criterion = criterion
        self.order = order
        self.tin = Delaunay(self.f_pts)
        self.convex_hull = ConvexHull(self.f_pts)
        self.losses = np.empty(shape=(self.f_pts.shape[0],), dtype=np.float64)
        self.del_losses = []
        self.del_f_pts = []
        self.del_m_pts = []

    def __influence_area_triangles(self, pt_idx):
        """
        Find triangles that form the "influence area" of a given point in the TIN. For a single point, the influence
        area is the polygon consists of its adjacent points.

        Parameters
        ----------
        pt_idx : int
            Index of the point.

        Returns
        -------
        triangles : array_like
            Triangles that make up the influence area, (n, 3) shaped, each record along the first axis contains 3
                indices of vertices that compose a triangle.
        """
        # use the nesting of np.any() and np.isin() to judge if the point is a vertex of a triangle(tin.simplex)
        triangles = self.tin.simplices[np.any(np.isin(self.tin.simplices, pt_idx), axis=1)]

        return triangles

    def __influence_area(self, pt_idx, nei_ia=True):
        """
        Extract fixed and moving image patches inside the influence area of a given point.

        Parameters
        ----------
        pt_idx : int
            Index of the point.
        nei_ia : bool, optional
            Add the neighbors points' influence area into this point's influence area or not.

        Returns
        -------
        f_ia_img : array_like
            The fixed image inside the influence area.
        m_ia_img : array_like
            The moving image inside the influence area.
        """
        ia_tris = self.__influence_area_triangles(pt_idx)

        if nei_ia:
            # indices of the adjacent points
            nbr_pts_idx = np.setdiff1d(np.unique(ia_tris), pt_idx)

            # its neighbors' influence area triangles
            for nbr_idx in nbr_pts_idx:
                ia_tris = np.concatenate([ia_tris, self.__influence_area_triangles(nbr_idx)])
            # every triangle in ia_tris is unique
            ia_tris = np.unique(ia_tris, axis=0)

        f_ia_mask = polygon2mask(self.f_img.shape,
                                 self.f_pts[ia_tris[0]])
        m_ia_mask = polygon2mask(self.m_img.shape,
                                 self.m_pts[ia_tris[0]])

        for i in range(1, ia_tris.shape[0]):
            f_tri_mask = polygon2mask(self.f_img.shape,
                                      self.f_pts[ia_tris[i]])
            f_ia_mask = np.bitwise_or(f_ia_mask, f_tri_mask)

            m_tri_mask = polygon2mask(self.m_img.shape,
                                      self.m_pts[ia_tris[i]])
            m_ia_mask = np.bitwise_or(m_ia_mask, m_tri_mask)

        # idx is the index of point inside the f_ia_pts
        pts_indices = np.unique(ia_tris)
        idx = np.where(pts_indices == pt_idx)[0][0]

        # coordinates of the triangles' vertices, (p, 2)shaped
        f_ia_pts = self.f_pts[pts_indices]
        m_ia_pts = self.m_pts[pts_indices]

        # extract sub-image by envelope of the influence area
        f_row_min = np.min(f_ia_pts[:, 0]).astype(np.int32)
        f_row_max = np.max(f_ia_pts[:, 0]).astype(np.int32)
        f_col_min = np.min(f_ia_pts[:, 1]).astype(np.int32)
        f_col_max = np.max(f_ia_pts[:, 1]).astype(np.int32)
        f_ia_img = self.f_img[f_row_min:f_row_max + 1, f_col_min:f_col_max + 1, :].copy()
        f_ia_mask = f_ia_mask[f_row_min:f_row_max + 1, f_col_min:f_col_max + 1, :]
        m_row_min = np.min(m_ia_pts[:, 0]).astype(np.int32)
        m_row_max = np.max(m_ia_pts[:, 0]).astype(np.int32)
        m_col_min = np.min(m_ia_pts[:, 1]).astype(np.int32)
        m_col_max = np.max(m_ia_pts[:, 1]).astype(np.int32)
        m_ia_img = self.m_img[m_row_min:m_row_max + 1, m_col_min:m_col_max + 1, :].copy()
        m_ia_mask = m_ia_mask[m_row_min:m_row_max + 1, m_col_min:m_col_max + 1, :]

        # transfer points' coordinate from image to sub-image, (m, 3, 2) shaped
        f_ia_pts = f_ia_pts - np.array([f_row_min, f_col_min])
        m_ia_pts = m_ia_pts - np.array([m_row_min, m_col_min])

        return f_ia_img, m_ia_img, f_ia_mask, m_ia_mask, f_ia_pts, m_ia_pts, idx

    def __single_point_test(self, pt_idx, nei_ia=True):
        """
        Calculate the loss for a single FPP.
        """
        # test if its point is on the boundary
        is_bdy = np.any(np.in1d(self.convex_hull.vertices, pt_idx))
        if is_bdy:
            loss = np.NINF
        else:
            # calculate this point's acc_diff
            f_ia_img, m_ia_img, f_ia_mask, m_ia_mask, f_ia_pts, m_ia_pts, idx = self.__influence_area(pt_idx, nei_ia)
            f_ia_img = np.ma.array(data=f_ia_img, mask=~f_ia_mask)
            f_ia_img.data[f_ia_img.mask] = np.nan

            tform_use = estimate_transform("piecewise-affine",
                                           src=np.flip(f_ia_pts, axis=1),
                                           dst=np.flip(m_ia_pts, axis=1))
            reg_img_use = warp(m_ia_img, tform_use, output_shape=f_ia_img.shape, order=self.order)
            reg_img_use = np.ma.array(data=reg_img_use, mask=~f_ia_mask)
            reg_img_use.data[reg_img_use.mask] = np.nan
            s_use = similarity_measure(f_ia_img, reg_img_use, criterion=self.criterion, mode="image")

            f_ia_pts_del = delete_point(f_ia_pts, idx)
            m_ia_pts_del = delete_point(m_ia_pts, idx)
            tform_del = estimate_transform("piecewise-affine",
                                           src=np.flip(f_ia_pts_del, axis=1),
                                           dst=np.flip(m_ia_pts_del, axis=1))
            reg_img_del = warp(m_ia_img, tform_del, output_shape=f_ia_img.shape, order=self.order)
            reg_img_del = np.ma.array(data=reg_img_del, mask=~f_ia_mask)
            reg_img_del.data[reg_img_del.mask] = np.nan

            s_del = similarity_measure(f_ia_img, reg_img_del, criterion=self.criterion, mode="image")

            loss = s_del - s_use

        return loss

    def __single_point_optimize(self, pt_idx):
        """
        Delete a single FPP, then update its neighbors' loss.
        """
        print(f"{pt_idx} should be deleted, "
              f"f_pt: {self.f_pts[pt_idx]}, m_pt: {self.m_pts[pt_idx]}, "
              f"loss: {self.acc_diffs[pt_idx]}")

        indptr, indices = self.tin.vertex_neighbor_vertices
        nei_indices = indices[indptr[pt_idx]:indptr[pt_idx + 1]]
        print(f"\tIt's neighbors: {nei_indices}")

        self.f_pts = delete_point(self.f_pts, pt_idx)
        self.m_pts = delete_point(self.m_pts, pt_idx)
        self.acc_diffs = delete_attribute(self.acc_diffs, pt_idx)
        self.tin = Delaunay(self.f_pts)
        self.convex_hull = ConvexHull(self.f_pts)

        for i in range(nei_indices.shape[0]):
            if nei_indices[i] > pt_idx:
                nei_indices[i] -= 1

        for nei_idx in nei_indices:
            original_acc_diff = self.acc_diffs[nei_idx]
            acc_diff = self.__single_point_test(nei_idx)
            print(f"\t\tidx: {nei_idx}, original loss: {original_acc_diff}, new loss: {acc_diff}")
            self.acc_diffs[nei_idx] = acc_diff

    def optimize(self):
        """
        Optimize the FPPs.
        """
        print("Start calculating the FPPs' loss!")
        time0 = datetime.now()
        for pt_idx in range(self.f_pts.shape[0]):
            self.losses[pt_idx] = self.__single_point_test(pt_idx)
            print(f"idx: {pt_idx}, f_pt: {self.f_pts[pt_idx]}, m_pt: {self.m_pts[pt_idx]}, "
                  f"loss: {self.losses[pt_idx]}")
        time1 = datetime.now()
        time_span0 = time1 - time0

        print("Start optimizing!")
        time0 = datetime.now()
        max_diff_idx = np.argsort(self.acc_diffs)[-1]
        max_diff = self.losses[max_diff_idx]
        while max_diff > 0:
            self.del_losses.append(max_diff)
            self.del_f_pts.append(self.f_pts[max_diff_idx])
            self.del_m_pts.append(self.m_pts[max_diff_idx])

            self.__single_point_optimize(max_diff_idx)

            max_diff_idx = np.argsort(self.acc_diffs)[-1]
            max_diff = self.acc_diffs[max_diff_idx]

        time1 = datetime.now()
        time_span1 = time1 - time0
        print(f"Finished the optimization of FPPs, used {time_span0.total_seconds():.2f} "
              f"seconds to calculate the losses, "
              f"{time_span1.total_seconds():.2f} seconds to optimize.")

    def transform(self):
        """
        Estimate and perform the PL transformation.
        """
        tform = estimate_transform("piecewise-affine", src=np.flip(self.f_pts, axis=1), dst=np.flip(self.m_pts, axis=1))
        reg_img = warp(self.m_img, tform, output_shape=self.f_img.shape, order=self.order)
        mask = polygon2mask(self.f_img.shape, self.f_pts[self.convex_hull.vertices])
        reg_img = np.ma.array(data=reg_img, mask=~mask)
        reg_img.data[reg_img.mask] = np.nan

        return reg_img

