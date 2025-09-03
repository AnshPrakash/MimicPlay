import numpy as np
from scipy.spatial.transform import Rotation as R

class CameraModel:
    def __init__(self, fx, fy, cx, cy, R_wc, t_wc):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.R_wc = R_wc
        self.t_wc = t_wc

    @property
    def K(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    @property
    def R_cw(self):
        return self.R_wc.T

    @property
    def t_cw(self):
        return - self.R_wc.T @ self.t_wc.reshape(3, 1)

    @property
    def P(self):
        Rt = np.hstack((self.R_cw, self.t_cw))  # [R_cw | t_cw]
        return self.K @ Rt

    def scaled(self, sx, sy):
        """Return a new CameraModel with intrinsics scaled by (sx, sy)."""
        return CameraModel(self.fx * sx, self.fy * sy,
                           self.cx * sx, self.cy * sy,
                           self.R_wc.copy(), self.t_wc.copy())
    
    def project_point(self, X):
        X_h = np.hstack((X, 1))  # homogeneous
        x = self.P @ X_h
        x /= x[2]
        return x[:2]

class StereoTriangulator:
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam

    def triangulate_point(self, x_left, x_right):
        P1, P2 = self.left_cam.P, self.right_cam.P
        A = np.zeros((4, 4))
        A[0] = x_left[0] * P1[2] - P1[0]
        A[1] = x_left[1] * P1[2] - P1[1]
        A[2] = x_right[0] * P2[2] - P2[0]
        A[3] = x_right[1] * P2[2] - P2[1]

        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X_h /= X_h[-1]
        return X_h[:3]

    def triangulate_points(self, pts_left, pts_right):
        assert pts_left.shape == pts_right.shape
        pts_3d = [self.triangulate_point(pts_left[i], pts_right[i]) for i in range(len(pts_left))]
        return np.vstack(pts_3d)

    def reconstruction_error(self, pts_3d_true):
        """
            Only for testing purposes.
            Compute the MSE between true 3D points and triangulated points from their projections.
        """
        # Project 3D points into both cameras
        pts_left = np.array([self.left_cam.project_point(X) for X in pts_3d_true])
        pts_right = np.array([self.right_cam.project_point(X) for X in pts_3d_true])

        # Triangulate them back
        pts_3d_est = self.triangulate_points(pts_left, pts_right)

        # Compute MSE in 3D space
        errors = np.linalg.norm(pts_3d_true - pts_3d_est, axis=1) ** 2
        mse = np.mean(errors)
        return mse, pts_3d_est

if __name__ == "__main__":
    left_cam = CameraModel(
        fx=1060.0899658203125,
        fy=1059.0899658203125,
        cx=958.9099731445312,
        cy=561.5670166015625,
        R_wc=R.from_quat([0.81395177, -0.40028226, -0.07631803, -0.41404371]).as_matrix(),
        t_wc=np.array([0.11261126, -0.52195948, 0.55795671])
    )

    right_cam = CameraModel(
        fx=1059.9764404296875,
        fy=1059.9764404296875,
        cx=963.07568359375,
        cy=522.3530883789062,
        R_wc=R.from_quat([-0.404974467935380, -0.808551385290863, 0.425767747250020, 0.031018753461827]).as_matrix(),
        t_wc=np.array([0.903701253331141, 0.444249176547482, 0.598645500102408])
    )

    triangulator = StereoTriangulator(left_cam, right_cam)

    # Generate some sample 3D points within a reasonable range near the cameras
    rng = np.random.default_rng(42)
    pts_3d_true = rng.uniform([0, -1, 1], [2, 1, 3], size=(1000, 3))

    mse, pts_3d_est = triangulator.reconstruction_error(pts_3d_true)

    print("True points:\n", pts_3d_true)
    print("Estimated points:\n", pts_3d_est)
    print("3D Reconstruction MSE:", mse)
