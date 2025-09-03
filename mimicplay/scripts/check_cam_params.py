import h5py
import numpy as np
from mimicplay.utils.triangulation import CameraModel
from scipy.spatial.transform import Rotation as R
import os
import cv2
from tqdm import tqdm 


out_dir = "buffer/demo_0_scaled"
os.makedirs(out_dir, exist_ok=True)
# --- Load HDF5 ---
hdf5_path = "/home/xiaoqi/MimicPlay/mimicplay/datasets/playdata/3d_scaled/demo_0_new.hdf5"   # update with your file path
with h5py.File(hdf5_path, "r") as f:
    # Extract robot0 end-effector positions (605, 1, 3)
    eef_pos = f["data/demo_0/obs/robot0_eef_pos"][:]  # shape (605,1,3)
    eef_pos = eef_pos.squeeze(axis=1)  # now (605, 3)

    # Extract images if needed
    agentview_img = f["data/demo_0/obs/agentview_image"][:] 
    agentview_img2 = f["data/demo_0/obs/agentview_image_2"][:] 

# --- Save raw 3D positions ---
np.savetxt(os.path.join(out_dir, "robot0_eef_pos.txt"), eef_pos, fmt="%.6f")

ZEDA_LEFT_CAM = CameraModel(
    fx=1059.9764404296875,
    fy=1059.9764404296875,
    cx=963.07568359375,
    cy=522.3530883789062,
    R_wc=R.from_quat([-0.404974467935380, -0.808551385290863, 0.425767747250020, 0.031018753461827]).as_matrix(),
    t_wc=np.array([0.903701253331141, 0.444249176547482, 0.598645500102408])
)

ZEDB_RIGHT_CAM = CameraModel(
    fx=1060.0899658203125,
    fy=1059.0899658203125,
    cx=958.9099731445312,
    cy=561.5670166015625,
    R_wc=R.from_quat([0.81395177, -0.40028226, -0.07631803, -0.41404371]).as_matrix(),
    t_wc=np.array([0.11261126, -0.52195948, 0.55795671])
)

# scale factor from 1920x1080 -> 640x360
sx = 640.0 / 1920.0   # = 1/3
sy = 360.0 / 1080.0   # = 1/3

ZEDA_LEFT_CAM  = ZEDA_LEFT_CAM.scaled(sx, sy)
ZEDB_RIGHT_CAM = ZEDB_RIGHT_CAM.scaled(sx, sy)


# --- Project and overlay ---
for i, (pos, img1, img2) in enumerate(tqdm(zip(eef_pos, agentview_img, agentview_img2), total=len(eef_pos))):
    uv1 = ZEDA_LEFT_CAM.project_point(pos).astype(int)
    uv2 = ZEDB_RIGHT_CAM.project_point(pos).astype(int)

    img1_draw = img1.copy()
    img2_draw = img2.copy()

    inside1, inside2 = False, False
    import pdb
    # pdb.set_trace()
    if 0 <= uv1[0] < img1_draw.shape[1] and 0 <= uv1[1] < img1_draw.shape[0]:
        cv2.circle(img1_draw, (uv1[0], uv1[1]), radius=5, color=(0, 255, 0), thickness=-1)
        inside1 = True

    if 0 <= uv2[0] < img2_draw.shape[1] and 0 <= uv2[1] < img2_draw.shape[0]:
        cv2.circle(img2_draw, (uv2[0], uv2[1]), radius=5, color=(0, 255, 0), thickness=-1)
        inside2 = True

    out1 = os.path.join(out_dir, f"agentview1_{i:04d}.png")
    out2 = os.path.join(out_dir, f"agentview2_{i:04d}.png")

    cv2.imwrite(out1, cv2.cvtColor(img1_draw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out2, cv2.cvtColor(img2_draw, cv2.COLOR_RGB2BGR))

    # Print a signal every N frames
    # if i % 20 == 0:
    print(f"[Frame {i}] saved → {out1}, {out2} | inside1={inside1}, inside2={inside2}")
print(f"Saved projections and images in '{out_dir}/'")

