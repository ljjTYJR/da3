import sys, argparse, torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
from debug_pointcloud import depth_to_pointcloud, visualize_pointcloud, render_pointcloud_frame
import cv2

MAKE_VIDEO = True


def w2c_to_c2w_quat(w2c):
    """Convert w2c [3,4] → c2w translation + quaternion (qx qy qz qw)."""
    R = w2c[:3, :3]
    t = w2c[:3,  3]
    R_c2w = R.T
    t_c2w = -R_c2w @ t
    quat  = Rotation.from_matrix(R_c2w).as_quat()   # xyzw order
    return t_c2w, quat


def save_cam_outputs(cam_dir: Path, images: list, prediction, idx_start: int,
                     idx_end: int, out_dir: Path):
    """Save depth/poses/video for one camera from a joint prediction."""
    cam_name  = cam_dir.name
    depth_dir = out_dir / f"{cam_name}_depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    poses_file = out_dir / f"{cam_name}_poses.txt"

    N = idx_end - idx_start

    with open(poses_file, "w") as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        f.write("# Camera-to-world transformation\n")
        f.write(f"# Data source: {cam_dir}\n")
        f.write("# Sampling interval: 1\n")
        for local_i in range(N):
            global_i = idx_start + local_i
            np.save(depth_dir / f"{local_i:06d}.npy", prediction.depth[global_i])
            t, q = w2c_to_c2w_quat(prediction.extrinsics[global_i])
            f.write(f"{local_i} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    print(f"  [{cam_name}] depth → {depth_dir}  poses → {poses_file}")

    if MAKE_VIDEO:
        video_path = out_dir / f"{cam_name}_pointcloud.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (1920, 1080))

    for local_i in range(N):
        global_i = idx_start + local_i
        rgb   = prediction.processed_images[global_i]
        depth = prediction.depth[global_i]
        K     = prediction.intrinsics[global_i]

        points, colors = depth_to_pointcloud(depth, rgb, K)
        depth_valid = depth[depth > 0]
        print(f"  [{cam_name} {local_i+1}/{N}] {images[global_i].name}  "
              f"pts={len(points)}  depth=[{depth_valid.min():.2f}, {depth_valid.max():.2f}]")

        if MAKE_VIDEO:
            frame = render_pointcloud_frame(points, colors)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            visualize_pointcloud(points, colors,
                                 title=f"{cam_name} frame {local_i}: {images[global_i].name}")

    if MAKE_VIDEO:
        writer.release()
        print(f"  [{cam_name}] video → {video_path}")


def find_cam_dirs(root: Path):
    """Return subdirs named cam* with no underscore (excludes cam01_depth etc.)."""
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("cam") and "_" not in d.name
    )


# ── Entry point ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path, help="Scene root directory containing camXX subdirs")
args = parser.parse_args()
root = args.root

cam_dirs = find_cam_dirs(root)
print(f"Found camera dirs: {[d.name for d in cam_dirs]}")

# Gather all images from all cameras, tracking per-camera index ranges
all_images, cam_slices = [], []
for cam_dir in cam_dirs:
    imgs = sorted(cam_dir.glob("*.png"))
    cam_slices.append((cam_dir, len(all_images), len(all_images) + len(imgs)))
    all_images.extend(imgs)

print(f"Total frames across all cameras: {len(all_images)}")

# Single joint inference over all cameras
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
model = model.to(device=device)

prediction = model.inference([str(p) for p in all_images])

# Split outputs per camera and save
out_dir = root / "da3_poses"
out_dir.mkdir(exist_ok=True)
print(f"Output dir: {out_dir}")

for cam_dir, i_start, i_end in cam_slices:
    save_cam_outputs(cam_dir, all_images, prediction, i_start, i_end, out_dir)
