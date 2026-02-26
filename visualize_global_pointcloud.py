"""
Merge all per-frame depth predictions into a single world-space point cloud
and visualize it interactively with Open3D.

Extrinsics convention: (3,4) OpenCV w2c  →  X_cam = R @ X_world + t
So camera-to-world:  X_world = R.T @ (X_cam - t)
"""
import argparse, glob
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


def load_frame(rgb_path: Path, out_dir: Path, conf_threshold: float, border: int = 0):
    stem = rgb_path.stem
    depth_file = out_dir / f"{stem}_depth.npy"
    if not depth_file.exists():
        return None, None

    depth = np.load(depth_file)                          # H, W  float32
    intr  = np.load(out_dir / f"{stem}_intrinsics.npy") # 3, 3
    w2c   = np.load(out_dir / f"{stem}_extrinsics.npy") # 3, 4  (R | t)
    rgb   = np.array(Image.open(rgb_path))               # H, W, 3 uint8

    # resize depth to match rgb if needed (they may differ in resolution)
    h_rgb, w_rgb = rgb.shape[:2]
    if depth.shape != (h_rgb, w_rgb):
        depth = np.array(Image.fromarray(depth).resize((w_rgb, h_rgb), Image.BILINEAR))

    # build conf mask at final (rgb) resolution
    conf_file = out_dir / f"{stem}_conf.npy"
    conf_mask = np.ones((h_rgb, w_rgb), dtype=bool)
    if conf_file.exists() and conf_threshold > 0:
        conf = np.load(conf_file)
        if conf.shape != (h_rgb, w_rgb):
            conf = np.array(Image.fromarray(conf).resize((w_rgb, h_rgb), Image.BILINEAR))
        conf_mask = conf >= conf_threshold

    h, w = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()
    x = ((u.flatten() - cx) * z / fx)
    y = ((v.flatten() - cy) * z / fy)

    pts_cam = np.stack([x, y, z], axis=-1)              # N, 3
    colors  = rgb.reshape(-1, 3) / 255.0                 # N, 3

    # mask out border pixels (distortion / cropping artefacts)
    margin = border if border > 0 else max(4, int(min(h, w) * 0.02))
    border_mask = ((v >= margin) & (v < h - margin) &
                   (u >= margin) & (u < w - margin)).flatten()

    valid = (z > 0) & np.isfinite(z) & conf_mask.flatten() & border_mask
    pts_cam, colors = pts_cam[valid], colors[valid]

    # camera → world:  X_w = R.T @ (X_c - t)
    R, t = w2c[:, :3], w2c[:, 3]
    pts_world = (pts_cam - t) @ R                        # N, 3  (equiv R.T @ (x-t))

    return pts_world, colors


def main():
    parser = argparse.ArgumentParser(description="Visualize all frames as a merged world-space point cloud")
    parser.add_argument("rgb_dir",  help="Directory with RGB images")
    parser.add_argument("out_dir",  help="da3_output directory with .npy files")
    parser.add_argument("--conf",   type=float, default=0.0,
                        help="Min confidence threshold (0=disabled)")
    parser.add_argument("--stride", type=int,   default=1,
                        help="Use every N-th frame (default 1 = all)")
    parser.add_argument("--voxel",  type=float, default=0.0,
                        help="Voxel downsampling size in metres (0=disabled)")
    parser.add_argument("--save",   default="",
                        help="Save merged point cloud to this .ply path")
    parser.add_argument("--border", type=int, default=0,
                        help="Border margin in pixels to crop (0=auto 2%% of image)")
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.out_dir)

    rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
    rgb_files = sorted(set(rgb_files))[::args.stride]
    print(f"Processing {len(rgb_files)} frames (stride={args.stride}) ...")

    all_points, all_colors = [], []
    for i, rgb_path in enumerate(rgb_files):
        pts, cols = load_frame(rgb_path, out_dir, args.conf, args.border)
        if pts is None:
            print(f"  [{i+1}] skip {rgb_path.name} (no depth)")
            continue
        all_points.append(pts)
        all_colors.append(cols)
        print(f"  [{i+1}/{len(rgb_files)}] {rgb_path.name}  +{len(pts)} pts")

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    print(f"\nTotal points: {len(all_points):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"After voxel downsampling ({args.voxel}m): {len(pcd.points):,} pts")

    if args.save:
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"Saved to {args.save}")

    print("Opening interactive viewer  (press Q to quit) ...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Global Point Cloud")
    vis.add_geometry(pcd)

    # draw world-origin axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axes)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
