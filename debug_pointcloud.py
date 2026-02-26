import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import open3d as o3d
import cv2


def load_camera_intrinsics(calib_file):
    calib_path = Path(calib_file)
    data = np.load(calib_file) if calib_path.suffix == '.npy' else np.loadtxt(calib_file)

    if data.shape == (3, 3):
        return data
    if data.shape == (4,):
        fx, fy, cx, cy = data
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    raise ValueError(f"Invalid calibration format: {data.shape}")


def depth_to_pointcloud(depth, rgb, intrinsics):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v, z = u.flatten(), v.flatten(), depth.flatten()

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    colors = rgb.reshape(-1, 3) / 255.0

    valid = (z > 0) & np.isfinite(z)
    return points[valid], colors[valid]


def render_pointcloud_frame(points, colors, width=1920, height=1080):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    return (np.asarray(image) * 255).astype(np.uint8)


def visualize_pointcloud(points, colors, title="Point Cloud"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


def debug_pointcloud_visualization(rgb_dir, depth_dir, calib_file, frame_idx=None, make_video=False,
                                  video_path="pointcloud.mp4"):
    rgb_path, depth_path = Path(rgb_dir), Path(depth_dir)

    print(f"Loading intrinsics from {calib_file}")
    intrinsics = load_camera_intrinsics(calib_file)
    print(f"K =\n{intrinsics}")

    rgb_files = sorted(list(rgb_path.glob("*.png")) + list(rgb_path.glob("*.jpg")) +
                      list(rgb_path.glob("*.jpeg")))
    if not rgb_files:
        raise ValueError(f"No images in {rgb_dir}")

    print(f"Found {len(rgb_files)} images")

    if frame_idx is not None:
        if frame_idx >= len(rgb_files):
            raise ValueError(f"Frame {frame_idx} out of range [0, {len(rgb_files)-1}]")
        rgb_files = [rgb_files[frame_idx]]

    video_writer = None
    if make_video and len(rgb_files) > 1:
        print(f"Creating video: {video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (1920, 1080))

    for i, rgb_file in enumerate(rgb_files):
        print(f"\n[{i+1}/{len(rgb_files)}] {rgb_file.name}")

        rgb = np.array(Image.open(rgb_file))
        h_rgb, w_rgb = rgb.shape[:2]
        stem = rgb_file.stem

        depth_npy = depth_path / f"{stem}_depth.npy"
        depth_npz = depth_path / f"{stem}.npz"
        depth_png = depth_path / f"{stem}_depth.png"

        if depth_npy.exists():
            depth = np.load(depth_npy)
        elif depth_npz.exists():
            data = np.load(depth_npz)
            depth = data['depth']
            print(f"  Loaded depth from .npz")
        elif depth_png.exists():
            depth = np.array(Image.open(depth_png)).astype(np.float32)
            print(f"  Warning: PNG depth lacks metric scale")
        else:
            print(f"  No depth file, skipping")
            continue

        h_depth, w_depth = depth.shape[:2]
        if (h_depth, w_depth) != (h_rgb, w_rgb):
            print(f"  Resizing depth ({h_depth}x{w_depth}) -> ({h_rgb}x{w_rgb})")
            # depth = np.array(Image.fromarray(depth).resize((w_rgb, h_rgb), Image.NEAREST))
            depth = np.array(Image.fromarray(depth).resize((w_rgb, h_rgb), Image.BILINEAR))

        points, colors = depth_to_pointcloud(depth, rgb, intrinsics)
        depth_valid = depth[depth > 0]
        print(f"  Points: {len(points)}, depth: [{depth_valid.min():.2f}, {depth_valid.max():.2f}]")

        if video_writer is not None:
            frame = render_pointcloud_frame(points, colors)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        else:
            visualize_pointcloud(points, colors, f"Frame {i}: {rgb_file.name}")

    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug tool to visualize colorful point clouds from RGB and depth")
    parser.add_argument("rgb_dir", help="Directory containing RGB images")
    parser.add_argument("depth_dir", help="Directory containing depth maps")
    parser.add_argument("calib_file", help="Camera intrinsic calibration file (.txt or .npy)")
    parser.add_argument("--frame", type=int, default=None, help="Specific frame index to visualize (default: all)")
    parser.add_argument("--video", action="store_true", help="Create video instead of interactive visualization")
    parser.add_argument("--output", default="pointcloud.mp4", help="Output video path (default: pointcloud.mp4)")

    args = parser.parse_args()

    debug_pointcloud_visualization(args.rgb_dir, args.depth_dir, args.calib_file, args.frame,
                                  args.video, args.output)
