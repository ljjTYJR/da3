"""
run_batch.py  –  Run demo.py on every scene subdirectory inside a root folder.

Usage:
    python run_batch.py /media/shuo/T7/robolab/scripts/processed_clips/11_08
    python run_batch.py /media/shuo/T7/robolab/scripts/processed_clips/11_08 --skip-done
    python run_batch.py /media/shuo/T7/robolab/scripts/processed_clips/11_08 --pattern "robodog_*"
"""

import subprocess, sys, argparse, fnmatch
from pathlib import Path


def has_cam_dirs(scene: Path) -> bool:
    """Scene must contain at least one camXX directory to be processable."""
    return any(
        d.is_dir() and d.name.startswith("cam") and "_" not in d.name
        for d in scene.iterdir()
    )


def is_done(scene: Path) -> bool:
    """Consider a scene done if da3_poses/ already exists and is non-empty."""
    out = scene / "da3_poses"
    return out.is_dir() and any(out.iterdir())


def main():
    parser = argparse.ArgumentParser(description="Batch-run demo.py over scene subdirectories.")
    parser.add_argument("root", type=Path, help="Parent directory containing scene subdirs")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip scenes where da3_poses/ already exists")
    parser.add_argument("--pattern", default="*",
                        help="Glob pattern to filter scene names (default: *)")
    parser.add_argument("--demo", type=Path,
                        default=Path(__file__).parent / "demo.py",
                        help="Path to demo.py (default: same dir as this script)")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory")

    scenes = sorted(
        d for d in root.iterdir()
        if d.is_dir() and fnmatch.fnmatch(d.name, args.pattern)
    )
    scenes = [s for s in scenes if has_cam_dirs(s)]

    if not scenes:
        sys.exit("No processable scene directories found.")

    print(f"Found {len(scenes)} scene(s) under {root}")

    skipped, failed, done = [], [], []
    for i, scene in enumerate(scenes, 1):
        if args.skip_done and is_done(scene):
            print(f"[{i}/{len(scenes)}] SKIP (already done): {scene.name}")
            skipped.append(scene.name)
            continue

        print(f"\n{'='*60}")
        print(f"[{i}/{len(scenes)}] Processing: {scene.name}")
        print(f"{'='*60}")

        ret = subprocess.run([sys.executable, str(args.demo), str(scene)])
        if ret.returncode != 0:
            print(f"  WARNING: demo.py exited with code {ret.returncode} for {scene.name}")
            failed.append(scene.name)
        else:
            done.append(scene.name)

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(done)} succeeded, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        print(f"Failed scenes: {failed}")


if __name__ == "__main__":
    main()
