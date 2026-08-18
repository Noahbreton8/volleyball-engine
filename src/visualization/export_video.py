"""
Offscreen video exporter. Renders a recorded episode to an MP4 without
needing a display. Copy the output file to your PC to watch.

Usage:
    python -m src.visualization.export_video recordings/test_episode.npz
    python -m src.visualization.export_video --latest
    python -m src.visualization.export_video recordings/test_episode.npz --out clip.mp4 --speed 0.5
"""
import os
import sys
import argparse
import numpy as np
import imageio
from pathlib import Path

# Use EGL for headless offscreen rendering (no display required)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

from .recorder import load_recording, list_recordings
from ..env.scene_builder import build_scene_xml
from ..env.constants import SIM_DT, PHYSICS_STEPS_PER_CONTROL

PLAYBACK_DT = SIM_DT * PHYSICS_STEPS_PER_CONTROL  # simulated seconds per frame
RENDER_FPS = 30
RENDER_W = 1280
RENDER_H = 720


def export(recording_path: str, out_path: str, speed: float = 1.0):
    arrays, metadata = load_recording(recording_path)
    n_frames = arrays["qpos"].shape[0]
    sim_duration = n_frames * PLAYBACK_DT
    print(f"[export] {recording_path}: {n_frames} frames ({sim_duration:.1f}s simulated)")

    xml = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    # Camera: angled overview of the full court
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 1.0]
    camera.distance = 20.0
    camera.azimuth = 135.0
    camera.elevation = -30.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # How many recording frames to skip between rendered frames
    # (speed=1.0 → render every frame at RENDER_FPS, speed=2.0 → 2× faster)
    sim_fps = 1.0 / PLAYBACK_DT
    frame_skip = max(1, int(sim_fps / RENDER_FPS * speed))

    print(f"[export] Rendering {n_frames} sim frames → {out_path}  "
          f"(speed={speed}x, output fps={RENDER_FPS})")

    with imageio.get_writer(str(out_path), fps=RENDER_FPS, codec="libx264",
                             quality=8, macro_block_size=None) as writer:
        for i in range(0, n_frames, frame_skip):
            data.qpos[:] = arrays["qpos"][i]
            data.qvel[:] = arrays["qvel"][i]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera)
            frame = renderer.render()   # (H, W, 3) uint8 RGB
            writer.append_data(frame)

            if i % 100 == 0:
                print(f"  frame {i}/{n_frames}", end="\r")

    renderer.close()
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[export] Done → {out_path}  ({size_mb:.1f} MB)")
    print(f"         Copy to your PC:  scp beast:{out_path.resolve()} .")


def main():
    parser = argparse.ArgumentParser(description="Export episode recording to MP4")
    parser.add_argument("recording", nargs="?", help="Path to .npz recording")
    parser.add_argument("--latest", action="store_true", help="Export most recent recording")
    parser.add_argument("--out", type=str, default=None, help="Output .mp4 path")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--dir", default="recordings", help="Recordings directory")
    args = parser.parse_args()

    if args.latest:
        recs = list_recordings(args.dir)
        if not recs:
            print(f"No recordings found in {args.dir}/")
            sys.exit(1)
        path = str(recs[-1])
        print(f"[export] Latest: {path}")
    elif args.recording:
        path = args.recording
    else:
        parser.print_help()
        sys.exit(1)

    out = args.out or Path(path).with_suffix(".mp4")
    export(path, str(out), speed=args.speed)


if __name__ == "__main__":
    main()
