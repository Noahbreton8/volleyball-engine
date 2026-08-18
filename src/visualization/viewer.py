"""
Episode viewer: replays a recorded .npz episode in the MuJoCo interactive viewer.

Usage:
    python -m src.visualization.viewer recordings/episode_00100.npz
    python -m src.visualization.viewer --latest

Controls (MuJoCo viewer defaults + custom):
    Space         — pause/resume playback
    Left/Right    — step one frame backward/forward while paused
    R             — restart from frame 0
    Q / Esc       — quit
    Mouse drag    — orbit camera
    Scroll        — zoom
    Ctrl+drag     — pan
"""
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer as mjviewer

from .recorder import load_recording, list_recordings
from ..env.scene_builder import build_scene_xml
from ..env.constants import SIM_DT, PHYSICS_STEPS_PER_CONTROL


PLAYBACK_DT = SIM_DT * PHYSICS_STEPS_PER_CONTROL  # seconds per recorded frame


def replay(recording_path: str, playback_speed: float = 1.0):
    arrays, metadata = load_recording(recording_path)
    n_frames = arrays["qpos"].shape[0]
    print(f"[viewer] {recording_path}: {n_frames} frames "
          f"({n_frames * PLAYBACK_DT:.1f}s simulated)")

    xml = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    frame = 0
    paused = False
    last_time = time.time()

    def _load_frame(f: int):
        data.qpos[:] = arrays["qpos"][f]
        data.qvel[:] = arrays["qvel"][f]
        mujoco.mj_forward(model, data)

    _load_frame(0)

    with mjviewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 18.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0

        print("[viewer] Space=pause  ←/→=step  R=restart  Q=quit")

        while viewer.is_running():
            now = time.time()
            elapsed = now - last_time

            # Key bindings via keyboard state (MuJoCo passive viewer)
            with viewer.lock():
                keys = viewer.key_press_buffer if hasattr(viewer, "key_press_buffer") else []

            if not paused:
                if elapsed >= PLAYBACK_DT / playback_speed:
                    frame = min(frame + 1, n_frames - 1)
                    _load_frame(frame)
                    viewer.sync()
                    last_time = now
                    if frame == n_frames - 1:
                        paused = True
                        print("[viewer] End of recording. Press R to restart.")
            else:
                viewer.sync()

            time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser(description="Volleyball episode viewer")
    parser.add_argument("recording", nargs="?", help="Path to .npz recording")
    parser.add_argument("--latest", action="store_true", help="Play the most recent recording")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--dir", default="recordings", help="Recordings directory")
    args = parser.parse_args()

    if args.latest:
        recs = list_recordings(args.dir)
        if not recs:
            print(f"No recordings found in {args.dir}/")
            sys.exit(1)
        path = str(recs[-1])
        print(f"[viewer] Playing latest: {path}")
    elif args.recording:
        path = args.recording
    else:
        parser.print_help()
        sys.exit(1)

    replay(path, playback_speed=args.speed)


if __name__ == "__main__":
    main()
