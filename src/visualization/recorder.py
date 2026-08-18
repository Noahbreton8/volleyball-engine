"""
Episode recorder: saves simulation frames to .npz for later playback.
"""
import numpy as np
import os
from pathlib import Path


def save_recording(frames: list[dict], path: str | Path, metadata: dict | None = None):
    """Save a list of snapshot dicts (from VolleyballEnv._snapshot) to .npz."""
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "qpos":      np.array([f["qpos"]      for f in frames], dtype=np.float32),
        "qvel":      np.array([f["qvel"]      for f in frames], dtype=np.float32),
        "ball_pos":  np.array([f["ball_pos"]  for f in frames], dtype=np.float32),
        "ball_vel":  np.array([f["ball_vel"]  for f in frames], dtype=np.float32),
        "ball_spin": np.array([f["ball_spin"] for f in frames], dtype=np.float32),
        "phase":     np.array([f["phase"]     for f in frames], dtype=np.int8),
    }
    if metadata:
        arrays["_meta_keys"] = np.array(list(metadata.keys()))
        arrays["_meta_vals"] = np.array(list(metadata.values()), dtype=object)

    np.savez_compressed(str(path), **arrays)
    print(f"[recorder] Saved {len(frames)} frames → {path}  "
          f"({path.stat().st_size / 1024:.1f} KB)")


def load_recording(path: str | Path) -> tuple[dict, dict]:
    """
    Load a recording. Returns (arrays_dict, metadata_dict).
    arrays_dict keys: qpos, qvel, ball_pos, ball_vel, ball_spin, phase.
    """
    path = Path(path)
    raw = np.load(str(path), allow_pickle=True)
    arrays = {
        k: raw[k]
        for k in ("qpos", "qvel", "ball_pos", "ball_vel", "ball_spin", "phase")
        if k in raw
    }
    metadata = {}
    if "_meta_keys" in raw:
        for k, v in zip(raw["_meta_keys"], raw["_meta_vals"]):
            metadata[str(k)] = v
    return arrays, metadata


def list_recordings(recordings_dir: str | Path = "recordings") -> list[Path]:
    d = Path(recordings_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime)
