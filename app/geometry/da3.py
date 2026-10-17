"""Multi-view reconstruction with Depth Anything 3 (DA3-BASE by default; Apache-2.0).

DA3 predicts, for a set of frames, per-pixel depth plus each camera's intrinsics and pose, all in one
shared but *unscaled* frame. Metric scale is applied afterwards from objects of known size
(app/geometry/scale.py); the model is never trusted for absolute size.

The package imports its file-export and benchmark helpers at module load (moviepy, pycolmap, evo,
trimesh). Inference uses none of them, so those two import paths are replaced with stubs before the
first import instead of installing ~1.5 GB of unused dependencies.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# DA3-BASE (0.12B, Apache-2.0). On a real 3-minute sweep DA3-SMALL produced a fragmented room
# (walls did not meet); BASE produced a clean box. Set DA3_MODEL=depth-anything/DA3-SMALL on slow machines.
MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3-BASE")


def _install_stubs() -> None:
    if "depth_anything_3.utils.export" not in sys.modules:
        export = types.ModuleType("depth_anything_3.utils.export")

        def _no_export(*_args, **_kwargs):
            raise RuntimeError("DA3 export is disabled in this project")

        export.export = _no_export
        sys.modules["depth_anything_3.utils.export"] = export
    if "evo" not in sys.modules:
        # Only used to align predicted poses to ground-truth poses in DA3's benchmarks.
        for name in ("evo", "evo.core", "evo.core.trajectory"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["evo.core.trajectory"].PosePath3D = object


@dataclass
class Reconstruction:
    frame_refs: list[str]
    depth: np.ndarray  # [N, h, w] unscaled depth
    conf: np.ndarray  # [N, h, w]
    extrinsics: np.ndarray  # [N, 3, 4] world-to-camera, OpenCV convention
    intrinsics: np.ndarray  # [N, 3, 3] at the processed resolution
    proc_size: tuple[int, int]  # (h, w) of the processed images
    orig_size: tuple[int, int]  # (h, w) of the source frames

    def cam_to_world(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        R, t = self.extrinsics[i, :, :3], self.extrinsics[i, :, 3]
        return R.T, -R.T @ t

    def unproject(self, i: int, uv_orig: np.ndarray) -> np.ndarray:
        """World points for pixel coordinates given in the ORIGINAL frame resolution. uv: [M, 2]."""
        h, w = self.proc_size
        H, W = self.orig_size
        uv = np.asarray(uv_orig, float) * np.array([w / W, h / H])
        u = np.clip(uv[:, 0].round().astype(int), 0, w - 1)
        v = np.clip(uv[:, 1].round().astype(int), 0, h - 1)
        z = self.depth[i, v, u]
        K = self.intrinsics[i]
        x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
        y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x, y, z], axis=1)
        R_wc, c = self.cam_to_world(i)
        return cam @ R_wc.T + c

    def point_cloud(self, conf_quantile: float = 0.3, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """All confident points in world coordinates. Returns (points [M,3], frame index per point)."""
        pts, idx = [], []
        h, w = self.proc_size
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        H, W = self.orig_size
        uv_orig = np.stack([uu.ravel() * W / w, vv.ravel() * H / h], axis=1)
        for i in range(len(self.frame_refs)):
            c = self.conf[i][vv, uu].ravel()
            keep = c >= np.quantile(self.conf[i], conf_quantile)
            p = self.unproject(i, uv_orig[keep])
            pts.append(p)
            idx.append(np.full(len(p), i))
        return np.concatenate(pts), np.concatenate(idx)


_model = None


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def available() -> bool:
    try:
        import torch  # noqa: F401

        _install_stubs()
        import depth_anything_3.api  # noqa: F401
    except Exception:
        return False
    return True


def reconstruct(sweep_dir: Path, frame_refs: list[str], process_res: int = 504) -> Reconstruction:
    global _model
    import torch
    from PIL import Image

    _install_stubs()
    from depth_anything_3.api import DepthAnything3

    if _model is None:
        _model = DepthAnything3.from_pretrained(MODEL_ID).to(device=_device()).eval()
    paths = [str(sweep_dir / r) for r in frame_refs]
    with Image.open(paths[0]) as im:
        W, H = im.size
    with torch.inference_mode():
        pred = _model.inference(paths, process_res=process_res)
    depth = np.asarray(pred.depth, np.float32)
    return Reconstruction(
        frame_refs=frame_refs,
        depth=depth,
        conf=np.asarray(pred.conf, np.float32),
        extrinsics=np.asarray(pred.extrinsics, np.float64),
        intrinsics=np.asarray(pred.intrinsics, np.float64),
        proc_size=depth.shape[1:],
        orig_size=(H, W),
    )
