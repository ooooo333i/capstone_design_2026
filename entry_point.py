from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJ_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJ_ROOT / "outputs" / "cap_pipeline"
DEFAULT_HAMER_ROOT = PROJ_ROOT / "gvhmr_smplx_visualizing" / "hamer_test" / "hamer"


@dataclass
class GVHMRRun:
    cfg: Any
    output_dir: Path
    hmr4d_results: Path
    vitpose_path: Path
    video_path: Path


@contextlib.contextmanager
def temporary_argv(argv: list[str]):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


@contextlib.contextmanager
def temporary_cwd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def normalize_input_path(raw: str | Path, base: Path | None = None) -> Path:
    text = str(raw).strip().strip("\"'")
    if text.startswith("file://"):
        text = text[7:]
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def prompt_path(
    message: str,
    default: Path | None = None,
    must_exist: bool = False,
    must_be_file: bool = False,
) -> Path:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{message}{suffix}: ").strip()
        if raw == "" and default is None:
            print("Please enter a path.")
            continue
        path = default if raw == "" and default is not None else normalize_input_path(raw)
        if must_exist and not path.exists():
            print(f"Path does not exist: {path}")
            continue
        if must_be_file and not path.is_file():
            print(f"Path is not a file: {path}")
            continue
        return path


def prompt_bool(message: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{message} [{hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes", "1", "true", "t"}:
            return True
        if raw in {"n", "no", "0", "false", "f"}:
            return False
        print("Please type y or n.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Give one video, then get merged GVHMR + HaMeR SMPL-X params."
    )
    parser.add_argument("input_video", nargs="?", help="Input video path.")
    parser.add_argument("--video", type=str, default=None, help="Input video path. If omitted, you will be prompted.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory.")
    parser.add_argument("--static-cam", action="store_true", help="Tell GVHMR the camera is static.")
    parser.add_argument("--use-dpvo", action="store_true", help="Use DPVO instead of GVHMR SimpleVO.")
    parser.add_argument("--f-mm", type=int, default=None, help="Full-frame focal length in mm for GVHMR.")
    parser.add_argument("--verbose", action="store_true", help="Save GVHMR preprocessing overlays.")
    parser.add_argument(
        "--render-preview",
        dest="skip_gvhmr_render",
        action="store_false",
        help="Also render GVHMR preview videos. Off by default for one-step runs.",
    )
    parser.add_argument(
        "--skip-gvhmr-render",
        dest="skip_gvhmr_render",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--force", action="store_true", help="Recompute generated outputs when possible.")
    parser.add_argument("--hamer-root", type=str, default=str(DEFAULT_HAMER_ROOT), help="Local HaMeR repository root.")
    parser.add_argument("--hamer-checkpoint", type=str, default=None, help="Optional HaMeR checkpoint path.")
    parser.add_argument("--hamer-batch-size", type=int, default=1, help="HaMeR inference batch size.")
    parser.add_argument("--hamer-rescale-factor", type=float, default=2.5, help="HaMeR hand crop padding factor.")
    parser.add_argument("--hand-min-conf", type=float, default=0.35, help="Minimum GVHMR/VitPose wrist confidence.")
    parser.add_argument("--skip-result-video", action="store_true", help="Skip the final merged mp4 render.")
    parser.add_argument("--no-interactive", action="store_true", help="Fail instead of prompting for missing values.")
    parser.set_defaults(skip_gvhmr_render=True)
    return parser.parse_args()


def complete_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.video is None and args.input_video is not None:
        args.video = args.input_video
    if args.video is None:
        if args.no_interactive:
            raise SystemExit("--video is required when --no-interactive is used.")
        print("Input a video path. You can drag and drop a file into this terminal.")
        args.video = str(prompt_path("Video", must_exist=True, must_be_file=True))
    return args


def ensure_project_on_path() -> None:
    root = str(PROJ_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def set_static_cam(cfg: Any) -> None:
    try:
        cfg.static_cam = True
        return
    except Exception:
        pass

    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.static_cam = True


def set_cfg_path(cfg: Any, dotted_key: str, value: str) -> None:
    from omegaconf import open_dict

    target = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)

    try:
        setattr(target, parts[-1], value)
    except Exception:
        with open_dict(target):
            setattr(target, parts[-1], value)


def run_gvhmr(
    video: Path,
    output_root: Path,
    static_cam: bool,
    use_dpvo: bool,
    f_mm: int | None,
    verbose: bool,
    render: bool,
    force: bool,
) -> GVHMRRun:
    ensure_project_on_path()
    from tools.demo import demo as gvhmr_demo

    gvhmr_args = [
        "tools/demo/demo.py",
        "--video",
        str(video),
        "--output_root",
        str(output_root),
    ]
    if static_cam:
        gvhmr_args.append("--static_cam")
    if use_dpvo:
        gvhmr_args.append("--use_dpvo")
    if verbose:
        gvhmr_args.append("--verbose")
    if f_mm is not None:
        gvhmr_args.extend(["--f_mm", str(f_mm)])

    with temporary_argv(gvhmr_args):
        cfg = gvhmr_demo.parse_args_to_cfg()

    paths = cfg.paths
    hmr4d_results = Path(paths.hmr4d_results)
    if force and hmr4d_results.exists():
        hmr4d_results.unlink()

    gvhmr_demo.Log.info("[CAP] Running GVHMR preprocess")
    try:
        gvhmr_demo.run_preprocess(cfg)
    except Exception as exc:
        if static_cam or use_dpvo:
            raise
        print(f"[CAP] Warning: moving-camera preprocess failed ({exc}). Falling back to static camera.")
        set_static_cam(cfg)
        gvhmr_demo.run_preprocess(cfg)
    data = gvhmr_demo.load_data_dict(cfg)

    if not hmr4d_results.exists():
        if not gvhmr_demo.torch.cuda.is_available():
            raise RuntimeError("GVHMR demo uses CUDA in this repo, but torch.cuda.is_available() is false.")
        gvhmr_demo.Log.info("[CAP] Running GVHMR prediction")
        model = gvhmr_demo.hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = gvhmr_demo.Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = gvhmr_demo.detach_to_cpu(pred)
        data_time = data["length"] / 30
        gvhmr_demo.Log.info(f"[CAP] GVHMR elapsed: {gvhmr_demo.Log.sync_time() - tic:.2f}s for {data_time:.1f}s")
        gvhmr_demo.torch.save(pred, hmr4d_results)
    else:
        gvhmr_demo.Log.info(f"[CAP] Reusing GVHMR results: {hmr4d_results}")

    if render:
        gvhmr_demo.render_incam(cfg)
        gvhmr_demo.render_global(cfg)
        horiz = Path(paths.incam_global_horiz_video)
        if not horiz.exists():
            gvhmr_demo.merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)

    return GVHMRRun(
        cfg=cfg,
        output_dir=Path(cfg.output_dir),
        hmr4d_results=hmr4d_results,
        vitpose_path=Path(paths.vitpose),
        video_path=Path(cfg.video_path),
    )


def clear_files(folder: Path, patterns: tuple[str, ...]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file():
                path.unlink()


def torch_load_file(torch_module: Any, path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def extract_video_frames(video_path: Path, frame_dir: Path, force: bool) -> int:
    import cv2

    existing = sorted(frame_dir.glob("*.jpg"))
    if existing and not force:
        return len(existing)

    clear_files(frame_dir, ("*.jpg", "*.png"))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(str(frame_dir / f"{frame_idx:04d}.jpg"), frame)
        frame_idx += 1
    capture.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frame_idx


def estimate_hand_boxes_from_coco17(
    keypoints: Any,
    width: int,
    height: int,
    min_conf: float,
) -> tuple["list[list[float]]", "list[int]"]:
    import numpy as np

    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] < 11 or kp.shape[1] < 2:
        return [], []

    boxes: list[list[float]] = []
    is_right: list[int] = []
    specs = (
        (9, 7, 0),  # left wrist, left elbow, HaMeR left-hand flag
        (10, 8, 1),  # right wrist, right elbow, HaMeR right-hand flag
    )

    for wrist_idx, elbow_idx, right_flag in specs:
        wrist = kp[wrist_idx]
        wrist_conf = float(wrist[2]) if kp.shape[1] > 2 else 1.0
        if wrist_conf < min_conf:
            continue

        wrist_xy = wrist[:2]
        center = wrist_xy.copy()
        box_size = 0.12 * max(width, height)

        if elbow_idx < len(kp):
            elbow = kp[elbow_idx]
            elbow_conf = float(elbow[2]) if kp.shape[1] > 2 else 1.0
            if elbow_conf >= min_conf:
                forearm = wrist_xy - elbow[:2]
                forearm_len = float(np.linalg.norm(forearm))
                if forearm_len > 1.0:
                    center = wrist_xy + 0.35 * forearm
                    box_size = max(box_size, 1.6 * forearm_len)

        x1 = max(0.0, float(center[0] - box_size / 2))
        y1 = max(0.0, float(center[1] - box_size / 2))
        x2 = min(float(width - 1), float(center[0] + box_size / 2))
        y2 = min(float(height - 1), float(center[1] + box_size / 2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        boxes.append([x1, y1, x2, y2])
        is_right.append(right_flag)

    return boxes, is_right


def ensure_hamer_on_path(hamer_root: Path) -> None:
    hamer_root = hamer_root.resolve()
    if not (hamer_root / "hamer").exists():
        raise FileNotFoundError(f"HaMeR package directory not found under: {hamer_root}")
    text = str(hamer_root)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def run_hamer_from_gvhmr_keypoints(
    frame_dir: Path,
    vitpose_path: Path,
    out_dir: Path,
    hamer_root: Path,
    checkpoint: Path | None,
    batch_size: int,
    rescale_factor: float,
    min_conf: float,
    force: bool,
) -> Path:
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    ensure_hamer_on_path(hamer_root)

    if out_dir.exists() and any(out_dir.glob("*.npz")) and not force:
        print(f"[CAP] Reusing HaMeR outputs: {out_dir}")
        return out_dir

    clear_files(out_dir, ("*.npz", "*.jpg", "*.png", "*.obj"))

    with temporary_cwd(hamer_root):
        from hamer.configs import CACHE_DIR_HAMER
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
        from hamer.utils import recursive_to

        ckpt = str(checkpoint.resolve()) if checkpoint is not None else DEFAULT_CHECKPOINT
        if checkpoint is None and not Path(ckpt).exists():
            download_models(CACHE_DIR_HAMER)

        model, model_cfg = load_hamer(ckpt)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()

        kp2d = torch_load_file(torch, vitpose_path, map_location="cpu")
        if hasattr(kp2d, "detach"):
            kp2d = kp2d.detach().cpu().numpy()
        else:
            kp2d = np.asarray(kp2d)

        frame_paths = sorted(frame_dir.glob("*.jpg"))
        total = min(len(frame_paths), len(kp2d))
        saved = 0

        for frame_idx, frame_path in tqdm(list(enumerate(frame_paths[:total])), desc="HaMeR"):
            img_cv2 = cv2.imread(str(frame_path))
            if img_cv2 is None:
                continue
            height, width = img_cv2.shape[:2]
            boxes, right = estimate_hand_boxes_from_coco17(kp2d[frame_idx], width, height, min_conf)
            if not boxes:
                continue

            boxes_np = np.asarray(boxes, dtype=np.float32)
            right_np = np.asarray(right, dtype=np.float32)

            debug_img = img_cv2.copy()
            for box, right_flag in zip(boxes_np, right_np):
                color = (0, 255, 0) if int(right_flag) == 1 else (255, 0, 0)
                cv2.rectangle(
                    debug_img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
            cv2.imwrite(str(out_dir / f"{frame_path.stem}_bbox.jpg"), debug_img)

            dataset = ViTDetDataset(
                model_cfg,
                img_cv2,
                boxes_np,
                right_np,
                rescale_factor=rescale_factor,
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                for n in range(batch["img"].shape[0]):
                    person_id = int(batch["personid"][n])
                    right_flag = int(float(batch["right"][n].detach().cpu().numpy()))
                    np.savez(
                        out_dir / f"{frame_path.stem}_{person_id}.npz",
                        vertices=out["pred_vertices"][n].detach().cpu().numpy(),
                        cam_t=out["pred_cam_t"][n].detach().cpu().numpy(),
                        mano_params={k: v[n].detach().cpu().numpy() for k, v in out["pred_mano_params"].items()},
                        is_right=right_flag,
                    )
                    saved += 1

        if saved == 0:
            print("[CAP] Warning: HaMeR produced no hand detections. The merged file will keep GVHMR hand poses.")
        else:
            print(f"[CAP] Saved {saved} HaMeR hand predictions to {out_dir}")

    return out_dir


def hamer_hand_pose_to_axis_angle(hand_pose: Any) -> "Any":
    import cv2
    import numpy as np

    arr = np.asarray(hand_pose)
    arr = np.squeeze(arr)

    if arr.shape == (45,):
        return arr.astype(np.float32)
    if arr.shape == (15, 3):
        return arr.reshape(45).astype(np.float32)
    if arr.shape == (15, 3, 3):
        pose = []
        for joint_matrix in arr:
            aa, _ = cv2.Rodrigues(joint_matrix.astype(np.float64))
            pose.append(aa.reshape(3))
        return np.asarray(pose, dtype=np.float32).reshape(45)

    if arr.size == 45:
        return arr.reshape(45).astype(np.float32)
    raise ValueError(f"Unsupported HaMeR hand_pose shape: {arr.shape}")


def load_hamer_hand_detections(hamer_out_dir: Path, num_frames: int) -> dict[str, dict[int, Any]]:
    import numpy as np

    detections: dict[str, dict[int, Any]] = {"left": {}, "right": {}}
    pattern = re.compile(r"^(\d+)_")
    for npz_path in sorted(hamer_out_dir.glob("*.npz")):
        match = pattern.match(npz_path.stem)
        if match is None:
            continue
        frame_idx = int(match.group(1))
        if frame_idx >= num_frames:
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            side = "right" if int(np.asarray(data["is_right"]).reshape(-1)[0]) == 1 else "left"
            mano_params = data["mano_params"].item()
            detections[side][frame_idx] = hamer_hand_pose_to_axis_angle(mano_params["hand_pose"])
        except Exception as exc:
            print(f"[CAP] Warning: failed to load {npz_path.name}: {exc}")

    return detections


def infer_num_frames_from_smpl_params(params: dict[str, Any]) -> int:
    for key in ("body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose"):
        value = params.get(key)
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer frame count from SMPL-X params.")


def ensure_hand_pose_tensor(params: dict[str, Any], key: str, num_frames: int, dtype: Any, torch_module: Any) -> Any:
    value = params.get(key)
    if hasattr(value, "reshape") and hasattr(value, "shape") and int(value.shape[0]) == num_frames:
        reshaped = value.reshape(num_frames, -1)
        if int(reshaped.shape[1]) == 45:
            return reshaped.clone()
    return torch_module.zeros((num_frames, 45), dtype=dtype)


def merge_hamer_hands_into_gvhmr(
    gvhmr_results: Path,
    hamer_out_dir: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    import torch

    pred = torch_load_file(torch, gvhmr_results, map_location="cpu")
    merged = copy.deepcopy(pred)

    base_params = merged.get("smpl_params_global")
    if base_params is None:
        base_params = merged.get("smpl_params_incam")
    if base_params is None:
        raise KeyError("GVHMR results do not contain smpl_params_global or smpl_params_incam.")

    num_frames = infer_num_frames_from_smpl_params(base_params)
    detections = load_hamer_hand_detections(hamer_out_dir, num_frames)

    for param_key in ("smpl_params_global", "smpl_params_incam"):
        if param_key not in merged:
            continue
        params = merged[param_key]
        dtype = params["body_pose"].dtype if "body_pose" in params else torch.float32
        left = ensure_hand_pose_tensor(params, "left_hand_pose", num_frames, dtype, torch)
        right = ensure_hand_pose_tensor(params, "right_hand_pose", num_frames, dtype, torch)

        for frame_idx, pose in detections["left"].items():
            left[frame_idx] = torch.from_numpy(pose).to(dtype=dtype)
        for frame_idx, pose in detections["right"].items():
            right[frame_idx] = torch.from_numpy(pose).to(dtype=dtype)

        params["left_hand_pose"] = left
        params["right_hand_pose"] = right

    report = {
        "gvhmr_results": str(gvhmr_results),
        "hamer_out_dir": str(hamer_out_dir),
        "merged_results": str(output_path),
        "num_frames": num_frames,
        "left_hand_frames": len(detections["left"]),
        "right_hand_frames": len(detections["right"]),
    }
    merged["cap_merge_meta"] = report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path, report


def render_merged_result_video(gvhmr_run: GVHMRRun, merged_path: Path, force: bool) -> Path:
    from tools.demo import demo as gvhmr_demo

    output_video = gvhmr_run.output_dir / "smplx_merged_hamer_incam.mp4"
    if output_video.exists() and not force:
        print(f"[CAP] Reusing merged result video: {output_video}")
        return output_video
    if output_video.exists():
        output_video.unlink()

    render_cfg = copy.deepcopy(gvhmr_run.cfg)
    set_cfg_path(render_cfg, "paths.hmr4d_results", str(merged_path))
    set_cfg_path(render_cfg, "paths.incam_video", str(output_video))

    print(f"[CAP] Rendering merged result video: {output_video}")
    gvhmr_demo.render_incam(render_cfg)
    return output_video


def main() -> None:
    launch_cwd = Path.cwd()
    args = complete_interactive_args(parse_args())
    ensure_project_on_path()

    video = normalize_input_path(args.video, launch_cwd)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    output_root = normalize_input_path(args.output_root, launch_cwd)
    hamer_root = normalize_input_path(args.hamer_root, launch_cwd)
    hamer_checkpoint = normalize_input_path(args.hamer_checkpoint, launch_cwd) if args.hamer_checkpoint else None

    os.chdir(PROJ_ROOT)

    gvhmr_run = run_gvhmr(
        video=video,
        output_root=output_root,
        static_cam=args.static_cam,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
        verbose=args.verbose,
        render=not args.skip_gvhmr_render,
        force=args.force,
    )

    frame_dir = gvhmr_run.output_dir / "hamer_frames"
    hamer_out_dir = gvhmr_run.output_dir / "hamer_out"
    frame_count = extract_video_frames(gvhmr_run.video_path, frame_dir, args.force)
    print(f"[CAP] Prepared {frame_count} frames for HaMeR: {frame_dir}")

    run_hamer_from_gvhmr_keypoints(
        frame_dir=frame_dir,
        vitpose_path=gvhmr_run.vitpose_path,
        out_dir=hamer_out_dir,
        hamer_root=hamer_root,
        checkpoint=hamer_checkpoint,
        batch_size=args.hamer_batch_size,
        rescale_factor=args.hamer_rescale_factor,
        min_conf=args.hand_min_conf,
        force=args.force,
    )

    merged_path = gvhmr_run.output_dir / "smplx_merged_hamer.pt"
    merged_path, report = merge_hamer_hands_into_gvhmr(gvhmr_run.hmr4d_results, hamer_out_dir, merged_path)
    result_video = None
    if not args.skip_result_video:
        result_video = render_merged_result_video(gvhmr_run, merged_path, args.force)

    print("[CAP] Done")
    print(f"[CAP] Merged SMPL-X params: {merged_path}")
    if result_video is not None:
        print(f"[CAP] Merged result video: {result_video}")
    print(f"[CAP] Left hand frames: {report['left_hand_frames']} / {report['num_frames']}")
    print(f"[CAP] Right hand frames: {report['right_hand_frames']} / {report['num_frames']}")


if __name__ == "__main__":
    main()
