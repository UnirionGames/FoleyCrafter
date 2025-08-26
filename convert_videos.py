import argparse
import subprocess
import re
from pathlib import Path

import torch
import torchaudio
import numpy as np
import torchvision.transforms as transforms
import imageio_ffmpeg
from moviepy.editor import VideoFileClip

from foleycrafter.data.dataset import get_mel

AUDIO_CFG = {
    "sample_rate": 16000,
    "window_size": 1024,
    "hop_size": 160,
    "fmin": 50,
    "fmax": 14000,
}


def probe_video(path: Path) -> dict:
    """Return basic metadata for a video file using ffmpeg."""
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_bin, "-i", str(path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    info = {"format": path.suffix.lstrip(".")}
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    if m:
        h, mnt, s = m.groups()
        info["duration"] = int(h) * 3600 + int(mnt) * 60 + float(s)
    m = re.search(r"Stream #\d+:\d+.*Video: ([^,]+).*?(\d+)x(\d+).*?(\d+(?:\.\d+)?) fps", out)
    if m:
        codec, w, h, fps = m.groups()
        info.update({"video_codec": codec, "width": int(w), "height": int(h), "fps": float(fps)})
    m = re.search(r"Stream #\d+:\d+.*Audio: ([^,]+), (\d+) Hz", out)
    if m:
        codec, sr = m.groups()
        info.update({"audio_codec": codec, "sample_rate": int(sr)})
    return info


def process_video(video_path: Path, feature_dir: Path, video_dir: Path, target_frames: int) -> None:
    meta = probe_video(video_path)
    print(f"Processing {video_path}\n  metadata: {meta}")

    clip = VideoFileClip(str(video_path))
    fps = float(meta.get("fps", clip.fps))
    trim_dur = target_frames / fps
    if clip.duration > trim_dur:
        clip = clip.subclip(0, trim_dur)
    sr = int(meta.get("sample_rate", clip.audio.fps if clip.audio else AUDIO_CFG["sample_rate"]))
    if clip.audio:
        chunks = [chunk for chunk in clip.audio.iter_chunks(fps=sr, chunksize=1024)]
        audio_np = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1))
    else:
        audio_np = np.zeros((0, 1))
    if audio_np.ndim == 2 and audio_np.shape[1] > 1:
        audio_np = audio_np.mean(axis=1, keepdims=True)
    audio = torch.from_numpy(audio_np.T).float()
    if sr != AUDIO_CFG["sample_rate"] and audio.numel() > 0:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_CFG["sample_rate"])(audio)
        meta["sample_rate"] = AUDIO_CFG["sample_rate"]
    mel = get_mel(audio, AUDIO_CFG).squeeze(0)

    frame_count = min(target_frames, int(round(clip.duration * fps)))
    times = np.linspace(0, frame_count / fps, frame_count, endpoint=False)
    frames = [torch.from_numpy(clip.get_frame(t)).permute(2, 0, 1) for t in times]
    clip.close()

    video = torch.stack(frames).float() / 255.0  # (T, C, H, W)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((112, 112)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    video = torch.stack([transform(f) for f in video], dim=0)  # (T, C, H, W)
    video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

    stem = video_path.stem
    torch.save(video, video_dir / f"{stem}.pt")
    torch.save({"mel": mel, "audio_info": meta, "text_embeds": torch.empty(0)}, feature_dir / f"{stem}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert videos into training dataset (clips are trimmed to first N frames)"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("clips"),
        help="Directory containing raw video clips",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/AudioSetStrong/train"),
        help="Root output directory",
    )
    parser.add_argument(
        "--frames", type=int, default=150, help="Frames to keep per clip before trimming"
    )
    args = parser.parse_args()

    print(f"Trimming all clips to {args.frames} frames before processing")

    args.input_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = args.output_dir / "feature"
    video_dir = args.output_dir / "video"
    feature_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    paths = [p for p in sorted(args.input_dir.iterdir()) if p.suffix.lower() in video_exts]
    if not paths:
        print(f"No video files found in {args.input_dir}")
        return

    for p in paths:
        process_video(p, feature_dir, video_dir, args.frames)


if __name__ == "__main__":
    main()
