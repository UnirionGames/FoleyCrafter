import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from foleycrafter.data.dataset import AudioSetStrong
from foleycrafter.models.time_detector.model import TimeDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TimeDetector with checkpointing")
    parser.add_argument("--data_path", type=str, default="data/AudioSetStrong/train/feature",
                        help="Path to AudioSetStrong feature directory")
    parser.add_argument("--video_path", type=str, default="data/AudioSetStrong/train/video",
                        help="Path to video frames directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/time_detector_finetune",
                        help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    return parser.parse_args()


def save_checkpoint(state, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = Path(output_dir) / f"checkpoint_epoch_{epoch}.pt"
    torch.save(state, ckpt_path)
    return ckpt_path


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch in tqdm(loader, desc="training", leave=False):
        videos = batch["videos"]  # expect shape (B, C, T, H, W)
        target_mel = batch["mel"]  # (B, n_mels, T_audio)

        videos = videos.to(device)
        target = target_mel.mean(dim=1).to(device)  # collapse mel bins -> (B, T_audio)

        pred = model({"frames": videos})
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * videos.size(0)
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    args = parse_args()

    dataset = AudioSetStrong(data_path=args.data_path, video_path=args.video_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = TimeDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        ckpt_path = save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }, args.output_dir, epoch)
        print(f"Epoch {epoch}: loss={loss:.4f}, saved checkpoint to {ckpt_path}")
