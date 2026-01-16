#!/usr/bin/env python3
print("🔥 Starting CLIP pretraining with the model saving")

# --------------------------------------------------
# INSTALL DEPENDENCIES (runtime-safe)
# --------------------------------------------------
import subprocess, sys

pkgs = [
    "webdataset",
    "pillow",
    "ftfy",
    "regex",
    "transformers==4.28.1"
]
for p in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds

from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPFeatureExtractor
)

# --------------------------------------------------
# DATA
# --------------------------------------------------
def get_dataloader(args, tokenizer, image_processor):
    def preprocess(sample):
        image, loc, climate, traffic = sample

        image = image_processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"][0]

        return image, loc, climate, traffic

    def is_valid(sample):
        return (
            "jpg" in sample and
            "loc.txt" in sample and
            "climate.txt" in sample and
            "traffic.txt" in sample
        )

    def log_and_continue(exn):
        print("WebDataset error (skipping):", exn)
        return True
    
    dataset = (
        wds.WebDataset(args.shards, shardshuffle=True,
            handler=log_and_continue,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,)
        .shuffle(1000)
        .decode("pil", handler=log_and_continue)
        .select(is_valid)   # 🔥 THIS IS CRITICAL
        .to_tuple("jpg", "loc.txt", "climate.txt", "traffic.txt", handler=log_and_continue)
        .map(preprocess)
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=True if args.workers > 0 else False,
    )


# --------------------------------------------------
# TRAINING
# --------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    image_processor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = get_dataloader(args, tokenizer, image_processor)
    model.train()

    pbar = tqdm(total=args.steps, desc="CLIP Pretraining")

    step = 0
    loader_iter = iter(loader)

    while step < args.steps:
        try:
            images, loc, climate, traffic = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            continue

        images = images.to(device)

        loss = 0.0
        captions_sets = [loc, climate, traffic]

        for captions in captions_sets:
            text_inputs = tokenizer(
                list(captions),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            outputs = model(
                pixel_values=images,
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                return_dict=True,
            )

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)

            logits = image_embeds @ text_embeds.T
            labels = torch.arange(len(logits)).to(device)

            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)

            loss += (loss_i + loss_t) / 2

        loss /= 3  # average over captions

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        step += 1

    print("💾 Saving model")
    MODEL_DIR = "/opt/ml/model"
    
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    image_processor.save_pretrained(MODEL_DIR)


# --------------------------------------------------
# ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    train(args)
