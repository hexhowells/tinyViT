import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import v2
from torch.amp import autocast, GradScaler
from datasets import load_dataset

import math

import wandb

from tinyvit.model import ViT
from tinyvit.utils import set_seed, load_config


set_seed(100)

config = load_config()

if config['trainer']['device'] == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = config['trainer']['device']
print(f'Running on device {device}')

wandb.init(
    project="tinyvit",
    config=config,
)


data_files = {
    "train": "/media/datasets/image-datasets/imagenet-100/data/train-*.parquet",
    "val": "/media/datasets/image-datasets/imagenet-100/data/validation-*.parquet",
    }
dataset = load_dataset("parquet", data_files=data_files)

transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def transform_batch(examples):
    examples['pixel_values'] = [transforms(img.convert('RGB')) for img in examples["image"]]
    return examples


dataset.set_transform(transform_batch)

train_loader = DataLoader(
    dataset['train'],  # type: ignore
    batch_size=config['trainer']['batch_size'],
    shuffle=True,
    num_workers=config['trainer']['num_workers'],
    collate_fn=lambda batch: {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }
)

model = ViT(config).to(device)

optimiser = model.configure_optimizers(config['trainer'])
model = torch.compile(model)
model.train()

scaler = GradScaler('cuda')

loss = torch.nn.CrossEntropyLoss()

# learning rate scheduler
learning_rate = config['trainer']['learning_rate']
warmup_steps = config['trainer']['warmup_steps']
min_lr = learning_rate / 10.0
lr_decay_steps = config['trainer']['epochs'] * len(train_loader)

def get_lr(global_step):
    """Computes learning rate with learning rate decay (cosine with warmup)"""
    # linear warmup
    if global_step < warmup_steps:
        return learning_rate * (global_step + 1) / (warmup_steps + 1)
    
    # min LR after decay completes
    if global_step > lr_decay_steps:
        return min_lr
    
    # cosine decay down to min_lr
    decay_ratio = (global_step - warmup_steps) / (lr_decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (learning_rate - min_lr)


global_step = 0
accumulation_steps = config['trainer']['accumulation_steps']

for epoch in range(config['trainer']['epochs']):
    for step, batch_dict in enumerate(train_loader):
        batch = batch_dict['pixel_values'].to(device)
        labels = batch_dict['labels'].to(device)

        with autocast(device_type=device, dtype=torch.bfloat16):
            preds = model(batch)
            loss_val = loss(preds, labels)
            loss_val = loss_val / accumulation_steps

        scaler.scale(loss_val).backward()

        if (step+1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            lr = get_lr(global_step)
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

            scaler.unscale_(optimiser)
            clip_grad_norm_(model.parameters(), config['trainer']['grad_norm_clip'])

            scaler.step(optimiser)
            scaler.update()

            optimiser.zero_grad(set_to_none=True)

            wandb.log({
                "loss": loss_val.item() * accumulation_steps,
                "lr": lr,
                "global_step": global_step
            }, step=global_step)

            global_step += 1

    torch.save(model.state_dict(), 'checkpoints/vit.pth')