import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from datasets import load_dataset

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
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda batch: {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }
)

for batch_dict in train_loader:
    batch = batch_dict['pixel_values']
    labels = batch_dict['labels']
    print(f'{batch.shape=}')
    print(f'{labels.shape=}')