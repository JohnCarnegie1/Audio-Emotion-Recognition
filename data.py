import os
import wave
import struct
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class RawAudioDataset(Dataset):
    def __init__(self, audio_folder, label_file_path, label_map, transform=None):
        self.audio_folder = audio_folder 
        self.label_map = label_map

        # Load labels
        label_df = pd.read_csv(label_file_path, header=None, names=["filename", "label"])
        label_df["filename"] = label_df["filename"].apply(lambda x: os.path.splitext(x)[0])
        self.labels_dict = dict(zip(label_df["filename"], label_df["label"]))

        self.files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        file_basename = os.path.splitext(filename)[0]
        audio_path = os.path.join(self.audio_folder, filename)
        audio = wave.open(audio_path)
        # waveform, sample_rate = torchaudio.load(audio_path)

        # Get label
        label_name = self.labels_dict.get(file_basename)
        label = self.label_map.get(label_name, -1)
        if label == -1:
            raise ValueError(f"Label for {filename} not found in label_map.")

        return audio, label

import torch
from torch.utils.data import DataLoader, random_split


def create_audio_dataloaders(dataset, batch_size=4, train_ratio=0.7, seed=42):
    """
    Splits raw audio dataset into train/val/test DataLoaders.

    Expected dataset output:
        waveform, label

    waveform shape:
        [1, T]  (raw WAV audio)

    Split:
        70% train
        15% validation
        15% test
    """

    total_size = len(dataset)

    # Train split
    train_size = int(total_size * train_ratio)

    remaining_size = total_size - train_size

    # Split remaining equally into val/test
    val_size = remaining_size // 2
    test_size = remaining_size - val_size

    generator = torch.Generator().manual_seed(seed)

    # Dataset splits 
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader