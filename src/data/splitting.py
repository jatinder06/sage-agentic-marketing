"""Deterministic 70/15/15 dataset splitting."""
from sklearn.model_selection import train_test_split


def split_frame(frame, label_column=None, seed=42):
    stratify = frame[label_column] if label_column and label_column in frame else None
    train, remainder = train_test_split(frame, test_size=0.30, random_state=seed, stratify=stratify)
    remainder_stratify = remainder[label_column] if label_column and label_column in remainder else None
    validation, test = train_test_split(remainder, test_size=0.50, random_state=seed, stratify=remainder_stratify)
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)