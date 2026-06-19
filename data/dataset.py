"""
data/dataset.py
ViT5VQADataset — PyTorch Dataset wrapping a DataFrame.
"""

from torch.utils.data import Dataset
from typing import Any, Dict
import pandas as pd

class ViT5VQADataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_key="image_path",
        ocr_key="ocr_path",
        q_key="question",
        a_key="answer",
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_key = image_key
        self.ocr_key = ocr_key
        self.q_key = q_key
        self.a_key = a_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = str(row[self.image_key])
        ocr_path = (
            str(row[self.ocr_key])
            if self.ocr_key in row and pd.notna(row[self.ocr_key])
            else None
        )
        return {
            "image_path": img_path,
            "ocr_path": ocr_path,
            "question": str(row[self.q_key]),
            "answer": (
                ""
                if self.a_key not in row or row[self.a_key] is None
                else str(row[self.a_key])
            ),
            "uid": img_path,
        }