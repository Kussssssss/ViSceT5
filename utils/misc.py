"""
utils/misc.py
SET_SEED(), _any_device(), show_example(), pick_consistent_indices().
"""

def show_example(df, image_path: str, id_=None, img_id=None, qa=True) -> None:
    img_list = df['image_filename'].unique().tolist()
    img_id = img_id if img_id is not None else random.choice(img_list)
    tmp = df[df['image_filename'] == img_id].reset_index(drop=True)
    img_path = f"{image_path}/{img_id}"
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(img_id)
    plt.axis('off')
    plt.show()

    if qa:
        display(tmp.iloc[:, 1:])

def SET_SEED(seed: int = 42) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

SET_SEED(SEED)

import torch
import torch.nn as nn
import torch.nn.functional as F
import editdistance
import collections
import random
import copy
import numpy as np
import re, unicodedata
from PIL import Image
from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained(model.config.vit5_name)
except NameError:
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")

def _any_device(*objs):
    for o in objs:
        if isinstance(o, dict):
            for v in o.values():
                d = _any_device(v)
                if d is not None:
                    return d
        elif torch.is_tensor(o):
            return o.device
        elif isinstance(o, (list, tuple)):
            for v in o:
                d = _any_device(v)
                if d is not None:
                    return d
    return torch.device("cpu")

import numpy as np
import random

def pick_consistent_indices(num_items: int, k: int, seed: int | None = None):
    k = min(k, num_items)
    rng = np.random.default_rng(seed)
    return rng.choice(num_items, size=k, replace=False).tolist()