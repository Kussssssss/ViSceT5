"""Lightweight, token-wise bridges from image/OCR features to ViT5."""

from torch import nn


class ModalityProjector(nn.Linear):
    """Map each feature to ViT5 without pooling or changing token alignment."""

    def __init__(self, input_dim, d_model):
        super().__init__(input_dim, d_model)
