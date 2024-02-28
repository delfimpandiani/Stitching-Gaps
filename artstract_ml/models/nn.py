from typing import Union, List, Callable
from pathlib import Path

from itertools import islice

import torch
import lightning as pl
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from PIL import Image

from typing import List, Dict
import os
import torch
from torch import nn
import numpy as np
import lightning.pytorch as pl

class BaseImageClassifier(pl.LightningModule):
  def __init__(self, num_classes: int, lr: float = 1e-4, only_head: bool = False):
    super().__init__()
    self.num_classes = num_classes
    self.lr = lr
    self.only_head = only_head

  def forward(self, img):
    x = self.model(img)
    x = self.classifier(img)
    return x

  def data_step(self, batch, batch_idx: int, log_name: str = "loss", prog_bar: bool = True) -> torch.optim:
    """
    Perform a step on the input data.

    Args:
        batch: The input batch.
        batch_idx (int): The batch index.
        log_name (str, optional): The name to log the loss into. Defaults to "loss".
        prog_bar (bool, optional): Show the loss on the progression bar. Defaults to True.

    Returns:
        torch.optim: The optimizer result
    """
    image, label_idx = batch

    one_hot_labels = torch.nn.functional.one_hot(
      torch.tensor(label_idx.view(-1)), 
      num_classes=self.num_classes)

    image_emb = self.model(image)
    pred = self.classifier(image_emb)
    loss = self.loss(pred, one_hot_labels.float())
    
    if log_name != "":
      self.log(log_name, loss, prog_bar=prog_bar)
      
    return loss, pred

  def predict(self, image: torch.tensor) -> str:
    """
    Perform a training step.

    Args:
        image (torch.tensor): The input batch.

    Returns:
        str: The output label
    """
    image_emb = self.model(image)
    pred = self.classifier(image_emb)
    return torch.argmax(pred, dim=-1)

  def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    loss, _ = self.data_step(batch, batch_idx, "train/loss", True)
    return loss

  def validation_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    _, label_idx = batch
    loss, pred =  self.data_step(batch, batch_idx, "valid/loss", False)

    preds_idxs = torch.argmax(pred, axis=-1)
    accuracy = (preds_idxs == label_idx.view(-1)).sum() / len(label_idx)

    self.log("valid/accuracy", accuracy)

    return loss

  def predict_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    _, label_idx = batch
    loss, pred =  self.data_step(batch, batch_idx, "", False)
    preds_idxs = torch.argmax(pred, axis=-1)
    return label_idx.view(-1), preds_idxs

  def configure_optimizers(self) -> torch.optim:
    """
    Returns:
        torch.optim: Create the optimizer for the neural network
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "valid/loss"}}


class VGG16ImageClassifier(BaseImageClassifier):
  def __init__(self, num_classes: int, lr: float = 1e-4, only_head: bool = False):
    """
    Initialise the image classifier using a specific model and weight.
    Train according to the findings of [1]
    
    [1] Kandel, I., & Castelli, M. (2020). 
      How deeply to fine-tune a convolutional neural network: a case study using a histopathology dataset. 
      Applied Sciences, 10(10), 3359.

    Args:
        lr (float, optional): Learning rate to be used. Defaults to 1e-4.
        only_head (bool, optional): Train only the classification head or the whole model. Defaults to the whole model.
    """
    super().__init__(num_classes, lr, only_head)
    self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    if self.only_head:
      for param in self.model.parameters():
        param.requires_grad = False

    # last two layers of the classifier are replaced with a Dropout layer
    out_shape = self.model.classifier[-1].in_features
    self.model.classifier[-1] = torch.nn.Dropout(0.5)
    
    self.classifier = nn.Linear(out_shape, self.num_classes)

    self.loss = nn.CrossEntropyLoss()

  def get_transforms(self):
    return VGG16_Weights.IMAGENET1K_V1.transforms()
  

class ResNet50ImageClassifier(BaseImageClassifier):
  def __init__(self, num_classes: int, lr: float = 1e-4, only_head: bool = False):
    super().__init__(num_classes, lr, only_head)
    self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    if self.only_head:
      for param in self.model.parameters():
        param.requires_grad = False
    
    # last two layers of the classifier are replaced with a Dropout layer
    out_shape = self.model.fc.in_features
    self.model.fc = torch.nn.Dropout(0.5)
    
    self.classifier = nn.Linear(out_shape, self.num_classes)

    self.loss = nn.CrossEntropyLoss()

  def get_transforms(self):
    return ResNet50_Weights.IMAGENET1K_V2.transforms()
  

class ViTImageClassifier(BaseImageClassifier):
  def __init__(self, num_classes: int, lr: float = 1e-4, only_head: bool = False):
    """
    ViT Classifier

    Args:
        lr (float, optional): Learning rate to be used. Defaults to 1e-4.
        only_head (bool, optional): Train only the classification head or the whole model. Defaults to the whole model.
        weights (np.array, optional): Class weights. Defaults to None.
    """
    super().__init__(num_classes, lr, only_head)
    self.lr = lr
    self.num_classes = num_classes

    self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    if self.only_head:
      for param in self.model.parameters():
        param.requires_grad = False
    
    out_shape = self.model.head.in_features
    self.model.head = nn.Dropout(0.5)
    
    self.classifier = nn.Linear(out_shape, self.num_classes)
    self.loss = nn.CrossEntropyLoss()

  def get_transforms(self):
    return create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
