from typing import Tuple, Callable
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd

def load_image_dataset(dataset_folder: Path, transform: Callable = None) -> Tuple[Dataset, Dataset, Dataset]:
  train = ImageFolder(dataset_folder / "train", transform=transform)
  valid = ImageFolder(dataset_folder / "validation", transform=transform)
  test = ImageFolder(dataset_folder / "test", transform=transform)
  return (train, valid, test)


def load_perceptual_dataset(json_path: Path):
  perceptual = json.load(json_path.open())
  
  images = []
  for image_id in perceptual.keys():
    image_d = dict()
    image_d["id"] = image_id
    
    image_data = perceptual[image_id]
    # extract action
    k = next(iter(image_data["act"].keys()))
    image_d["action"] = image_data["act"][k]["action_label"]
    
    # extract emotions
    k = next(iter(image_data["em"].keys()))
    image_d["emotion"] = image_data["em"][k]["emotion"]

    # detected objects
    k = next(iter(image_data["od"].keys()))
    # at most 5 objects
    objects = list(set(jo["detected_object"] for jo in image_data["od"][k]["detected_objects"]))
    for i in range(5):
      image_d[f"object_{i}"] = objects[i] if i < len(objects) else "empty"
    
    # art style
    k = next(iter(image_data["as"].keys()))
    image_d["art_style"] = image_data["as"][k]["art_style"]
    
    # color
    k = next(iter(image_data["color"].keys()))
    # at most 5 colors
    colors = list(set(jo["webcolor_name"] for jo in image_data["color"][k]))
    for i in range(5):
      image_d[f"color_{i}"] = colors[i] if i < len(colors) else "empty"
        
    # age detected
    k = next(iter(image_data["age"].keys()))
    image_d["age"] = image_data["age"][k]["age_tier"]

    # add cluster
    if "cluster" in image_data:
      image_d["cluster"] = image_data["cluster"]
    
    image_d["split"] = image_data["split"]
    
    images.append(image_d)
  
  images_df = pd.DataFrame(images)
  return images_df