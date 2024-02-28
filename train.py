from pathlib import Path

import torch
import numpy as np
import lightning as pl
import os
import shutil
import json
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from itertools import combinations

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization, BayesianEstimator, PC, HillClimbSearch, ExhaustiveSearch, BDeuScore
from pgmpy.inference import VariableElimination
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

import pandas as pd

from artstract_ml.dataset import load_image_dataset, load_perceptual_dataset
from artstract_ml.models.nn import VGG16ImageClassifier, ResNet50ImageClassifier, ViTImageClassifier
from sklearn.metrics import classification_report

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--outdir", type=Path, required=True)

subparsers = parser.add_subparsers(dest="method")

parser_cv = subparsers.add_parser("cv")
parser_cv.add_argument("-m", "--model", choices=["vgg16", "resnet50", "vit"], required=True)
parser_cv.add_argument("--epochs", type=int, default=100)
parser_cv.add_argument("--batch-size", type=int, default=32)
parser_cv.add_argument("--learning-rate", type=float, default=4e-4)
parser_cv.add_argument("--only-head", action="store_true")

parser_ml = subparsers.add_parser("ml")
parser_ml.add_argument("-m", "--model", choices=["nb", "dt", "rf", "xgb", "svm"], required=True)
parser_ml.add_argument("-j", "--json", type=Path, required=True)

parser_bn = subparsers.add_parser("bn")
parser_bn.add_argument("-j", "--json", type=Path, required=True)

parser_comb = subparsers.add_parser("comb")
parser_comb.add_argument("-ml", "--ml-model", choices=["nb", "dt", "rf", "xgb", "svm"], required=True)
parser_comb.add_argument("-ml-path", "--ml-path", type=Path, required=True)
# parser_comb.add_argument("-nn", "--nn-model", choices=["vgg16", "resnet50", "vit"], required=True)
# parser_comb.add_argument("-nn-p", "--nn-path", type=Path, required=True)

MODELS_MAP = {
  "vgg16": VGG16ImageClassifier,
  "resnet50": ResNet50ImageClassifier,
  "vit": ViTImageClassifier,
  "nb": BernoulliNB,
  "dt": DecisionTreeClassifier,
  "rf": RandomForestClassifier,
  "xgb": HistGradientBoostingClassifier,
  "svm": LinearSVC,
}

def train_cv(args):
  NUM_CLASSES = len(os.listdir(args.dataset / "train"))
  wandb_logger = pl.pytorch.loggers.WandbLogger(project="artstract-ml", name=args.name)
  wandb_logger.experiment.config.update(args)

  # automatically overwrite results
  outdir = args.outdir / args.name
  if outdir.exists():
    shutil.rmtree(outdir)
  outdir.mkdir()

  model = MODELS_MAP[args.model](num_classes=NUM_CLASSES, lr=args.learning_rate, only_head=args.only_head)

  train, valid, test = load_image_dataset(args.dataset, transform=model.get_transforms())

  train_dl = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, persistent_workers=True, num_workers=os.cpu_count())
  valid_dl = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True, persistent_workers=True, num_workers=os.cpu_count())
  test_dl = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True, persistent_workers=True, num_workers=os.cpu_count())

  callbacks = [
    pl.pytorch.callbacks.ModelCheckpoint(
      dirpath=outdir,
      save_top_k=1,
      monitor="valid/accuracy",
      mode="max",
      filename="model"),
    pl.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
  ]

  trainer = pl.Trainer(
    max_epochs=args.epochs,
    devices=1, 
    accelerator="gpu",
    log_every_n_steps=10,
    logger=wandb_logger,
    callbacks=callbacks)

  trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
  predictions = trainer.predict(model, dataloaders=test_dl)

  true, preds = zip(*predictions)
  true = np.concatenate(true)
  preds = np.concatenate(preds)

  return trainer, true, preds, test.classes


def train_ml(args):
  # automatically overwrite results
  outdir = args.outdir / args.name
  if outdir.exists():
    shutil.rmtree(outdir)
  outdir.mkdir()

  clf = MODELS_MAP[args.model]()
  train_i, valid_i, test_i = load_image_dataset(args.dataset)
  perceptual = load_perceptual_dataset(args.json)

  train_ids = [i[0].split("/")[-1].split(".")[0] for i in train_i.imgs]
  train = perceptual.set_index("id").loc[train_ids].reset_index()

  test_ids = [i[0].split("/")[-1].split(".")[0] for i in test_i.imgs]
  test = perceptual.set_index("id").loc[test_ids].reset_index()

  valid_ids = [i[0].split("/")[-1].split(".")[0] for i in valid_i.imgs]
  valid = perceptual.set_index("id").loc[valid_ids].reset_index()
  
  TO_REMOVE_COLUMNS = ["id", "color_4", "object_3", "object_4", "age", "split"]
  
  df = pd.concat([train, valid, test], axis=0).drop(TO_REMOVE_COLUMNS, axis=1)
  train = pd.concat([train, valid], axis=0).drop(TO_REMOVE_COLUMNS, axis=1)

  feature_encoder = OneHotEncoder().fit(df.drop("cluster", axis=1))
  label_encoder = LabelEncoder().fit(df["cluster"])

  X_train = feature_encoder.transform(train.drop("cluster", axis=1)).toarray()
  y_train = label_encoder.transform(train["cluster"])
  clf.fit(X_train, y_train)

  dump({
    "clf": clf,
    "fe": feature_encoder,
    "le": label_encoder,
    "train": train,
    "test": test
  }, outdir / "clf.joblib")

  X_test = feature_encoder.transform(test.drop([*TO_REMOVE_COLUMNS, "cluster"], axis=1)).toarray()
  true = label_encoder.transform(test["cluster"]).reshape(-1)
  preds = clf.predict(X_test).reshape(-1)
  target_names = label_encoder.classes_
  return clf, true, preds, target_names


if __name__ == "__main__":
  args = parser.parse_args()

  pl.seed_everything(args.seed)

  if args.method == "cv":
    trainer, true, preds, target_names = train_cv(args)
  elif args.method == "ml":
    clf, true, preds, target_names = train_ml(args)
  elif args.method == "bn":
    # automatically overwrite results
    outdir = args.outdir / args.name
    if outdir.exists():
      shutil.rmtree(outdir)
    outdir.mkdir()

    train_i, valid_i, test_i = load_image_dataset(args.dataset)
    perceptual = load_perceptual_dataset(args.json)

    train_ids = [i[0].split("/")[-1].split(".")[0] for i in train_i.imgs]
    train = perceptual.set_index("id").loc[train_ids].reset_index()

    test_ids = [i[0].split("/")[-1].split(".")[0] for i in test_i.imgs]
    test = perceptual.set_index("id").loc[test_ids].reset_index()

    valid_ids = [i[0].split("/")[-1].split(".")[0] for i in valid_i.imgs]
    valid = perceptual.set_index("id").loc[valid_ids].reset_index()

    TO_REMOVE_COLUMNS = ["id", "color_3", "color_4", "object_2", "object_3", "object_4", "split"]

    df = pd.concat([train, valid, test]).drop(TO_REMOVE_COLUMNS, axis=1)
    train = pd.concat([train, valid]).drop(TO_REMOVE_COLUMNS, axis=1)
    test = test.drop(TO_REMOVE_COLUMNS, axis=1)
    
    model = BayesianNetwork([
      ('cluster', 'age'), ('cluster', 'art_style'), ('cluster', 'action'), ('cluster', 'emotion'),
      ('cluster', 'color_0'), ('cluster', 'color_1'), ('cluster', 'color_2'),
      ('emotion', 'color_0'), ('emotion', 'color_1'), ('emotion', 'color_2'),
      ('cluster', 'object_0'), ('cluster', 'object_1'),
      ('object_0', 'action'), ('object_1', 'action'), 
      *list(combinations(["object_0", "object_1", ], 2))
    ])
    model.fit(train, estimator=MaximumLikelihoodEstimator)
  
    dump(model, outdir / "model.joblib")
    infer = VariableElimination(model)

    preds = []
    MAP_QUERY = infer.map_query(["cluster"])["cluster"]
    for row_idx, row in tqdm(test.drop(["cluster"], axis=1).iterrows()):
      try:
        out = infer.query(["cluster"], evidence=row.to_dict())
        labels = out.state_names["cluster"]
        preds.append(labels[out.values.argmax()])
      except Exception as e:
        preds.append(MAP_QUERY)

    true = test["cluster"]
    target_names = df["cluster"].unique()
    
  print(classification_report(true, preds, target_names=target_names))
  cr = classification_report(true, preds, target_names=target_names, output_dict=True)

  outdir = args.outdir / args.name
  with open(outdir / "results.json", "w") as outjson:
    json.dump(cr, outjson)