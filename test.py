import argparse
import random

import utils

random.seed(0)
import glob
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import DataLoader

import arguments
import models
from dataset import get_dataset_dataloader

args = arguments.parser(sys.argv)
print(args)

# read labels
with open(args.filename) as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    id_to_y = {l[0]: float(l[1]) for l in lines}

with open(args.key_dir + "/test_keys.pkl", "rb") as f:
    test_keys = pickle.load(f)


# model
cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ["CUDA_VISIBLE_DEVICES"] = cmd
if args.model == "pignet":
    model = models.PIGNet(args)
elif args.model == "gnn":
    model = models.GNN(args)
elif args.model == "cnn3d_kdeep":
    model = models.CNN3D_KDEEP(args)
else:
    print(f"No {args.model} model")
    exit(-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.restart_file)

n_param = sum(param.numel() for param in model.parameters() if p.requires_grad)
print("number of parameters : ", n_param)

# dataloader
test_dataset, test_data_loader = get_dataset_dataloader(
    test_keys, args.data_dir, id_to_y, args.batch_size, args.num_workers, False
)

# loss
loss_fn = nn.MSELoss()

# test
st = time.time()

test_losses1 = []
test_losses2 = []

test_pred = dict()
test_pred2 = dict()
test_true = dict()
mc_dropout = args.n_mc_sampling > 1
if mc_dropout:
    epi_var_dict = dict()

model.eval()
for i_batch, sample in enumerate(test_data_loader):
    model.zero_grad()
    if sample is None:
        continue
    sample = utils.dic_to_device(sample, device)
    keys = sample["key"]
    affinity = sample["affinity"]

    if mc_dropout:
        mc_preds = []
        for idx in range(args.n_mc_sampling):
            with torch.no_grad():
                pred, _, _ = model(sample)
            mc_preds.append(pred.data.cpu.numpy())
        mc_preds = np.array(mc_preds)
    else:
        with torch.no_grad():
            pred, _, _ = model(sample)
            pred = pred.data.cpu().numpy()
    affinity = affinity.data.cpu().numpy()

    for idx in range(len(keys)):
        key = keys[idx]
        test_true[key] = affinity[idx]

        if mc_dropout:
            mc_preds_i = mc_preds[i].sum(-1)
            test_pred[key] = np.mean(mc_preds, axis=0)
            epi_var_dict[key] = np.var(mc_preds_i, axis=0)
        else:
            test_pred[key] = pred[idx]

# compute metrics
true_list = np.array([test_true[key] for key in test_true.keys()])
pred_list = np.array([test_pred[key] for key in test_pred.keys()])
test_r2 = r2_score(true_list, pred_list)
r_value = stats.linregress(true_list, pred_list)[2]
end = time.time()

# Write prediction
with open(args.test_result_filename, "w") as writer:
    for key in sorted(test_pred.keys()):
        line = f"{key}\t{test_true[key]:.3f}\t{test_pred[key].sum():.3f}\t"
        writer.write(line)
        if mc_dropout:
            writer.write(f"{epi_var_dict[key]:.3f}")
        else:
            for idx in range(test_pred[key].shape[0]):
                writer.write(f"{test_pred[key][idx]:.3f}")
        writer.write("\n")

# Cal R2
print(f"R2: {test_r2:.3f}")
print(f"R: {r_value:.3f}")
print(f"Time: {end-st:.3f}")
