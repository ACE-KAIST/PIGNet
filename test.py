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
import model
from dataset import DTISampler, MolDataset, tensor_collate_fn

args = arguments.parser(sys.argv)
print(args)
print(sys.version)
if os.path.exists(args.test_result_filename):
    exit()

# Read labels
with open(args.filename) as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    id_to_y = {l[0]: float(l[1]) for l in lines}

with open(args.key_dir + "/test_keys.pkl", "rb") as f:
    test_keys = pickle.load(f)


# Model
cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ["CUDA_VISIBLE_DEVICES"] = cmd
if args.potential == "morse":
    model = model.DTILJ(args)
elif args.potential == "morse_all_pair":
    model = model.DTILJAllPair(args)
elif args.potential == "harmonic":
    model = model.DTIHarmonic(args)
elif args.potential == "gnn":
    model = model.GNN(args)
elif args.potential == "cnn3d":
    model = model.CNN3D(args)
elif args.potential == "cnn3d_kdeep":
    model = model.CNN3D_KDEEP(args)
else:
    print(f"No {args.potential} potential")
    exit(-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.restart_file)

# print (model.vina_hbond_coeff)
# print (model.vina_hydrophobic_coeff)
# print (model.torsion_coeff)
# print (model.vdw_coeff)
# exit(-1)
print(
    "number of parameters : ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

# Dataloader
test_dataset = MolDataset(test_keys, args.data_dir, id_to_y)
test_data_loader = DataLoader(
    test_dataset,
    args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=tensor_collate_fn,
)

# loss
loss_fn = nn.MSELoss()

# test
st = time.time()

test_losses1 = []
test_losses2 = []

test_pred1 = dict()
test_pred2 = dict()
test_true = dict()


model.eval()
for i_batch, sample in enumerate(test_data_loader):
    model.zero_grad()
    if sample is None:
        continue
    sample = utils.dic_to_device(sample, device)
    keys = sample["key"]
    affinity = sample["affinity"]

    with torch.no_grad():
        pred1, _, _ = model(sample)
    affinity = affinity.data.cpu().numpy()
    pred1 = pred1.data.cpu().numpy()
    # pred2 = pred2.data.cpu().numpy()

    for i in range(len(keys)):
        test_pred1[keys[i]] = pred1[i]
        # test_pred2[keys[i]] = pred2[i]
        test_true[keys[i]] = affinity[i]
    # if i_batch>2: break

test_r2 = r2_score(
    [test_true[k].sum(-1) for k in test_true.keys()],
    [test_pred1[k].sum(-1) for k in test_true.keys()],
)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    [test_true[k].sum(-1) for k in test_true.keys()],
    [test_pred1[k].sum(-1) for k in test_true.keys()],
)

end = time.time()

# Write prediction
w_test = open(args.test_result_filename, "w")

for k in sorted(test_pred1.keys()):
    w_test.write(f"{k}\t{test_true[k]:.3f}\t")
    w_test.write(f"{test_pred1[k].sum():.3f}\t")
    for j in range(test_pred1[k].shape[0]):
        w_test.write(f"{test_pred1[k][j]:.3f}\t")
    w_test.write("\n")
# w_test.write(f"R2: {test_r2:.3f}\n")
# w_test.write(f"R: {r_value:.3f}\n")
# w_test.write(f"Time: {end-st:.3f}\n\n")
w_test.close()

# Cal R2
print(f"R2: {test_r2:.3f}")
print(f"R: {r_value:.3f}")
print(f"Time: {end-st:.3f}")
