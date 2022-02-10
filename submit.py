import os
import socket
import time
from itertools import product

PATH = "/home/wykgroup/mseok/data/DTI_PDBbind/CHEMSCI_REVISION1/"
if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("log"):
    os.mkdir("log")
if not os.path.exists("output"):
    os.mkdir("output")
_exps = os.listdir(PATH)
exps = []
for exp in _exps:
    try:
        int(exp)
        exps.append(exp)
    except:
        pass

seeds = [0, 1, 2, 3]
# names = list(product(seeds, ["350"]))
names = list(product(seeds, ["478"]))
# nodes = ["horus16"] * 4 + ["horus17"] * 4 + ["horus18"] * 4
nodes = ["horus17"] * 4

# for j, model_name in enumerate(["cnn3d_kdeep", "gnn", "harmonic"]):
for j, model_name in enumerate(["harmonic"]):
    for i, exp_name in enumerate(names):
        seed, cluster = exp_name
        idx = j * 4 + i
        node = nodes[idx]
        exp_name = f"{model_name}_{cluster}_{seed}"
        if model_name == "cnn3d_kdeep":
            epoch = 201
            save_every = 10
        elif model_name == "gnn":
            epoch = 1301
            save_every = 100
        elif model_name == "harmonic":
            epoch = 2301
            save_every = 100

        command = f"""#!/bin/bash
#PBS -N MSEOK_dti_{model_name}_{cluster}_{seed}
#PBS -l nodes={node}:ppn=9:gpus=1
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`
date

source ~/mseok/.bashrc
mamba activate mseok

python -u ./train.py --batch_size=8 \
--save_every=100 \
--save_dir=results/{exp_name} \
--tensorboard_dir=runs/{exp_name} \
--n_gnn=3 \
--dim_gnn=128 \
--ngpu=1 \
--train_result_filename=output/{exp_name}.txt \
--test_result_filename=output/{exp_name}.txt \
--train_result_docking_filename=output/{exp_name}.txt \
--test_result_docking_filename=output/{exp_name}.txt \
--train_result_screening_filename=output/{exp_name}.txt \
--test_result_screening_filename=output/{exp_name}.txt \
--loss_der1_ratio=10.0 \
--loss_der2_ratio=10.0 \
--min_loss_der2=-20.0 \
--min_loss_docking=-1.0 \
--loss_docking_ratio=10.0 \
--loss_screening_ratio=5.0 \
--loss_screening2_ratio=5.0 \
--data_dir=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_refined_wonho/data \
--filename=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_refined_wonho/pdb_to_affinity.txt \
--key_dir={os.path.join(PATH, cluster, "refined")} \
--data_dir2=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_docking_nowater_wonho/data/ \
--filename2=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_docking_nowater_wonho/pdb_to_affinity.txt \
--key_dir2={os.path.join(PATH, cluster, "docking")} \
--data_dir3=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_random_screening_wonho/data/ \
--filename3=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_random_screening_wonho/pdb_to_affinity.txt \
--key_dir3={os.path.join(PATH, cluster, "random")} \
--data_dir4=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_cross_screening_wonho/data/ \
--filename4=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_cross_screening_wonho/pdb_to_affinity.txt \
--key_dir4={os.path.join(PATH, cluster, "cross")} \
--num_workers=4 \
--potential={model_name} \
--num_epochs={epoch} \
--dropout_rate=0.1 \
--save_every={save_every} \
--grid_rotation \
--edgeconv \
--dev_vdw_radius=0.2 \
> log/{exp_name}.out 2>log/{exp_name}.err"""

        with open("jobscript_train.x", "w") as w:
            w.write(command)

        os.system("qsub jobscript_train.x")

        time.sleep(25)
