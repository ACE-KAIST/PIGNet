import glob
import sys
import numpy as np
from scipy import stats

def bootstrap_confidence(values, n=10000, confidence=0.9 ):
    metrics = []
    for _ in range(n):
        indice = np.random.randint(0, len(values), len(values))
        sampled = [values[i] for i in indice]
        metrics.append(sum(sampled)/len(sampled))
    metrics = np.array(metrics)       
    return stats.t.interval(confidence, len(metrics)-1, 
                loc=np.mean(metrics), scale=np.std(metrics))

#read rmsd
rmsd_filenames = glob.glob("../data/CASF-2016/decoys_docking/*_rmsd.dat")
if len(rmsd_filenames) == 0:
    print("please download 'CASF-2016' dataset in '../data' directory.")
    print("Follow the instructions in '../data' directory.")
    exit(-1)
id_to_rmsd = dict()
for fn in rmsd_filenames:
    with open(fn) as f:
        lines = f.readlines()[1:]
    lines = [l.strip().split() for l in lines]
    for l in lines:
        id_to_rmsd[l[0]] = float(l[1])

#read data
decoy_filenames = glob.glob(sys.argv[1])
n_bootstrap = int(sys.argv[2])
decoy_filenames = sorted(decoy_filenames, key=lambda x:int(x.split('_')[-1]))

for fn in decoy_filenames:
    with open(fn) as f:
        lines = f.readlines()
        lines = [l.strip().split() for l in lines]
    #id to affinity
    id_to_pred = {l[0]:float(l[2]) for l in lines}

    #existing pdb
    pdbs = sorted(list(set([k.split()[0].split('_')[0] for k in id_to_pred.keys()])))
    pdb_success=[]
    for pdb in pdbs:
        selected_keys = [k for k in id_to_pred.keys() if pdb in k]
        pred = [id_to_pred[k] for k in selected_keys]
        pred, sorted_keys = zip(*sorted(zip(pred, selected_keys)))
        rmsd = [id_to_rmsd[k] for k in sorted_keys]
        top_n_success = []
        #print (pdb, sorted_keys[:3], pred[:3], rmsd[:3])
        for top_n in [1,2,3]:
            if min(rmsd[:top_n])<2.0:
                top_n_success.append(1)
            else:             
                top_n_success.append(0)
        pdb_success.append(top_n_success)                
    

    print (fn, end='\t')
    for top_n in [1,2,3]:
        success = [s[top_n-1] for s in pdb_success]
        print (f'{sum(success)/len(success):.3f}', end='\t')
    top1_success = [s[0] for s in pdb_success]
    confidence_interval = bootstrap_confidence(top1_success, n_bootstrap)
    print (f'[{confidence_interval[0]:.5f} ~ {confidence_interval[1]:.5f}]')
