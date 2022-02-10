import sys
import glob
import statistics
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

def choose_best_pose(id_to_pred):
    """
    pairs = ['_'.join(k.split('_')[:-1]) for k in id_to_pred.keys()]
    pairs = sorted(list(set(pairs)))
    retval = dict()
    for pair in pairs:
        print (pair)
        selected_keys = [k for k in id_to_pred.keys() if pair in k]
        preds = [id_to_pred[k] for k in selected_keys]
        preds, selected_keys = zip(*sorted(zip(preds, selected_keys)))
        retval[selected_keys[0]]=preds[0]
    """
    pairs = ['_'.join(k.split('_')[:-1]) for k in id_to_pred.keys()]
    pairs = sorted(list(set(pairs)))
    retval = {p:[] for p in pairs}
    for k in id_to_pred.keys():
        pair = '_'.join(k.split('_')[:-1])
        retval[pair].append(id_to_pred[k])
    for k in retval.keys():
        retval[k]=min(retval[k])
    return retval

true_binder_list = []
#make cluster
try:
    with open("../data/CASF-2016/power_screening/TargetInfo.dat") as f:
        lines = f.readlines()[9:]
        for l in lines:
            l = l.split()
            true_binder_list+=[(l[0], a) for a in l[1:6]]
except FileNotFoundError as e:
    print(e)
    print("please download 'CASF-2016' dataset in '../data' directory.")
    print("Follow the instructions in '../data' directory.")
    exit(-1)

filename = sys.argv[1]
n_bootstrap = int(sys.argv[2])
filenames = glob.glob(filename)
#filenames = sorted(filenames, key=lambda x:int(x.split('_')[-1]))

for fn in filenames:
    ntb_top = []
    ntb_total = []
    high_affinity_success = []
    with open(fn) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
    id_to_pred = {l[0]:float(l[2]) for l in lines}        
    id_to_pred = choose_best_pose(id_to_pred)
    
    pdbs = sorted(list(set([k.split('_')[0] for k in id_to_pred.keys()])))
    
    for pdb in pdbs:
        selected_keys = [k for k in id_to_pred.keys() if k.split('_')[0]==pdb]
        preds = [id_to_pred[k] for k in selected_keys]
        preds, selected_keys = zip(*sorted(zip(preds, selected_keys)))
        true_binders = [k for k in selected_keys 
                if (k.split('_')[0], k.split('_')[1]) in true_binder_list]
        ntb_top_pdb, ntb_total_pdb, high_affinity_success_pdb = [], [], []
        for top_n in [0.01, 0.05, 0.1]:
            n=int(top_n*len(selected_keys))
            top_keys = selected_keys[:n]
            n_top_true_binder = len(list(set(top_keys)&set(true_binders)))
            ntb_top_pdb.append(n_top_true_binder)
            ntb_total_pdb.append(len(true_binders)*top_n)
            if f'{pdb}_{pdb}' in top_keys: 
                high_affinity_success_pdb.append(1)
            else:
                high_affinity_success_pdb.append(0)
        ntb_top.append(ntb_top_pdb)
        ntb_total.append(ntb_total_pdb)
        high_affinity_success.append(high_affinity_success_pdb)
    print (f'{fn}', end='\t')
    for i in range(3):
        ef = []
        for j in range(len(ntb_total)):
            if ntb_total[j][i]==0: continue
            ef.append(ntb_top[j][i]/ntb_total[j][i])
        if i==0:
            confidence_interval = bootstrap_confidence(ef, n_bootstrap)
        #if i==0: print (ef)
        print (f'{statistics.mean(ef):.3f}', end='\t')            
    print (f'[{confidence_interval[0]:.5f} ~ {confidence_interval[1]:.5f}]', end='\t')
    for i in range(3):
        success = []
        for j in range(len(ntb_total)):
            if high_affinity_success[j][i]>0: 
                success.append(1)
            else:
                success.append(0)
        if i==0:
            confidence_interval = bootstrap_confidence(success, n_bootstrap)
        print (f'{statistics.mean(success):.3f}', end='\t')            
    print (f'[{confidence_interval[0]:.5f} ~ {confidence_interval[1]:.5f}]')


