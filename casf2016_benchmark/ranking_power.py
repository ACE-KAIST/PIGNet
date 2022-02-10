import sys
import glob
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

def predictive_index(pred, true):
    n = len(pred)
    ws, cs = [], []
    for i in range(n):
        for j in range(i+1, n):
            w = abs(true[j] - true[i])
            c=-1
            if (pred[j]-pred[i])*(true[j]-true[i])>0: c=1
            elif true[j]-true[i]==0: c=0
            ws.append(w)
            cs.append(c)
    ws = np.array(ws)
    cs = np.array(cs)
    return np.sum(ws*cs)/np.sum(ws)

#make cluster
try:
    with open('../data/CASF-2016/power_ranking/CoreSet.dat') as f:
        lines = f.readlines()[1:]
except FileNotFoundError as e:
    print(e)
    print("please download 'CASF-2016' dataset in '../data' directory.")
    print("Follow the instructions in '../data' directory.")
    exit(-1)
pdbs = [l.split()[0].lower() for l in lines]    
clusters = [pdbs[i*5:i*5+5] for i in range(57)]
pdb_to_true = {l.split()[0]:float(l.split()[3]) for l in lines} 

#filename
filename = sys.argv[1]
n_bootstrap = int(sys.argv[2])
filenames = glob.glob(f'{filename}*')
filenames = sorted(filenames, key=lambda x:int(x.split('_')[-1]))
for fn in filenames:
    with open(fn) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        pdb_to_pred = dict({l[0]:-float(l[2]) for l in lines})

    average_s_r, average_t_r = [], []
    for cluster in clusters:
        no_data = False
        for c in cluster: 
            if c not in pdb_to_pred.keys(): no_data=True
        if no_data: continue        

        preds = [pdb_to_pred[p] for p in cluster]
        preds, ordered_pdb = zip(*sorted(zip(preds, cluster[:])))
        true_order = [1,2,3,4,5] 
        pred_order = [ordered_pdb.index(p)+1 for p in cluster]
        s_r, _ = stats.spearmanr(true_order, pred_order)
        t_r, _ = stats.kendalltau(true_order, pred_order)
        average_s_r.append(s_r)
        average_t_r.append(t_r)
    confidence_interval = bootstrap_confidence(average_s_r, n_bootstrap)
    average_s_r = sum(average_s_r)/len(average_s_r)
    average_t_r = sum(average_t_r)/len(average_t_r)
    pi = predictive_index([pdb_to_pred[k] for k in pdb_to_pred.keys()], 
                          [pdb_to_true[k] for k in pdb_to_pred.keys()])
    print (f'{fn}\t{average_s_r:.3f}\t{average_t_r:.3f}\t{pi:.3f}\t'+
           f'[{confidence_interval[0]:.5f} ~ {confidence_interval[1]:.5f}]')
