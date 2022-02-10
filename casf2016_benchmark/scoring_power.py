import sys
import glob
from scipy import stats
import numpy as np
from sklearn.utils import resample

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def bootstrap_confidence(true, pred, n=10000, confidence=0.9 ):
    Rs = []
    for _ in range(n):
        indice = np.random.randint(0, len(pred), len(pred))
        t = [true[i] for i in indice]
        p = [pred[i] for i in indice]
        a, b, R, _, std_err = stats.linregress(t, p)
        Rs.append(R)
    Rs = np.array(Rs)       
    return stats.t.interval(confidence, len(Rs)-1, loc=np.mean(Rs), scale=np.std(Rs))
    #return mean_confidence_interval(Rs, confidence)


#filename
filename = sys.argv[1]
n_bootstrap = int(sys.argv[2])
filenames = glob.glob(f'{filename}*')
filenames = sorted(filenames, key=lambda x:int(x.split('_')[-1]))
for fn in filenames:
    with open(fn) as f:
        lines = f.readlines()
    lines = [l.split() for l in lines]
    true = np.array([float(l[1]) for l in lines])
    pred = np.array([float(l[2]) for l in lines])
    a, b, R, _, std_err = stats.linregress(true, pred)
    fit_pred = a*pred+b
    SD = np.power(np.power(true-fit_pred, 2).sum()/(len(true)-1), 0.5)
    confidence_interval = bootstrap_confidence(true, pred, n_bootstrap)
    print (f'{fn}\t{len(pred)}\t{R:.3f}\t{SD:.3f}\t'+
           f'[{confidence_interval[0]:.5f} ~ {confidence_interval[1]:.5f}]')


