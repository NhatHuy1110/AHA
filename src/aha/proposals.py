"""Label-free phase proposals; fixed coverage; one score per unique video second."""
import numpy as np

def onset_curve(prob, window=30):
    p=np.asarray(prob,dtype=np.float64)
    p=np.convolve(np.pad(p,(4,4),mode='edge'),np.ones(9)/9,mode='valid')
    c=np.r_[0,np.cumsum(p)]
    t=np.arange(len(p)); hi=np.minimum(t+window,len(p)); lo=np.maximum(t-window,0)
    after=(c[hi]-c[t])/np.maximum(hi-t,1)
    before=(c[t]-c[lo])/np.maximum(t-lo,1)
    return np.maximum(after-before,0).astype(np.float32)

def select_times(phase, phase_index, config):
    prob=phase[:,phase_index]/np.maximum(phase.sum(1),1e-8)
    curve=onset_curve(prob)
    centers=[]
    for t in np.argsort(-curve,kind='stable'):
        if curve[t] <= 0 or len(centers)>=config['topk']:
            break
        if all(abs(int(t)-c)>=config['nms_seconds'] for c in centers):
            centers.append(int(t))
    centers += np.linspace(0,len(phase)-1,config['coverage'],dtype=int).tolist()
    radius=config['radius']
    times=np.unique(np.concatenate([np.arange(max(0,c-radius),min(len(phase),c+radius+1)) for c in centers]))
    return times.astype(np.int64), sorted(set(centers)), curve

def context_features(hidden,times,radius=4):
    """Actual contiguous before/after context, even across proposal-window gaps."""
    hidden=np.asarray(hidden,dtype=np.float32)
    n=len(hidden)
    sums=np.vstack([np.zeros((1,hidden.shape[1]),np.float32),np.cumsum(hidden,axis=0)])
    lo=np.maximum(times-radius,0); hi=np.minimum(times+radius+1,n)
    before=(sums[times]-sums[lo])/np.maximum(times-lo,1)[:,None]
    after=(sums[hi]-sums[times+1])/np.maximum(hi-times-1,1)[:,None]
    return np.concatenate([hidden[times],before,after],axis=1).astype(np.float32)
