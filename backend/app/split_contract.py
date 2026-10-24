"""Deterministic counts shared by preview and dataset creation."""
import math
def split_counts(total,train,val,test):
    ratios=[float(train),float(val),float(test)]
    if total<0 or any(not math.isfinite(r) or r<0 or r>1 for r in ratios) or abs(sum(ratios)-1)>1e-6:raise ValueError('划分比例须在 0 到 1 之间且合计为 1。')
    raw=[total*r/sum(ratios) for r in ratios];counts=[math.floor(n) for n in raw]
    order=sorted(range(3),key=lambda i:(-(raw[i]-counts[i]),i))
    for i in order[:total-sum(counts)]:counts[i]+=1
    return dict(zip(['train','val','test'],counts))
