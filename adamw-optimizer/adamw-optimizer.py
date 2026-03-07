import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    wt_1 = np.array(w,dtype=float)
    mt_1 = np.array(m,dtype=float)
    vt_1 = np.array(v,dtype=float)
    gt = np.array(grad,dtype=float)
    mt = beta1*mt_1 + (1-beta1)*gt
    vt = beta2*vt_1 + (1-beta2)*(gt**2)
    wt = wt_1 - (lr*weight_decay*wt_1)- (lr*mt)/(eps + np.sqrt(vt))
    return (wt,mt,vt)