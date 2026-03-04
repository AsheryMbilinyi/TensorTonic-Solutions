import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """

    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)

    # first moment
    m_new = beta1 * m + (1 - beta1) * grad

    # second moment 
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # bias correction
    m_t = m_new / (1 - beta1 ** t)
    v_t = v_new / (1 - beta2 ** t)

    # parameter update
    param_new = param - lr * (m_t / (np.sqrt(v_t) + eps))

    return (param_new, m_new, v_new)