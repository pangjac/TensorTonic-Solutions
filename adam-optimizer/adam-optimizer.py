import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Convert inputs to numpy arrays to allow float multiplication
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v) 
    
    m_t = beta1 * m + (1-beta1)*grad 
    v_t = beta2 * v + (1-beta2)*(grad**2) 

    # Adam initializes the moving averages ( ma and v) at zero. 
    # Because we use decay rates like beta1=0.9, your first step would be 90% zero and only 10% actual gradient, making the update tiny and "biased" toward the inital zeros.
    # dividing by 1-beta**t boosts to reflect the true average of gradient.
    m_t_hat = m_t/(1-beta1 ** t)
    v_t_hat = v_t / (1-beta2 ** t)

    param_t =  param - lr * m_t_hat / (np.sqrt(v_t_hat)+eps)

    # NOTE what we are returnning!
    # In the Adam algorithm, the bias correction is a local calculation used only to compute the current step. 
    # The actual state of the optimizer that gets passed to the next time step must be the exponentially decaying averages, as the **core of the Adam** algorithm is the **Exponentially Weighted** Moving Average (EWMA). 
    return param_t, m_t, v_t
    