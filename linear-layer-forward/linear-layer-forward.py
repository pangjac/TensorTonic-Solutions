def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X_ = np.array(X)
    W_ = np.array(W)
    b_ = np.array(b)

    res = X_@W_ + b_
    return res.tolist()