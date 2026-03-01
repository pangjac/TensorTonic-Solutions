import numpy as np
def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    res = series
    for _ in range(order):
        # diff between consectuive itesm 
        res = [res[i]-res[i-1] for i in range(1, len(res))]

    return res 
        

def differencing_sol1(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    series = np.array(series)
    # np.diff handles the recursive subtraction for any 'n' order
    return np.diff(series, n=order)[0]


# Example usage:
# data = [10, 12, 15, 20, 22]
# print(differencing(data, 1)) -> [2, 3, 5, 2]
# Key ConsiderationsData 
# Loss: Every time you apply an order of differencing, you lose exactly one data point from the beginning of your array. 
# If your original series has length $N$, the result will have length $N - d$.
# Stationarity: If your data has a linear trend, $d=1$ usually fixes it. 
# If the trend is quadratic (curved), you likely need $d=2$. Be careful—over-differencing can introduce unnecessary noise into your model.