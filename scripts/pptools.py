def maxnorm(arr):
    """
    Performs max-normalization on input array.
    :param arr: Numpy array.
    :return: Normalized Numpy array and max value within input array.
    """
    import numpy as np

    arr_max = np.max(arr)
    arr /= arr_max

    return np.array(arr), arr_max


def znorm(arr):
    """
    Performs Z-normalization on input array.
    :param arr: Numpy array.
    :return: Normalized Numpy array, mean and std value within input array.
    """
    import numpy as np

    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    arr = arr - arr_mean / arr_std

    return np.array(arr), arr_mean, arr_std


def minmaxnorm(arr):
    """
    Performs minmax-normalization on input array.
    :param arr: Numpy array.
    :return: Normalized Numpy array, min and max value within input array.
    """
    import numpy as np

    arr_max = np.max(arr)
    arr_min = np.min(arr)
    arr = (arr - arr_max) / (arr_max - arr_min)

    return np.array(arr), arr_min, arr_max
