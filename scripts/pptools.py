def normalization(arr, type='max'):
    """
    Perform normalization on data based on type
    :param args: Numpy array to be normalized
    :param type: Type of normalization
    :return: Normalized array
    """
    import numpy as np

    if type == 'max':
        arr_max = np.max(arr)
        arr /= arr_max

        return np.array(arr), arr_max

    if type == 'meanstd':
        arr_mean = np.mean(arr)
        arr_std = np.std(arr)
        arr = arr - arr_mean / arr_std

        return np.array(arr), arr_mean, arr_std

    if type == '01':
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        arr = (arr - arr_max) / (arr_max - arr_min)

        return np.array(arr), arr_max, arr_min