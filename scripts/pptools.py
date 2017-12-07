def normalization(arr, type='Max'):
    """
    Perform normalization on data based on type
    :param args: Numpy array to be normalized
    :param type: Type of normalization
    :return: Normalized array
    """
    import numpy as np

    if type == 'Max':
        arr_max = np.max(arr)
        arr /= arr_max

    return np.array(arr), arr_max