def normalization(*args, type='Max'):
    """
    Perform normalization on data based on type
    :param args: Array(s) to be normalized
    :param type: Type of normalization
    :return: Normalized array(s)
    """
    if type == 'Max':
        