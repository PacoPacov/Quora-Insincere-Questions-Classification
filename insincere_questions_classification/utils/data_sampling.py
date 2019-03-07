from sklearn.utils import resample


def downsampling(majority, minority, replace=True):
    """ Downsampling the dataset.
    :param majority: Majority dataset.
    :param minority: Minority dataset.
    :param replace=True: Implements resampling with replacement. 
        If False, this will implement (sliced) random permutations.
    """

    majority_downsampled = resample(majority,
                                    replace=replace,
                                    n_samples=minority.shape[0],
                                    random_state=123)
    return pd.concat([majority_downsampled, minority])


def upsampling(majority, minority, replace=True):
    """ Upsampling the dataset.
    :param majority: Majority dataset.
    :param minority: Minority dataset.
    :param replace=True: Implements resampling with replacement. 
        If False, this will implement (sliced) random permutations.
    """

    minority_upsampled = resample(minority,
                                  replace=replace,
                                  n_samples=majority.shape[0],
                                  random_state=123)
    return pd.concat([minority_upsampled, majority])