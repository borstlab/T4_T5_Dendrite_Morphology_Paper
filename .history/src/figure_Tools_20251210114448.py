


def asymmetric_mad(data):
    """
    Computes the asymmetric median absolute deviation in a vectorized manner.
    """
    # Calculate the median for each column (axis=0)
    median = np.nanmedian(data, axis=0)

    # Calculate absolute deviations from the median
    # Broadcasting makes this efficient: `data` is (M, N), `median` is (1, N)
    diffs = np.abs(data - median)

    # Create boolean masks for values below and above the median
    below = data < median
    above = data > median

    # Use np.where to replace non-relevant deviations with NaN
    # Where `below` is False, place NaN; otherwise, keep the deviation
    diffs_low = np.where(below, diffs, np.nan)
    # Where `above` is False, place NaN; otherwise, keep the deviation
    diffs_high = np.where(above, diffs, np.nan)

    # Calculate the median of the deviations, ignoring the NaNs we just created
    mad_low = np.nanmedian(diffs_low, axis=0)
    mad_high = np.nanmedian(diffs_high, axis=0)

    return median, mad_low, mad_high
