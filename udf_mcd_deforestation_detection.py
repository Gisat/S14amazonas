import xarray as xr
import numpy as np
from scipy import stats
from skimage.morphology import remove_small_holes
from copy import copy
from openeo.udf.debug import inspect

DEBUG = False


def create_nan_mask(numpy_array, vv_vh_bandcount):
    # Create masks for both VV and VH parts of the array
    mask_vv = np.isnan(numpy_array[:vv_vh_bandcount]) | (numpy_array[:vv_vh_bandcount] < -99)
    mask_vh = np.isnan(numpy_array[vv_vh_bandcount:2 * vv_vh_bandcount]) | (
            numpy_array[vv_vh_bandcount:2 * vv_vh_bandcount] < -99)

    # Combine the masks (element-wise logical OR)
    combined_mask = mask_vv | mask_vh
    return combined_mask


def apply_threshold(stat_array, pol_item, DEC_array_threshold,
                    stat_item_name=None, previous_stat_array_bool=None):
    stat_array_copy = copy(stat_array)

    if stat_item_name == 'std':
        if pol_item == "VH":
            pol_thr = 2.0
        else:
            pol_thr = 1.5
        stat_array = np.where(np.isnan(stat_array) | (stat_array < pol_thr), 0, 1)

    elif stat_item_name == 'mean_change':
        pol_thr = -1.75
        stat_array = np.where(np.isnan(stat_array) | (stat_array > pol_thr), 0, 1)

    elif stat_item_name == 'tstat':
        tstat_thr = 3.5
        stat_array = np.where(np.isnan(stat_array) | (stat_array < tstat_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    elif stat_item_name == 'pval':
        pvalue_thr = 0.1
        stat_array = np.where(np.isnan(stat_array) | (stat_array > pvalue_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    elif stat_item_name == 'ratio_slope':
        stat_thr = 0.025
        stat_array = np.where(np.isnan(stat_array) | (stat_array < stat_thr), 0, 1)

    elif stat_item_name == 'ratio_rsquared':
        stat_thr = 0.6
        stat_array = np.where(np.isnan(stat_array) | (stat_array < stat_thr), 0, 1)

    elif stat_item_name == 'ratio_mean_change':
        stat_thr = 2.0
        stat_array = np.where(np.isnan(stat_array) | (stat_array < stat_thr), 0, 1)

    elif stat_item_name == 'ratio_tstat':
        tstat_thr = 3.5
        stat_array = np.where(np.isnan(stat_array) | (stat_array > tstat_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    elif stat_item_name == 'ratio_pval':
        pvalue_thr = 0.1
        stat_array = np.where(np.isnan(stat_array) | (stat_array > pvalue_thr), 0, 1)
        if previous_stat_array_bool is not None:
            stat_array[previous_stat_array_bool == 0] = 0

    DEC_array_threshold += stat_array.astype(int)
    if DEBUG:
        return DEC_array_threshold, stat_array_copy, stat_array.astype(int)
    else:
        return DEC_array_threshold, None, stat_array.astype(int)


def calculate_lsfit_r(vv_vh_r, vv_vh_bandcount):
    """
    Perform linear regression on band differences to calculate slope and R-squared.

    Parameters:
    vv_vh_r (np.array): Difference between VV and VH bands
    vv_vh_bandcount (int): Number of bands per polarization

    Returns:
    tuple: slope array, R-squared array
    """
    x = np.arange(vv_vh_bandcount)
    A = np.c_[x, np.ones_like(x)]

    y = np.reshape(vv_vh_r, (vv_vh_bandcount, -1))

    col_mean = np.nanmean(y, axis=0)
    inds = np.where(np.isnan(y))
    y[inds] = np.take(col_mean, inds[1])

    np.nan_to_num(y, copy=False, nan=0.0)

    m, resid, rank, s = np.linalg.lstsq(A, y, rcond=None)

    slope_r = np.reshape(m[0], vv_vh_r.shape[1:])
    r_squared = 1 - resid / (x.size * np.var(y, axis=0))
    r_squared = np.reshape(r_squared, vv_vh_r.shape[1:])

    return slope_r, r_squared


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    # Squeeze the 'bands' dimension only if its length is 1
    if 'bands' in cube.dims and cube.sizes['bands'] == 1:
        cube_squeezed = cube.squeeze(dim='bands')
    else:
        cube_squeezed = cube

    # Convert to NumPy array (now shape is (2, 100, 100))
    numpy_array = cube_squeezed.values.astype(float)
    del cube_squeezed
    if DEBUG:
        numpy_array = numpy_array[:, 0:128, 0:128]
    numpy_array[numpy_array < -99] = np.nan

    bands, dim1, dim2 = numpy_array.shape
    total_time_steps = bands // 2
    window_size = 10
    half_window = window_size // 2
    vv_vh_bandcount = total_time_steps

    DEC_array_threshold_list = []
    DEC_array_mask_list = []
    DEC_VV_pmin = []
    DEC_VH_pmin = []

    for i in range(total_time_steps - window_size + 1):
        master_combined_metrics = []  # Initialize master list to store metrics for each polarization

        DEC_array_threshold = np.zeros((dim1, dim2), dtype=int)

        # Process bands directly to conserve RAM
        for pol_index, pol_item in enumerate(["VV", "VH"]):

            # Loop twice for the two halves
            # Build index arrays for past and future
            if pol_item == "VV":
                # VV: past starts from i to i+5, future from i+5 to i+10
                pol_stack_past = list(np.arange(i, i + half_window))
                pol_stack_future = list(np.arange(i + half_window, i + window_size))
            else:  # pol_item == "VH"
                # VH bands start after VV
                start_idx = vv_vh_bandcount  # offset for VH bands
                pol_stack_past = list(np.arange(start_idx + i, start_idx + i + half_window))
                pol_stack_future = list(np.arange(start_idx + i + half_window, start_idx + i + window_size))

            # Combine past + future slices
            full_stack_indices = np.concatenate((pol_stack_past, pol_stack_future))
            if DEBUG:
                print(f"time {i}: {pol_item} stack past: {pol_stack_past} ")
                print(f"time {i}: {pol_item} stack future: {pol_stack_future} ")

            # Calculate the mean for Stack_p along the band axis (axis=0)
            Stack_p_MIN = np.nanmean(numpy_array[pol_stack_past], axis=0)

            # Calculate the mean for Stack_f along the band axis (axis=0)
            Stack_f_MIN = np.nanmean(numpy_array[pol_stack_future], axis=0)

            POL_std = np.nanstd(numpy_array[pol_stack_past + pol_stack_future], axis=0)
            DEC_array_threshold, POL_std, _ = apply_threshold(POL_std, pol_item, DEC_array_threshold,
                                                              stat_item_name="std")
            if not DEBUG: del POL_std

            ######## MOVING WINDOW TTEST ON STACK
            POL_mean_change = np.subtract(Stack_f_MIN, Stack_p_MIN)
            DEC_array_threshold, POL_mean_change, POL_mean_change_bool = apply_threshold(POL_mean_change, pol_item,
                                                                                         DEC_array_threshold,
                                                                                         stat_item_name="mean_change")
            if not DEBUG: del POL_mean_change

            if pol_item == "VV":
                DEC_VV_pmin.append((Stack_p_MIN*-1000).astype(int))
            else:
                DEC_VH_pmin.append((Stack_p_MIN*-1000).astype(int))


            # Perform t-test across bands
            ttest_results = stats.ttest_ind(numpy_array[pol_stack_past],
                                            numpy_array[pol_stack_future],
                                            axis=0, nan_policy='omit')
            ttest_pvalue, ttest_tstatistic = ttest_results.pvalue, ttest_results.statistic
            DEC_array_threshold, ttest_pvalue, _ = apply_threshold(ttest_pvalue, pol_item, DEC_array_threshold,
                                                                   stat_item_name="pval",
                                                                   previous_stat_array_bool=POL_mean_change_bool)
            DEC_array_threshold, ttest_tstatistic, _ = apply_threshold(ttest_tstatistic, pol_item, DEC_array_threshold,
                                                                       stat_item_name="tstat",
                                                                       previous_stat_array_bool=POL_mean_change_bool)
            if not DEBUG: del ttest_results, ttest_pvalue, ttest_tstatistic
            del _, POL_mean_change_bool

            if DEBUG:
                combined_metrics = np.stack([
                    Stack_p_MIN,
                    Stack_f_MIN,
                    POL_std,
                    POL_mean_change,
                    ttest_pvalue,
                    ttest_tstatistic
                ], axis=0)

                # Append to master list for both VV and VH
                master_combined_metrics.append(combined_metrics)
            else:
                del pol_stack_past, pol_stack_future, Stack_p_MIN, Stack_f_MIN

        if DEBUG:
            # Convert master list to numpy array for final output
            master_combined_metrics = np.concatenate(master_combined_metrics, axis=0)

        ## VV - VH
        # vv_vh_r = np.subtract(numpy_array[list(np.arange(vv_vh_bandcount))] -
        #                       numpy_array[list(np.arange(vv_vh_bandcount) + vv_vh_bandcount)])
        vv_vh_r = numpy_array[i: i + window_size] - numpy_array[vv_vh_bandcount + i: vv_vh_bandcount + i + window_size]

        ratio_slope, ratio_r_squared = calculate_lsfit_r(vv_vh_r, window_size)
        DEC_array_threshold, ratio_slope, _ = apply_threshold(ratio_slope, pol_item, DEC_array_threshold,
                                                              stat_item_name="ratio_slope")
        DEC_array_threshold, ratio_r_squared, _ = apply_threshold(ratio_r_squared, pol_item, DEC_array_threshold,
                                                                  stat_item_name="ratio_rsquared")

        if not DEBUG:
            del ratio_slope, ratio_r_squared

        # Calculate Mean Change Between Future and Past Stacks
        ratio_mean_change = (np.nanmean(vv_vh_r[list(np.arange(half_window) + half_window)], axis=0) -
                             np.nanmean(vv_vh_r[list(np.arange(half_window))], axis=0)
                             )
        DEC_array_threshold, ratio_mean_change, ratio_mean_change_bool = apply_threshold(ratio_mean_change, pol_item,
                                                                                         DEC_array_threshold,
                                                                                         stat_item_name="ratio_mean_change")

        if not DEBUG:
            del ratio_mean_change

        # Perform T-Test Efficiently
        ratio_ttest_results = stats.ttest_ind(vv_vh_r[list(np.arange(half_window))],
                                              vv_vh_r[list(np.arange(half_window) + half_window)],
                                              axis=0, nan_policy='omit')
        # Extract p-value and t-statistic from the result
        DEC_array_threshold, ratio_ttest_pvalue, _ = apply_threshold(ratio_ttest_results.pvalue, pol_item,
                                                                     DEC_array_threshold,
                                                                     stat_item_name="ratio_pval",
                                                                     previous_stat_array_bool=ratio_mean_change_bool)
        DEC_array_threshold, ratio_ttest_tstatistic, _ = apply_threshold(ratio_ttest_results.statistic, pol_item,
                                                                         DEC_array_threshold,
                                                                         stat_item_name="ratio_tstat",
                                                                         previous_stat_array_bool=ratio_mean_change_bool)

        if not DEBUG:
            del ratio_ttest_results, ratio_mean_change_bool, ratio_ttest_pvalue, ratio_ttest_tstatistic

        if DEBUG:
            ratio_combined_metrics = np.stack([
                ratio_slope,
                ratio_r_squared,
                ratio_mean_change,
                ratio_ttest_pvalue,
                ratio_ttest_tstatistic
            ], axis=0)

        DEC_array_mask = np.zeros_like(DEC_array_threshold)
        DEC_array_mask[DEC_array_threshold > 3] = 1

        # Convert to boolean array (assuming the input is binary, 0 and 1)
        # Invert binary band
        inverted_band = np.logical_not(DEC_array_mask.astype(bool))
        processed_band = remove_small_holes(inverted_band, area_threshold=15)
        # Invert back to original
        DEC_array_mask = np.logical_not(processed_band).astype(np.uint8)

        if DEBUG:
            DEC_array_mask_list.append(DEC_array_mask)
            DEC_array_threshold_list.append(DEC_array_threshold)
            combined_data = np.concatenate([
                master_combined_metrics,  # Shape (1, y, x) for predicted labels
                ratio_combined_metrics  # Shape (n_classes, y, x) for probabilities
            ], axis=0)
        else:
            DEC_array_mask_list.append(DEC_array_mask)
            DEC_array_threshold_list.append(DEC_array_threshold)

    DEC_array_mask_threshold_array = np.stack(DEC_array_mask_list + DEC_array_threshold_list + DEC_VV_pmin + DEC_VH_pmin, axis=0)

    return xr.DataArray(
        DEC_array_mask_threshold_array,
        dims=["bands", "y", "x"],
        coords={
            'y': cube.coords['y'],
            'x': cube.coords['x']
        }
    )