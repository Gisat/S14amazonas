from datetime import datetime, timedelta
import subprocess

reference_date = datetime(2021, 12, 16)


def compute_next_date(current_date):
    """Compute the next date based on the 12/24 day stepping rule."""
    if current_date + timedelta(days=12) < reference_date:
        return current_date + timedelta(days=12), 12
    else:
        return current_date + timedelta(days=24), 24

def compute_previous_date(current_date):
    """Compute the previous date based on the 12/24 day stepping rule (backward stepping)."""
    if current_date - timedelta(days=12) < reference_date:
        return current_date - timedelta(days=12), 12
    else:
        return current_date - timedelta(days=24), 24

def generate_temporal_extents(start_date, end_date, group_by_n=None):
    current_date = start_date
    extents = []

    while True:
        if group_by_n:
            group_extent = [current_date.strftime('%Y-%m-%d')]
            for _ in range(group_by_n):
                start_current_date = current_date
                next_date, used_timedelta = compute_next_date(current_date)

                if next_date > end_date:
                    next_date = end_date
                    print(f"{start_current_date.strftime('%Y/%m/%d')} - {next_date.strftime('%Y/%m/%d')} = clipped")
                    if start_current_date != next_date:
                        current_date = next_date
                    break
                else:
                    print(f"{start_current_date.strftime('%Y/%m/%d')} - {next_date.strftime('%Y/%m/%d')} = {used_timedelta}")
                    current_date = next_date

            if group_extent[0] != current_date.strftime('%Y-%m-%d'):
                group_extent.append(current_date.strftime('%Y-%m-%d'))
                extents.append(group_extent)

            if current_date >= end_date:
                break
        else:
            period = [current_date.strftime('%Y-%m-%d')]
            next_date, _ = compute_next_date(current_date)

            if next_date > end_date:
                next_date = end_date

            if current_date != next_date:
                period.append(next_date.strftime('%Y-%m-%d'))
                extents.append(period)

            if next_date >= end_date:
                break
            current_date = next_date

    return extents


def get_monthyear_periods_joblist(start_yearmonth, end_yearmonth, n_jobs=3):
    start_date = datetime.strptime(str(start_yearmonth), "%Y%m%d")
    end_date = datetime.strptime(str(end_yearmonth), "%Y%m%d")
    return generate_temporal_extents(start_date, end_date, group_by_n=n_jobs)


def get_temporalextents_mastertemporalextent(start, end):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    temporal_extents = generate_temporal_extents(start_date, end_date)
    master_temporal_extent = [temporal_extents[0][0], temporal_extents[-1][1]]
    return temporal_extents, master_temporal_extent


####



def get_extended_temporalextents_with_padding(start, end, pad_before=5, pad_after=4):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    # Backtrack to get 'pad_before' periods
    backward_dates = []
    temp_start = start_date
    for _ in range(pad_before):
        prev_date, _ = compute_previous_date(temp_start)
        backward_dates.append(prev_date)
        temp_start = prev_date
    backward_dates = sorted(backward_dates)  # Ensure chronological order

    # Forward extend to get 'pad_after' periods
    forward_dates = []
    temp_end = end_date
    for _ in range(pad_after):
        next_date, _ = compute_next_date(temp_end)
        forward_dates.append(next_date)
        temp_end = next_date

    # Final date range
    extended_start = backward_dates[0]
    extended_end = forward_dates[-1]

    temporal_extents = generate_temporal_extents(extended_start, extended_end)
    return temporal_extents, extended_start.strftime('%Y-%m-%d'), extended_end.strftime('%Y-%m-%d')


####
def extract_band(src_filepath, dst_filepath, band_number, datatype="Float32"):
    command = [
        "gdal_translate",
        "-b", str(band_number),
        "-ot", datatype,
        "-co", "TILED=YES",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        str(src_filepath),
        str(dst_filepath)
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Band {band_number} extracted to {dst_filepath}")
    except subprocess.CalledProcessError as e:
        print("Error running gdal_translate:", e)

