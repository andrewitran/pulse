import numpy as np
from scipy import signal

def extract_gsr(gsr):
    diff_max_mean_gsr = np.amax(gsr) - np.mean(gsr) # Calculate difference between the max and mean of GSR.
    diff_mean_min_gsr = np.mean(gsr) - np.amin(gsr) # Calculate difference between the mean and min of GSR.
    diff_max_min_gsr = np.amax(gsr) - np.amin(gsr) # Calculate difference between the max and min of GSR.

    return diff_max_mean_gsr, diff_mean_min_gsr, diff_max_min_gsr  

def extract_ppg(ppg):
    # Derive heart rate (HR) from blood volume pulse (BVP), given by the PPG sensor.
    peaks_temp = signal.argrelextrema(ppg, comparator = np.greater_equal, order = 3)
    peaks_temp = peaks_temp[0]  # Find initial, unfiltered locations of the heart beat peaks.
    peaks = []

    window_start = 0
    window_end = 400
    t = 0.5 # Tolerance for locating true heart beat peaks.
    j = 0

    # Filter peaks in peaks_temp to locate the true heart beat peaks in the samples, separating
    # them from misidentified peaks.
    while window_end != 8400:
        window_mean = np.mean(ppg[window_start : window_end + 1])
        
        for i in peaks_temp:
            if (len(peaks) == 0):
                if (ppg[i] > t * np.amax(ppg[window_start : window_end + 1]) + t * window_mean and i >= window_start and i <= window_end and ppg[i] == np.amax(ppg[window_start : window_end + 1])):
                    peaks.append(i)
            else:
                if (ppg[i] > t * np.amax(ppg[window_start : window_end + 1]) + t * window_mean and i - peaks[j] > window_end and i >= window_start and i <= window_end and ppg[i] == np.amax(ppg[window_start : window_end + 1])):
                    peaks.append(i)
                    j += 1

        # Move the window along the samples to check for true peaks among 400 samples per loop (the size of our window).
        window_start += 400
        window_end += 400

    peaks = np.array(peaks)
    peaks_value = [ppg[a] for a in peaks]

    beat_count = len(peaks)  
    print("Beat count:", beat_count)

    # Calculate instantaneous HR, using 60.0 as seconds per minute.
    if beat_count >= 2:
        time_intervals = (peaks[1 : beat_count - 1] - peaks[0]) / 800.0 # Peak time interval differences from the first peak, using 800.0 as the sampling rate. 
        hr = np.zeros(beat_count - 1)

        for i in range(1, beat_count - 1):
            hr[i - 1] = 60.0 / (time_intervals[i - 1] / i)  
    else:
        hr = np.zeros(1)

    # Extract HR features.
    mean_hr = np.mean(hr)  # Calculate the mean of HR.

    # Calculate the mean of the absolute value of the first difference of the HR values.
    if len(hr) >= 2:
        mean_first_diff_hr = np.mean(np.fabs(np.diff(hr)))  
    else:
        mean_first_diff_hr = 0

    # Derive peak-to-peak interval (PPI) from BVP.
    if len(peaks) >= 2:
        ppi = (np.diff(peaks) * 1000.0) / 800.0
    else:
        ppi = np.zeros(1)

    # Extract heart rate variability (HRV) features.
    global_max_ppi = np.amax(ppi)  # Calculate the global maximum of the PPI signal.
    global_min_ppi = np.amin(ppi)  # Calculate the global minimum of the PPI signal.
    mean_ppi = np.mean(ppi)  # Calculate the mean of the PPI signal.
    std_ppi = np.std(ppi)  # Calculate the standard deviation of the PPI signal.

    # Calculate the square root of the mean of the squares of the differences between adjacent PP intervals (RMSSD).
    if len(ppi) >= 2:
        rmssd = np.sqrt(np.mean(np.power(np.diff(ppi), 2)))  
        ppi_diff = np.fabs(np.diff(ppi)) # First difference of PPI values, used for PP50 calculation.
    else:
        rmssd = 0
        ppi_diff = np.zeros(1) 

    pp50_count = 0

    # Calculate the proportion of the number of pairs of successive PPs that differ by more than 50 ms (PP50), divided by the total number of PPs (pPP50).
    if beat_count >= 2:
        for i in range(0, beat_count - 2):
            if ppi_diff[i] > 50:
                pp50_count += 1

        ppp50 = (float(pp50_count) / float(beat_count)) * 100.0  
    else:
        ppp50 = 0

    return mean_hr, mean_first_diff_hr, global_max_ppi, global_min_ppi, mean_ppi, std_ppi, rmssd, ppp50  
