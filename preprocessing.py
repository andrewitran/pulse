import numpy as np

# The signals initially arrive in a noisy state, so they must first be preprocessed using a
# moving average filter to smooth the signals and enable them to be used for feature extraction. 

def movingavg(signal):
    # The order of the moving average filter is set to 100.
    N = 100 

    temp_signal = np.insert(signal, 0, 0)
    cumsum = np.cumsum(temp_signal)
    filtered = (cumsum[N:] - cumsum[:-N])/N
    
    return filtered

def preprocess(ppg, gsr):
    ppg = movingavg(ppg)
    gsr = movingavg(gsr)
    
    return ppg, gsr
