import serialread
import preprocessing
import feature
import classify
import sys
import socket
import numpy as np
import time
import bluetooth
from scipy import signal

trained = classify.train()

bt_address = "20:16:08:04:80:97" # MAC address of our emotion recognition hardware.
bt_port = 1
bt = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

print("Establishing Bluetooth connection...")
bt.connect((bt_address, bt_port))

print("Bluetooth connection established.")
print("Calibrating for your relaxed state. Please wait...")
time.sleep(5)

std_features_r = [0]

# Extract features for 50 seconds when user is relaxed and uses the means and 
# standard deviations of the extracted, relaxed state features for normalization later on.
while 0 in std_features_r:
    features_r = []

    for _ in range (0, 5):
        ppg_r, gsr_r = serialread.sample_sensors(bt)
        ppg_r, gsr_r = preprocessing.preprocess(ppg_r, gsr_r)

        diff_max_mean_gsr_r, diff_mean_min_gsr_r, diff_max_min_gsr_r = feature.extract_gsr(gsr_r)
        mean_hr_r, mean_first_diff_hr_r, global_max_ppi_r, global_min_ppi_r, mean_ppi_r, std_ppi_r, rmssd_ppi_r, ppp50_ppi_r = feature.extract_ppg(ppg_r)

        features_r.append([diff_max_mean_gsr_r, diff_mean_min_gsr_r, diff_max_min_gsr_r, mean_hr_r, mean_first_diff_hr_r, global_max_ppi_r, global_min_ppi_r, mean_ppi_r, std_ppi_r, rmssd_ppi_r, ppp50_ppi_r])

    features_r = np.array(features_r)

    std_features_r = np.std(features_r, axis=0)

mean_features_r = np.mean(features_r, axis=0)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8000))
server.listen(1)

print("Waiting for client connection...")
connection, address = server.accept()

print("Established connection to client!")
print("Now predicting emotions...")
time.sleep(5)

try:
    while True:
        ppg, gsr = serialread.sample_sensors(bt)
        ppg, gsr = preprocessing.preprocess(ppg, gsr)

        diff_max_mean_gsr, diff_mean_min_gsr, diff_max_min_gsr = feature.extract_gsr(gsr)
        mean_hr, mean_first_diff_hr, global_max_ppi, global_min_ppi, mean_ppi, std_ppi, rmssd_ppi, ppp50_ppi = feature.extract_ppg(ppg)

        features = np.array([diff_max_mean_gsr, diff_mean_min_gsr, diff_max_min_gsr, mean_hr, mean_first_diff_hr, global_max_ppi, global_min_ppi, mean_ppi, std_ppi, rmssd_ppi, ppp50_ppi])

        data = (features - mean_features_r) / std_features_r

        print("Diff max mean GSR:", data[0])
        print("Diff mean min GSR:", data[1])
        print("Diff max min GSR:", data[2])
        print("Mean HR:", data[3])
        print("Mean First Diff HR:", data[4])
        print("Global Max PPI:", data[5])
        print("Global Min PPI:", data[6])
        print("Mean PPI:", data[7])
        print("Std PPI:", data[8])
        print("RMSSD:", data[9])
        print("pPP50:", data[10])

        data = data.reshape(1, -1)
        result = classify.test(trained, data)

        # Prints 0 for relaxed and 1 for fearful.
        print(result[0])
        connection.send(result[0].encode('ascii'))

except KeyboardInterrupt:
    pass

connection.shutdown()
connection.close()
server.shutdown()
server.close()
sys.exit()
