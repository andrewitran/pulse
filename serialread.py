import bluetooth
import numpy as np

def bt_readline(bt):
    line = ""

    while True:
        temp_byte = bt.recv(1)

        if temp_byte.decode("utf-8") != "\n":
            line += temp_byte.decode("utf-8")
        elif temp_byte.decode("utf-8") == "\n":
            break

    return line

def sample_sensors(bt):
    ppg = []
    gsr = []

    # Send 1 to the Arduino program to signal it to start sampling.
    bt.send(str("1f").encode()) 

    for i in range(0, 8000): # We want 8000 samples for each signal per call.
        temp_string = bt_readline(bt)
        temp_list = [int(i) for i in temp_string.split(",")]

        ppg.append(temp_list[0])
        gsr.append(temp_list[1])

    ppg = np.array(ppg)
    gsr = np.array(gsr)

    return ppg, gsr
