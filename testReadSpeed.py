import nidaqmx
import time
from utils import *

task = nidaqmx.Task()
for port_name in port_list:
    task.ai_channels.add_ai_voltage_chan(port_name)

number = []
voltage_list = [[] for k in range(num_of_ports)]

start_time = time.time()

i = 0

data_list = []

cur_time = start_time
while cur_time - start_time < 10:
    data = task.read()
    # data_list.append(data)
    cur_time = time.time()
    i = i + 1

read_time = cur_time - start_time

print("Time: " + str(read_time))
print("Number of data: " + str(i))
print("Speed: " + str(i / read_time))
print("Data: " + str(data_list))
