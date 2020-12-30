import nidaqmx
from utils import *
import statistics

task = nidaqmx.Task()
for port_name in port_list:
    task.ai_channels.add_ai_voltage_chan(port_name)
var_list = []
voltage_list = [[] for k in range(num_of_ports)]
for i in range(100):
    data = task.read()
    for j in range(num_of_ports):
        voltage_list[j].append(data[j])
    i = i + 1

for i in range(num_of_ports):
    var_list.append(statistics.stdev(voltage_list[i]))
print(var_list)