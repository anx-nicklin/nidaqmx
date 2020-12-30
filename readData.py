import json
from utils import *
import sys
import csv
import pandas as pd

if len(sys.argv) > 1:
	encoded_file_name = sys.argv[1]

	encoded_file = open(encoded_file_name,'r')

	# number_list = []
	data_list = [[] for k in range(num_of_ports)]

	for line in encoded_file.readlines():
		encoded_data = json.loads(line)
		# number_list.extend(encoded_data["x"])
		for i in range(num_of_ports):
			data_list[i].extend(encoded_data["y"][i])
	decoded_file_name = encoded_file_name[8:-5]

	df = pd.DataFrame(data_list).T
	df.columns = port_list
	df.to_csv(decoded_file_name + ".csv")

else:
	print("Please enter encoded file name\nSample usage: python readData.py encoded_data2020-12-11_15-33-29_118998.json")
