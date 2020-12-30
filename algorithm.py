from algorithm_utils import *
import numpy as np
import pandas as pd
import sys

normal_list = [1.9754, 0.3754, -2.8822, 1.749, 0.206, -2.7693, 1.8632, 0.1239, -2.763, 1.9289, 0.085, -2.8969, 1.9575, 0.2265, -2.9142, 1.8117, 0.1203, -2.7155]
d = 0.1

def calculate(data):
	b_list = []

	for i in range(len(data)):
		b_list.append(constant * (data[i] - normal_list[i - 1]))
	# B1x = data[0]
	# B1y = data[1]
	# B1z = data[2]
	# B2x = data[3]
	# B2y = data[4]
	# B2z = data[5]
	# B3x = data[6]
	# B3y = data[7]
	# B3z = data[8]
	# B4x = data[9]
	# B4y = data[10]
	# B4z = data[11]
	# B5x = data[12]
	# B5y = data[13]
	# B5z = data[14]
	# B6x = data[15]
	# B6y = data[16]
	# B6z = data[17]

	Bxx0 = (b_list[12] - b_list[3]) / (2 * d)
	Byy0 = (b_list[7] - b_list[16]) / (2 * d)
	Bxx1 = (b_list[9] - b_list[0]) / (2 * d)

	Bxxx0 = (b_list[12] + b_list[3] - 2 * b_list[9]) / (d ** 2)
	Bxyx0 = (b_list[13] + b_list[4] - 2 * b_list[10]) / (d ** 2)
	Bxzx0 = (b_list[14] + b_list[5] - 2 * b_list[11]) / (d ** 2)

	Byxy0 = (b_list[6] + b_list[15] - 2 * b_list[9]) / (d ** 2)
	Byyy0 = (b_list[7] + b_list[16] - 2 * b_list[10]) / (d ** 2)
	Byzy0 = (b_list[8] + b_list[17] - 2 * b_list[11]) / (d ** 2)

	Bxxx1 = (b_list[9] + b_list[0] - 2 * b_list[3]) / (d ** 2)
	Bxyx1 = (b_list[10] + b_list[1] - 2 * b_list[4]) / (d ** 2)
	Bxzx1 = (b_list[11] + b_list[2] - 2 * b_list[5]) / (d ** 2)

	first_matrix = np.array([
		[Bxxx0, Bxyx0, Bxzx0],
		[Byxy0, Byyy0, Byzy0],
		[Bxxx1, Bxyx1, Bxzx1]
		])

	second_matrix = np.array([
		[-4 * Bxx0], 
		[-4 * Byy0], 
		[-4 * Bxx1 + d * Bxxx1]
		])

	first_matrix_inverse = np.linalg.inv(first_matrix) 
	result = np.matmul(first_matrix_inverse, second_matrix)
	return result

if __name__ == "__main__":
	collected_data = pd.read_csv(sys.argv[1])

	results = []
	distances = []
	x = []
	y = []
	z = []
	for index, rows in collected_data.iterrows():
		result = calculate(rows[1:len(rows)-2])
		# distance = (result[0] ** 2 + result[1] ** 2 + result[2] ** 2) ** 0.5
		x.extend(result[0])
		y.extend(result[1])
		z.extend(result[2])
		distances.extend((result[0] ** 2 + result[1] ** 2 + result[2] ** 2) ** 0.5)
		# results.append(result)

	df = pd.DataFrame([x, y, z, distances])
	df = df.T
	df.columns = ["x", "y", "z", "distance"]
	df.to_csv("rough_algorithm.csv")
	print(df)
	# print(results)
	# for result in results:
	# 	print(result)
	# print(distances)
