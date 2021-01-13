import copy
port_list = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6", "Dev1/ai7", "Dev1/ai16", "Dev1/ai17", "Dev1/ai18", "Dev1/ai19", 
"Dev1/ai20", "Dev1/ai21", "Dev1/ai22", "Dev1/ai23"]
var_list = [0.0004365496553127318, 0.00023444448933298355, 0.0002318819081556312, 0.00018601543403380557, 0.00021082701820800205, 0.00017087159124938955, 0.00025873157741958963, 
0.00019665109361356387, 0.00036177300621802313, 0.0002467290710328135, 0.000225551196112637, 0.00022064568304907046, 0.0002658352105814952, 0.0002830671651907305, 
# 0.0001988989053616439, 0.0002545969509871539]
0.0014788650909157912, 0.00017007969255281415]
# extra_var_list = [0.0014788650909157912, 0.00017007969255281415]
extra_var_list = [0.0001988989053616439, 0.0002545969509871539]

num_of_ports = len(port_list)
num_of_extra = 2
basic_port_names_1 = ["0x", "0y", "0z", "1x", "1y", "1z", "2x", "2y", "2z", "3x", "3y", "3z", "4x", "4y", "4z", "5x"]
basic_port_names_2 = ["0x", "0y", "0z", "1x", "1y", "1z", "2x", "2y", "2z", "3x", "3y", "3z", "5x", "5y", "5z", "4z"]
group_port_names_1 = ["0x", "1x", "2x", "3x", "0y", "1y", "2y", "3y", "0z", "1z", "2z", "3z", "4x", "5x", "", "", "4y", "", "", "", "4z"]
group_port_names_2 = ["0x", "1x", "2x", "3x", "0y", "1y", "2y", "3y", "0z", "1z", "2z", "3z", "5x", "4z", "", "", "5y", "", "", "", "5z"]
basic_positions_1 = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4], [3, 1], [3, 2], [3, 3], [3, 4], [4, 1], [4, 2], [4, 3], [4, 4]]
basic_positions_2 = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4], [3, 1], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4], [4, 1]]
group_positions_1 = [[1, 1], [2, 1], [3, 1], [1, 2], [2, 2], [3, 2], [1, 3], [2, 3], [3, 3], [1, 4], [2, 4], [3, 4], [4, 1], [5, 1], [6, 1], [4, 2]]
group_positions_2 = [[1, 1], [2, 1], [3, 1], [1, 2], [2, 2], [3, 2], [1, 3], [2, 3], [3, 3], [1, 4], [2, 4], [3, 4], [5, 1], [6, 1], [4, 2], [4, 1]]
replaced_channel = [12, 13]

basic_specs_1 = {"position": basic_positions_1, "num_of_rows": 4, "num_of_cols": 4, "subplot_titles": basic_port_names_1, "color": ["red", "cyan", "yellow", "darkblue", 
"deeppink", "purple", "blue", "brown", "maroon", "orange", "lime", "magenta", "green", "darkkhaki", "darkorange", "darkgreen"]}
basic_specs_2 = {"position": basic_positions_2, "num_of_rows": 4, "num_of_cols": 4, "subplot_titles": basic_port_names_2, "color": ["red", "cyan", "yellow", "darkblue", 
"deeppink", "purple", "blue", "brown", "maroon", "orange", "lime", "magenta", "green", "darkkhaki", "darkorange", "darkgreen"]}
group_specs_1 = {"position": group_positions_1, "num_of_rows": 6, "num_of_cols": 4, "subplot_titles": group_port_names_1, "color": ["red", "red", "red", "green", 
"green", "green", "orange", "orange", "orange", "cyan", "cyan", "cyan", "lime", "lime", "lime", "purple"]}
group_specs_2 = {"position": group_positions_2, "num_of_rows": 6, "num_of_cols": 4, "subplot_titles": group_port_names_2, "color": ["red", "red", "red", "green", 
"green", "green", "orange", "orange", "orange", "cyan", "cyan", "cyan", "lime", "lime", "lime", "purple"]}
port = "5556"
topic = "100"
publisher_interval = 1.5
limit = 50
collect_time_interval = 2
std_difference = 1e-3
collect_folder_name = "collected_data/"
stored_folder_name = "stored_data/"
sample_rate = 1000

points_x = [1, 2, 3, 4, 5, 6]
points_y = [1, 2, 3, 4, 5, 6]
points_z = [1, 2, 3, 4, 5, 6]
num_of_points = len(points_x)
points_color = [1 for k in range(num_of_points)]

matrix_multiplication = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15] + replaced_channel]
