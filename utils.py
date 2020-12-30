port_list = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6", "Dev1/ai7", "Dev1/ai16", "Dev1/ai17", "Dev1/ai18", "Dev1/ai19", 
"Dev1/ai20", "Dev1/ai21", "Dev1/ai22", "Dev1/ai23"]
var_list = [0.0004365496553127318, 0.00023444448933298355, 0.0002318819081556312, 0.00018601543403380557, 0.00021082701820800205, 0.00017087159124938955, 0.00025873157741958963, 
0.00019665109361356387, 0.00036177300621802313, 0.0002467290710328135, 0.000225551196112637, 0.00022064568304907046, 0.0002658352105814952, 0.0002830671651907305, 
0.0001988989053616439, 0.0002545969509871539]
# 0.0014788650909157912, 0.00017007969255281415]
# extra_var_list = [0.0014788650909157912, 0.00017007969255281415]
extra_var_list = [0.0001988989053616439, 0.0002545969509871539]

num_of_ports = len(port_list)
num_of_extra = 2
basic_specs = {"position": [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4], [3, 1], [3, 2], [3, 3], [3, 4], [4, 1], [4, 2], [4, 3], [4, 4]], 
"num_of_rows": 4, "num_of_cols": 4, "subplot_titles": ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6", "Dev1/ai7", 
"Dev1/ai16", "Dev1/ai17", "Dev1/ai18", "Dev1/ai19", "Dev1/ai20", "Dev1/ai21", "Dev1/ai22", "Dev1/ai23"], "color": ["red", "cyan", "yellow", "darkblue", 
"deeppink", "purple", "blue", "brown", "maroon", "orange", "lime", "magenta", "green", "darkkhaki", "darkorange", "darkgreen"]}
group_specs = {"position": [[1, 1], [2, 1], [3, 1], [1, 2], [2, 2], [3, 2], [1, 3], [2, 3], [3, 3], [1, 4], [2, 4], [3, 4], [4, 1], [5, 1], [6, 1], [4, 2]], 
"num_of_rows": 6, "num_of_cols": 4, "subplot_titles": ["Dev1/ai0", "Dev1/ai3", "Dev1/ai6", "Dev1/ai17", "Dev1/ai1", "Dev1/ai4", "Dev1/ai7", "Dev1/ai18", 
"Dev1/ai2", "Dev1/ai5", "Dev1/ai16", "Dev1/ai19", "Dev1/ai20", "Dev1/ai23", "", "", "Dev1/ai21", "", "", "", "Dev1/ai22"], "color": ["red", "red", "red", "green", 
"green", "green", "orange", "orange", "orange", "cyan", "cyan", "cyan", "lime", "lime", "lime", "purple"]}
port = "5556"
topic = "100"
publisher_interval = 1.5
limit = 50
collect_time_interval = 2
std_difference = 1e-3
collect_folder_name = "collected_data/"
store_folder_name = "stored_data/"
sample_rate = 1000
