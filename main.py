import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly
import plotly.express as px
import nidaqmx
from nidaqmx.constants import Edge
from nidaqmx.constants import AcquisitionType
import multiprocessing
from multiprocessing import Process, Value, Array, Manager
import random
import json
import zmq
import webbrowser
import time
import datetime
import statistics
import pandas as pd
import os
from ctypes import c_char
from utils import *
from algorithm import calculate
from readData import read_data
from cal3group import *
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Connect to nidaqmx and add all the ports to task
task = nidaqmx.Task()
for port_name in port_list:
    task.ai_channels.add_ai_voltage_chan(port_name)

# manager = Manager()
# global_static_data = manager.Value(c_wchar_p, json.dumps({}))
global_static_data = Array('c', 100000)
global_static_data.value = bytes(json.dumps({}), 'utf-8')
calibration_complete = Value('d', 0)

# https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019ZWxSAM&l=en-US
# task.timing.cfg_samp_clk_timing(sample_rate)

# Create different figure with different layout 
# There are two modes: basic and group
def create_fig(mode):
    if mode == 0:
        specs = basic_specs_1
    elif mode == 1:
        specs = basic_specs_2
    elif mode == 2:
        specs = group_specs_1
    elif mode == 3:
        specs = group_specs_2
    fig = plotly.tools.make_subplots(rows=specs["num_of_rows"], cols=specs["num_of_cols"], subplot_titles=specs["subplot_titles"])
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 0, 't': 40
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    fig.update_layout(showlegend=False)
    fig.update_layout(legend_x=1, legend_y=0)

    # Initialize all the plots
    for i in range(num_of_ports):
        fig.append_trace({
            'name': specs["port_names"][i],
            'mode': 'lines+markers',
            'type': 'scatter',
            'line': {'color': specs['color'][i]}
        }, specs["position"][i][0], specs["position"][i][1])
    return fig
fig = create_fig(1)

def create_result_fig(mode):
    fig = None
    if mode == 0:
        fig = px.scatter_3d(x=points_x, y=points_y, z=points_z, color=points_color, height=1000)
    elif mode == 1:
        result_rows = 5
        result_cols = 2
        fig = plotly.tools.make_subplots(rows=result_rows, cols=result_cols, 
            specs = [[{'type': 'scene'}, {'type': 'scene'}], [{'type': 'scene'}, {'type': 'scene'}], [{'type': 'scene'}, 
            {'type': 'scene'}], [{'type': 'scene'}, {'type': 'scene'}], [{'type': 'scene'}, {'type': 'scene'}]])
        fig['layout']['margin'] = {
            'l': 0, 'r': 0, 'b': 0, 't': 0
        }
        fig.update_layout(showlegend=False)
        fig['layout']['height'] = 3000
        for i in range(result_rows):
            for j in range(result_cols):
                fig.append_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers'), i + 1, j + 1)
    return fig

# Different tabs
# Index tab
index_layout = html.Div(
    html.Div([
        html.H4('NI-DAQmx'),
        html.Div(id='live-update-text'),
        html.Div(
            [html.Button("Basic Mode", id='basic-button', n_clicks=0),
            html.Button("Group Mode", id='group-button', n_clicks=0),]
        ),
        html.Br(),
        html.Button(id='store-button', n_clicks=0),
        dcc.Graph(id='live-update-graph', style={"height":800}, figure=fig),
        html.Div(json.dumps([[] for k in range(num_of_ports)]), id='data_list', style={'display': 'none'}),
        html.Div([], id='number_list', style={'display': 'none'}),
        html.Div(1, id='figure_number', style={'display': 'none'}),
        dcc.Interval(
            id='interval-component',
            interval=500,
            n_intervals=0
        ),
    ])
)

# Collect tab
collect_layout = html.Div(
    html.Div([
        html.H4('NI-DAQmx'),
        html.Div([
            dcc.Slider(
                id='collect-time-slider',
                min=1,
                max=20,
                step=1,
                value=2,
            ),
            html.Button('Collect', id='collect-button', n_clicks=0),],
            style={'width': '300px'}
        ),
        html.Button('Calculate', id='calculate-button', n_clicks=0),
        html.Button('Calibrate Phase 1', id='calibrate-button-1', n_clicks=0),
        html.Button('Calibrate Phase 2', id='calibrate-button-2', n_clicks=0),
        html.Button('Output', id='output-button', n_clicks=0),
        html.Button('Clear', id='clear-button', n_clicks=0),
        html.Div(id='collect-counter'),
        dcc.Graph(id='collect-graph', style={"height":600}, figure=fig),
        html.Div(id='display-lists'),
    ])
)

result_layout = html.Div(
    html.Div([
        html.H4('NI-DAQmx'),
        html.Button(id='apply-button', n_clicks=0, style={'display': 'none'}),
        html.Div(
        	[html.Div(id='matrix-value', style={'float': 'left', 'width': '20%', 'height': '3000px'}),
        	html.Div(dcc.Graph(id='result-graph', style={"height": '3000px', 'width': '79%', 'float': 'right'}))], style={'width': '100%'}
        ),
    ])
)

# Initialize webpage
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div(
        [
            html.P(id='calibration-status', style={'float': 'right'}),
            html.Div(dcc.Slider(
                id='group-slider',
                min=0,
                max=1,
                step=None,
                marks={
                    0: 'Group 1',
                    1: 'Group 2',
                },
                value = 0,
            ), style = {'width': '200px'}),
        ],
    ),
    html.Br(),
    dcc.Tabs(id='tab-button', value='index-tab', children=[
        dcc.Tab(label='Display Data', value='index-tab'),
        dcc.Tab(label='Collect Data', value='collect-tab'),
        dcc.Tab(label='Result', value='result-tab'),
    ]),
    html.Div(index_layout, id='tab-content'),
    html.Div(json.dumps([]), id='mean-lists', style={'display': 'none'}),
    html.Div(json.dumps([]), id='collected-data', style={'display': 'none'}),
    html.Div([], id='display-lists-content', style={'display': 'none'}),
    html.Div(json.dumps([]), id='static-data', style={'display': 'none'}),
    html.Div("False", id='store-button-state', style={'display': 'none'}),
    html.Div(id='result-number', style={'display': 'none'}),
    html.Div("Apply Calibration", id='apply-state', style={'display': 'none'}),
    html.Div(id='calibration-type', style={'display': 'none'}),
])

# Switch tabs
@app.callback(Output('tab-content', 'children'),
              Input('tab-button', 'value'))
def render_content(tab):
    if tab == 'collect-tab':
        return collect_layout
    elif tab == 'result-tab':
        return result_layout
    else:
        return index_layout

# Continuously receive data from device
def get_data():
    number = []
    voltage_list = [[] for k in range(num_of_ports)]
    pub_context = zmq.Context()
    pub = pub_context.socket(zmq.PUB)
    pub.bind("tcp://*:%s" % port)
    start_time = time.time()
    i = 0
    while True:
        data = task.read()
        number.append(i)
        # number.append(time.time())
        for j in range(num_of_ports):
            # voltage_list[j].append(i + j)
            voltage_list[j].append(data[j])
        # time.sleep(0.3)
        cur_time = time.time()
        if cur_time - start_time >= publisher_interval:
            message = json.dumps({'x': number, 'y': voltage_list})
            pub.send_string("%d %s" % (int(topic), message))
            voltage_list = [[] for k in range(num_of_ports)]
            number = []
            start_time = time.time()
        i = i + 1

# Update graph continuously
sub_context = zmq.Context()
sub = sub_context.socket(zmq.SUB)
sub.connect("tcp://localhost:%s" % port)
sub.setsockopt_string(zmq.SUBSCRIBE, topic)
@app.callback(
    Output('live-update-graph', 'figure'),
    Output('data_list', 'children'),
    Output('number_list', 'children'),
    Output('figure_number', 'children'),
    Output('calibration-status', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('basic-button', 'n_clicks'),
    Input('group-button', 'n_clicks'),
    State('live-update-graph', 'figure'),
    State('data_list', 'children'),
    State('number_list', 'children'),
    State('figure_number', 'children'),
    State('static-data', 'children'),
    State('apply-state', 'children'),
    State('calibration-type', 'children'),
    State('group-slider', 'value'),
)
def update_graph_live(n, basic_clicks, group_clicks, figure, data_list, number_list, fig_num, static_data, 
    apply_state, calibration_type, group_slider):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # Use different figure layout with different mode
    data_list = json.loads(data_list)
    calibration_status = ""
    if calibration_complete.value == 1:
        calibration_status = "Calibration Complete"

    fig_change = True
    if (fig_num < 2 or 'basic-button' in changed_id) and fig_num != group_slider:
        fig_num = group_slider
    elif (fig_num >= 2 or 'group-button' in changed_id) and fig_num != group_slider + 2:
        fig_num = group_slider + 2
    else:
        fig_change = False
    if fig_change:
        figure = create_fig(fig_num)
        legend = False if fig_num <= 2 else True
        figure.update_layout(showlegend=legend)

    # Receive data
    received = False
    while True:
        try:
            message = sub.recv(flags=zmq.NOBLOCK)
            received = True
            decodedMessage = message.decode("utf-8")
            data = json.loads(decodedMessage[4 :])
            
            number_list.extend(data["x"])
            number_list = number_list[-limit:]

        except zmq.ZMQError as e:
            if e.errno != zmq.EAGAIN:
                raise
            break

    # Update Graph
    if received:
        for i in range(num_of_ports):
            data_list[i].extend(data["y"][i])
            data_list[i] = data_list[i][-limit:]
        data = data_list
        static_data = json.loads(global_static_data.value.decode('utf-8'))
        if apply_state == "Stop" and len(static_data) > 0:
            matrix_multiplication = matrix_multiplication_list[calibration_type]
            matrix_list = static_data['matrix']
            ones = np.array([1 for k in range(len(data_list[0]))])
            new_data_list = [[] for k in range(num_of_ports)]
            for i in range(len(matrix_list)):
                matrix = np.array(matrix_list[i])
                point = []
                for j in matrix_multiplication[i]:
                    data = np.array(data_list[j])
                    point.append(data)
                point.append(ones)
                point = np.array(point)
                new_point = np.matmul(matrix, point)
                l = 0
                for j in matrix_multiplication[i]:
                    new_data_list[j] = new_point[l].tolist()
                    l = l + 1
            data = new_data_list
            calibration_result_temp = global_static_data
        for i in range(num_of_ports):
            figure["data"][i]["x"] = number_list
            figure["data"][i]["y"] = data[i]
            xaxis_name = "xaxis"
            if i > 0 and fig_num != 3:
                xaxis_name = xaxis_name + str(i + 1)
            figure["layout"][xaxis_name]["range"] = [number_list[-1] - limit, number_list[-1]]
    data_list = json.dumps(data_list)
    return figure, data_list, number_list, fig_num, calibration_status

# Write data to file until killed, use read_data.py to transform into csv
def store_data_helper(finish):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:%s" % port)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    if not os.path.exists(stored_folder_name):
        os.makedirs(stored_folder_name)
    current_time = str(datetime.datetime.now())
    current_time = current_time.replace(" ", "_")
    current_time = current_time.replace(".", "_")
    current_time = current_time.replace(":", "-")
    stored_file_name = stored_folder_name + "stored_data_" + current_time + ".json"
    data_file = open(stored_file_name, "a")
    while finish.value == 0:
        print("Storing data")
        message = socket.recv()
        decodedMessage = message.decode("utf-8")
        data_file.write(decodedMessage[4 :] + "\n")
    data_file.close()
    read_data(stored_file_name)
    os.remove(stored_file_name)

finish = Value('d', 0)
# Use a new process to store data
@app.callback(
    Output('store-button', 'children'),
    Output('store-button-state', 'children'),
    Input('store-button', 'n_clicks'),
    State('store-button-state', 'children'), # Used for storing button state when switching tabs
    )
def store_data(n_clicks, store_state):
    if n_clicks > 0:
        if store_state == "True":
            if n_clicks != 0:
                finish.value = 1
            return "Store Data", "False"
        else:
            finish.value = 0
            store_process = Process(target=store_data_helper, args=(finish, ))
            store_process.start()
            return "Stop", "True"
    else:
        return ("Stop", "True") if store_state == "True" else ("Store Data", "False")

# Check if collected data are valid
def check_data(data_list):
    var_list_local = var_list
    for i in range(num_of_ports):
        if len(data_list[i]) > 0:
            difference = abs(statistics.stdev(data_list[i]) - var_list_local[i])
            if difference > std_difference:
                return port_list[i]
    return True

@app.callback(
    Output('collect-button', 'children'),
    Input('collect-time-slider', 'value')
    )
def update_collect_button(value):
    return 'Collect {} Seconds'.format(value)

# Collect data for static location calculation
@app.callback(
    Output('collect-graph', 'figure'),
    Output('collect-counter', 'children'),
    Output('mean-lists', 'children'),
    Output('display-lists-content', 'children'),
    Output('display-lists', 'children'),
    Output('collected-data', 'children'),
    Input('collect-button', 'n_clicks'),
    Input('output-button', 'n_clicks'),
    Input('clear-button', 'n_clicks'),
    State('collect-graph', 'figure'),
    State('mean-lists', 'children'),
    State('collected-data', 'children'),
    State('display-lists-content', 'children'),
    State('group-slider', 'value'),
    State('collect-time-slider', 'value'),
    )
def collect_data(collect_clicks, output_clicks, clear_clicks, figure, mean_lists, collected_data, display_lists_content, group_slider, collect_time):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    valid = False
    mean_lists = json.loads(mean_lists)
    data_list = json.loads(collected_data)
    if 'collect-button' in changed_id and collect_clicks > 0:
    	# Collect different number of channels
        data_list = [[] for k in range(num_of_ports + 2)]
        collect_context = zmq.Context()
        collect_socket = collect_context.socket(zmq.SUB)
        collect_socket.connect("tcp://localhost:%s" % port)
        collect_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        start_time = time.time()
        cur_time = start_time
        mean_list = []
        number_list = []
        group_number_list = []
        figure = create_fig(group_slider)
        print("collect " + str(collect_time) + " seconds")
        while cur_time - start_time < collect_time:
            message = collect_socket.recv()
            decodedMessage = message.decode("utf-8")
            data = json.loads(decodedMessage[4 :])
            number_list.extend(data["x"])

            for i in range(num_of_ports):
                j = i
                if group_slider == 1:
                    if i == replaced_channel[0]:
                        j = num_of_ports
                    elif i == replaced_channel[1]:
                        j = num_of_ports + 1
                data_list[j].extend(data["y"][i])
            cur_time = time.time()
        for i in range(num_of_ports):
            j = i
            if group_slider == 1:
                if i == replaced_channel[0]:
                    j = num_of_ports
                elif i == replaced_channel[1]:
                    j = num_of_ports + 1
            figure["data"][i]["x"] = number_list
            figure["data"][i]["y"] = data_list[j]
        check_result = check_data(data_list)
        if check_result:
            if group_slider == 0:
                for i in range(num_of_ports):
                    mean_list.append(round(statistics.mean(data_list[i]), 4))
                display_lists_content.append(html.P(str(len(data_list[0])) + " value collected: " + str(mean_list)))
                # mean_list.append(len(data_list[0]))
                mean_lists.append(mean_list)
            else:
                found = False
                for j in range(len(mean_lists)):
                    if len(mean_lists[j]) == num_of_ports:
                        mean_lists[j].extend([[], []])
                        for i in range(num_of_ports + 2):
                            if i not in replaced_channel:
                                mean_lists[j][i] = round(statistics.mean(data_list[i]), 4)
                        display_lists_content.append(html.P(str(len(data_list[0])) + " value collected: " + str(mean_lists[j])))
                        # mean_lists[j].append(len(data_list[0]))
                        found = True
                        break
                if not found:
                    display_lists_content.append(html.P("Collect more group 1"))
        else:
            display_lists_content.append(html.P(str(len(data_list[0])) + " value collected: " + check_result + " Invalid"))
    # Output collected data to csv
    elif 'output-button' in changed_id:
        if output_clicks > 0:
            print("Outputing data")
            current_time = str(datetime.datetime.now())
            current_time = current_time.replace(" ", "_")
            current_time = current_time.replace(".", "_")
            current_time = current_time.replace(":", "-")
            file_name = "collected_data_" + current_time
            df = pd.DataFrame(mean_lists)
            port_list_extra = basic_port_names_1.copy()
            if len(mean_lists[0]) == num_of_ports + num_of_extra:
                port_list_extra.extend(["5y", "5z"])
            # elif len(mean_lists[0]) == num_of_ports + 1:
            #   port_list_extra.append("Number of Group 1")
            df.columns = port_list_extra
            if not os.path.exists(collect_folder_name):
                os.makedirs(collect_folder_name)
            df.to_csv(collect_folder_name + file_name + ".csv")
            display_lists_content.append(html.P("Ouput Success"))
    # Clear collected data and the graph 
    elif 'clear-button' in changed_id:
        if clear_clicks > 0:
            mean_lists = []
            display_lists_content = []
            port_list_extra = []
            figure = create_fig(1)
    display_text = str(len(mean_lists)) + " groups collected"
    mean_lists = json.dumps(mean_lists)
    data_list = json.dumps(data_list)
    return figure, display_text, mean_lists, display_lists_content, display_lists_content, data_list

def calibrate(collected_data, type, global_static_data, calibration_complete):
    cols = [0, 3, 6, 9, 12]
    data = {}
    ref = None
    static_data = {}
    static_data["matrix"] = []
    static_data["old_data"] = []
    static_data["new_data"] = []
    data = {}
    calibration_complete.value = 0
    for i in range(len(cols)):
        data[i] = [[]]
        c = cols[i]
        row = []
        for j in range(len(collected_data[0])):
            if type == 1 and replaced_channel[0] in [c, c + 1, c + 2] and len(collected_data) > num_of_ports:
                data[i][0].append([collected_data[15][j], collected_data[16][j], collected_data[17][j], 1])
            else:
                data[i][0].append([collected_data[c][j], collected_data[c + 1][j], collected_data[c + 2][j], 1])
    if len(data[0]) > 0:
        estimate_center_radius(data)
        for i in range(len(data)):
            data_list, r, c = data[i]
            print('Calibrate data list {}'.format(i + 1))
            result = optimize(data_list, c, r)
            data[i].append(result)
            mat = generate_matrix(result)

            transformed, nr_list, _ = transform_data(mat, data_list, 1)
            newr = np.median(nr_list)
            if not ref:
                ref = newr
            scaling = ref / newr
            mat = generate_matrix(result, scaling)
            transformed, nr_list, or_list = transform_data(mat, data_list, scaling)
            # print('Mat =', mat)
            # print('new center = {}, old center = {}'.format(
            #   [0, 0, 0], (-mat[:, 3] / scaling).tolist()))
            # print('corrected r = {:.5f}, std = {:.5f}, old r = {:.5f}, std = {:.5f}'.format(
            #   np.median(nr_list), np.std(nr_list), np.median(or_list), np.std(or_list)))
            static_data["matrix"].append(mat.tolist())
            static_data["old_data"].append(data_list)
            static_data["new_data"].append(transformed)
        calibration_complete.value = 1
        global_static_data.value = bytes(json.dumps(static_data), 'utf-8')
    # return static_data


# Calculate use collected data
@app.callback(
    Output('tab-button', 'value'),
    Output('static-data', 'children'),
    Output('result-number', 'children'),
    Output('calibration-type', 'children'),
    Input('calculate-button', 'n_clicks'),
    Input('calibrate-button-1', 'n_clicks'),
    Input('calibrate-button-2', 'n_clicks'),
    Input('group-slider', 'value'),
    State('tab-button', 'value'),
    State('result-number', 'children'),
    State('mean-lists', 'children'),
    State('collected-data', 'children'))
def calclulate_action(calculate_clicks, calibrate_clicks_1, calibrate_clicks_2, group_slider, tab_value, result_number, mean_lists, collected_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    calibration_type = group_slider
    static_data = []
    if 'calculate-button' in changed_id and calculate_clicks > 0:
        mean_lists = json.loads(mean_lists)
        result_number = 0
        result_list = []
        static_data = [[] for k in range(4)]
        for mean_list in mean_lists:
            if len(mean_list) == num_of_ports + num_of_extra:
                result = calculate(mean_list[:num_of_ports + num_of_extra])
                result_list.append(result)
                static_data[0].extend(result[0])
                static_data[1].extend(result[1])
                static_data[2].extend(result[2])
                static_data[3].append(0.5)
        tab_value = 'result-tab'
    elif 'calibrate-button-1' in changed_id:
        collected_data = json.loads(collected_data)
        calibrate_process = Process(target=calibrate, args=(collected_data, group_slider, global_static_data, calibration_complete,))
        calibrate_process.start()
        result_number = 1
        tab_value = 'index-tab'
    static_data = json.dumps(static_data)
    return tab_value, static_data, result_number, calibration_type

# Update result graph
@app.callback(
    Output('matrix-value', 'children'),
    Output('result-graph', 'figure'),
    Output('apply-button', 'children'),
    Output('apply-button', 'style'),
    Output('apply-state', 'children'),
    Input('apply-button', 'n_clicks'),
    State('result-number', 'children'),
    State('static-data', 'children'),
    State('apply-state', 'children'),
    )
def update_result_graph(apply_clicks, result_number, data, apply_state):
    figure = create_result_fig(result_number)
    data = json.loads(data)
    matrix_value = []
    style = {'display': 'none'}
    if result_number == 0:
        figure["data"][0]["x"] = np.hstack([figure["data"][0]["x"], data[0]])
        figure["data"][0]["y"] = np.hstack([figure["data"][0]["y"], data[1]])
        figure["data"][0]["z"] = np.hstack([figure["data"][0]["z"], data[2]])
        figure["data"][0]["marker"]["color"] = np.hstack([figure["data"][0]["marker"]["color"], data[3]])

    elif result_number == 1:
        data = json.loads(global_static_data.value.decode('utf-8'))
        if len(data) > 0:
            matrix_data = data["matrix"]
            old_data = data["old_data"]
            new_data = data["new_data"]
            for i in range(len(old_data)):
                for j in range(len(old_data[i])):
                    figure["data"][2 * i]["x"] = figure["data"][2 * i]["x"] + (old_data[i][j][0],)
                    figure["data"][2 * i]["y"] = figure["data"][2 * i]["y"] + (old_data[i][j][1],)
                    figure["data"][2 * i]["z"] = figure["data"][2 * i]["z"] + (old_data[i][j][2],)
                    figure["data"][2 * i + 1]["x"] = figure["data"][2 * i + 1]["x"] + (new_data[i][j][0],)
                    figure["data"][2 * i + 1]["y"] = figure["data"][2 * i + 1]["y"] + (new_data[i][j][1],)
                    figure["data"][2 * i + 1]["z"] = figure["data"][2 * i + 1]["z"] + (new_data[i][j][2],)

            for matrix in matrix_data:
                matrix_list = []
                for matrix_row in matrix:
                    row = ""
                    for matrix_number in matrix_row:
                        row = row + str(round(matrix_number, 4)) + " "
                    matrix_list.append(html.P(row))
                matrix_div = html.Div(matrix_list, style={'height': '200px', 'margin-top': '200px', 'margin-bottom': '450px'})
                matrix_value.append(matrix_div)
            style = {'display': 'block'}

    if apply_clicks > 0:
        apply_state = "Stop" if apply_state == "Apply Calibration" else "Apply Calibration"
    return matrix_value, figure, apply_state, style, apply_state

def start_app():
    # app.run_server(debug=True)
    app.run_server()

if __name__ == '__main__':
    process = Process(target=start_app)
    process.start()
    # Open website automatically
    webbrowser.open('http://127.0.0.1:8050/', new=2)
    get_data()
