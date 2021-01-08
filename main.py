import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly
import plotly.express as px
# import nidaqmx
# from nidaqmx.constants import Edge
# from nidaqmx.constants import AcquisitionType
import multiprocessing
from multiprocessing import Process, Value, Array
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
# task = nidaqmx.Task()
# for port_name in port_list:
#     task.ai_channels.add_ai_voltage_chan(port_name)

# https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019ZWxSAM&l=en-US
# task.timing.cfg_samp_clk_timing(sample_rate)

# Create different figure with different layout 
# There are two modes: basic and group
def create_fig(mode):
    if mode == 1:
        specs = basic_specs
    elif mode == 2:
        specs = group_specs
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
            'name': port_list[i],
            'mode': 'lines+markers',
            'type': 'scatter',
            'line': {'color': specs['color'][i]}
        }, specs["position"][i][0], specs["position"][i][1])
    return fig
fig = create_fig(1)

def create_result_fig(mode):
    fig = None
    if mode == "1":
        fig = px.scatter_3d(x=points_x, y=points_y, z=points_z, color=points_color)
    elif mode == "2":
        result_rows = 5
        result_cols = 2
        fig = plotly.tools.make_subplots(rows=result_rows, cols=result_cols, 
            specs = [[{'type': 'scene'}, {'type': 'scene'}], [{'type': 'scene'}, {'type': 'scene'}], [{'type': 'scene'}, 
            {'type': 'scene'}], [{'type': 'scene'}, {'type': 'scene'}], [{'type': 'scene'}, {'type': 'scene'}]])
            # specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}], 
            # [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]])# , subplot_titles=specs["subplot_titles"]
        fig.update_layout(showlegend=False)
        for i in range(result_rows):
            for j in range(result_cols):
                fig.append_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers'), i + 1, j + 1)
                    # px.scatter_3d(x=points_x, y=points_y, z=points_z, color=points_color), i, j)
    return fig

# Different tabs
# Index tab
index_layout = html.Div(
    html.Div([
        html.H4('NI-DAQmx'),
        html.Div(id='live-update-text'),
        html.Button("Basic Mode", id='basic-button', n_clicks=0),
        html.Button("Group Mode", id='group-button', n_clicks=0),
        html.Br(),
        html.Button(id='store-button', n_clicks=0),
        dcc.Graph(id='live-update-graph', style={"height":800}, figure=fig),
        html.Div([[] for k in range(num_of_ports)], id='data_list', style={'display': 'none'}),
        html.Div([], id='number_list', style={'display': 'none'}),
        html.Div(1, id='figure_number', style={'display': 'none'}),
        html.Div([True for k in range(num_of_ports)], id='visibility', style={'display': 'none'}),
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
        html.Button('Collect Group 1', id='collect-button-1', n_clicks=0),
        html.Button('Collect Group 2', id='collect-button-2', n_clicks=0),
        html.Button('Calculate', id='calculate-button', n_clicks=0),
        html.Button('Calibrate 1', id='calibrate-button-1', n_clicks=0),
        html.Button('Calibrate 2', id='calibrate-button-2', n_clicks=0),
        html.Button('Output', id='output-button', n_clicks=0),
        html.Button('Clear', id='clear-button', n_clicks=0),
        html.Div(id='collect-counter'),
        html.Div([], id='mean-lists', style={'display': 'none'}),
        dcc.Graph(id='collect-graph', style={"height":600}, figure=fig),
        html.Div([], id='display-lists')
    ])
)

result_layout = html.Div(
    html.Div([
        html.H4('NI-DAQmx'),
        html.Button(id='button', n_clicks=0, style={'display': 'none'}),
        dcc.Graph(id='result-graph', style={"height":2000}),
    ])
)

# Initialize webpage
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Tabs(id='tab-button', value='index-tab', children=[
        dcc.Tab(label='Display Data', value='index-tab'),
        dcc.Tab(label='Collect Data', value='collect-tab'),
        dcc.Tab(label='Result', value='result-tab'),
    ]),
    html.Div(index_layout, id='tab-content'),
    # html.Div([[] for k in ]range(4), id='static-data', style={'display': 'none'}),
    html.Div({'matrix': [[[  1.,   0.,   0., -47.],
       [  0.,   1.,   0., -48.],
       [  0.,   0.,   1., -49.]], [[  1.03614139,   0.        ,   0.        , -47.        ],
       [  0.        ,   0.93631447,   0.        , -48.        ],
       [  0.        ,   0.        ,   0.83263147, -49.        ]], [[ 1.00592210e+00,  6.28143487e-11, -8.84690646e-12,
        -4.70000000e+01],
       [ 6.31166538e-11,  8.79464556e-01,  3.24569882e-11,
        -4.80000000e+01],
       [ 8.26513409e-11,  9.05096570e-11,  7.47265613e-01,
        -4.90000000e+01]], [[  0.96843576,   0.        ,   0.        , -47.        ],
       [  0.        ,   0.83012544,   0.        , -48.        ],
       [  0.        ,   0.        ,   0.68211759, -49.        ]], [[  0.62408346,   0.        ,   0.        , -47.        ],
       [  0.        ,   0.7667852 ,   0.        , -48.        ],
       [  0.        ,   0.        ,   0.94062555, -49.        ]], [[  1.01051319,   0.        ,   0.        , -53.91269746],
       [  0.        ,   0.63886962,   0.        , -55.05977613],
       [  0.        ,   0.        ,   0.62944696, -56.2068548 ]]], 
       'old_data': [[[34.5, 35.5, 36.5, 1], [59.5, 60.5, 61.5, 1]], [[37.5, 38.5, 39.5, 1], [62.5, 63.5, 64.5, 1]], [[40.5, 41.5, 42.5, 1], [65.5, 66.5, 67.5, 1]], [[43.5, 44.5, 45.5, 1], [68.5, 69.5, 70.5, 1]], [[46.5, 47.5, 48.5, 1], [71.5, 72.5, 73.5, 1]], [[49.5, 63.5, 64.5, 1], [74.5, 83.5, 84.5, 1]]], 
       'new_data': [[[-12.5, -12.5, -12.5], [12.5, 12.5, 12.5]], [[-8.144697754524763, -11.951892890338932, -16.111056912111785], [17.758837075792073, 11.455968869181241, 4.70472985237442]], [[-6.2601550422136185, -11.502220914218626, -17.241211468106265], [18.88789740078952, 10.484392996600413, 1.4404288493655955]], [[-4.87304431318578, -11.059417810789817, -17.9636498743466], [19.337849759695956, 9.693718250564217, -0.9107102448667064]], [[-17.980119207617452, -11.577702949889897, -3.379660882984666], [-2.3780327600999556, 7.591927076483842, 20.135977837126333]], [[-3.892294490878065, -14.491555234311669, -15.607526084782755], [21.370535291295646, -1.7141628265563469, -3.0185869488122563]]]
       }, id='static-data', style={'display': 'none'}),
    html.Div("False", id='store-button-state', style={'display': 'none'}),
    html.Div("2", id='result-number', style={'display': 'none'}),
])

# Switch tabs
@app.callback(Output('tab-content', 'children'),
              Input('tab-button', 'value'))
def render_content(tab):
    # if tab == 'index-tab':
    #     return index_layout
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
        # data = task.read()
        number.append(i)
        # number.append(time.time())
        for j in range(num_of_ports):
            voltage_list[j].append(i + j)
            # voltage_list[j].append(data[j])
        time.sleep(0.3)
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
    Input('interval-component', 'n_intervals'),
    Input('basic-button', 'n_clicks'),
    Input('group-button', 'n_clicks'),
    State('live-update-graph', 'figure'),
    State('data_list', 'children'),
    State('number_list', 'children'),
    State('figure_number', 'children'),
)
def update_graph_live(n, basic_button, group_button, figure, data_list, number_list, fig_num):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # Use different figure layout with different mode
    if 'button' in changed_id:
        legend = False
        if 'basic-button' in changed_id:
            fig_num = 1
            legend = False
        elif 'group-button' in changed_id:
            fig_num = 2
            legend = True
        figure = create_fig(fig_num)
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
            figure["data"][i]["x"] = number_list
            figure["data"][i]["y"] = data_list[i]
            xaxis_name = "xaxis"
            if i > 0 and fig_num != 3:
                xaxis_name = xaxis_name + str(i + 1)
            figure["layout"][xaxis_name]["range"] = [number_list[-1] - limit, number_list[-1]]
    return figure, data_list, number_list, fig_num

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
        if store_state == "True":
            return "Stop", "True"
        else:
            return "Store Data", "False"

# Check if collected data are valid
def check_data(data_list, num_of_ports):
    # var_list_local = var_list
    # if num_of_ports == num_of_extra:
    #     var_list_local = extra_var_list
    # for i in range(num_of_ports):
    #     difference = abs(statistics.stdev(data_list[i]) - var_list_local[i])
    #     if difference > std_difference:
    #         return port_list[i]
    return True

# Collect data for static location calculation
@app.callback(
    Output('collect-graph', 'figure'),
    Output('collect-counter', 'children'),
    Output('mean-lists', 'children'),
    Output('display-lists', 'children'),
    Input('collect-button-1', 'n_clicks'),
    Input('collect-button-2', 'n_clicks'),
    Input('output-button', 'n_clicks'),
    Input('clear-button', 'n_clicks'),
    State('collect-graph', 'figure'),
    State('mean-lists', 'children'),
    State('display-lists', 'children'),
    )
def collect_data(collect_1_clicks, collect_2_clicks, output_clicks, clear_clicks, figure, mean_lists, display_lists):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    valid = False
    if 'collect-button' in changed_id:
    	# Collect different number of channels
        if 'collect-button-1' in changed_id and collect_1_clicks > 0:
                num_of_ports_local = num_of_ports
                valid = True
        elif 'collect-button-2' in changed_id and collect_2_clicks > 0:
                num_of_ports_local = num_of_extra
                valid = True

        if valid:
            collect_context = zmq.Context()
            collect_socket = collect_context.socket(zmq.SUB)
            collect_socket.connect("tcp://localhost:%s" % port)
            collect_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            start_time = time.time()
            cur_time = start_time
            data_list = [[] for k in range(num_of_ports_local)]
            mean_list = []
            number_list = []
            while cur_time - start_time < collect_time_interval:
                message = collect_socket.recv()
                decodedMessage = message.decode("utf-8")
                data = json.loads(decodedMessage[4 :])
                number_list.extend(data["x"])
                for i in reversed(range(1, num_of_ports_local + 1)):
                    data_list[-i].extend(data["y"][-i])
                cur_time = time.time()
            figure = create_fig(1)
            for i in range(num_of_ports_local):
                figure["data"][i]["x"] = number_list
                figure["data"][i]["y"] = data_list[i]
            check_result = check_data(data_list, num_of_ports_local)
            if check_result == True:
                if 'collect-button-1' in changed_id:
                    for i in range(num_of_ports_local):
                        mean_list.append(round(statistics.mean(data_list[i]), 4))
                    display_lists.append(html.P(str(len(data_list[0])) + " value collected: " + str(mean_list)))
                    # mean_list.append(len(data_list[0]))
                    mean_lists.append(mean_list)
                else:
                    found = False
                    for j in range(len(mean_lists)):
                        if len(mean_lists[j]) == num_of_ports:
                            for i in range(num_of_ports_local):
                                mean_lists[j].append(round(statistics.mean(data_list[i]), 4))
                            display_lists.append(html.P(str(len(data_list[0])) + " value collected: " + str(mean_lists[j])))
                            # mean_lists[j].append(len(data_list[0]))
                            found = True
                            break
                    if not found:
                        display_lists.append(html.P("Collect more group 1"))
            else:
                display_lists.append(html.P(str(len(data_list[0])) + " value collected: " + check_result + " Invalid"))
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
            port_list_extra = port_list.copy()
            if len(mean_lists[0]) == num_of_ports + num_of_extra:
                port_list_extra.extend(["Dev1/ai24", "Dev1/ai25"])
            # elif len(mean_lists[0]) == num_of_ports + 1:
            #     port_list_extra.append("Number of Group 1")
            df.columns = port_list_extra
            if not os.path.exists(collect_folder_name):
                os.makedirs(collect_folder_name)
            df.to_csv(collect_folder_name + file_name + ".csv")
            display_lists.append(html.P("Ouput Success"))
    # Clear collected data and the graph 
    elif 'clear-button' in changed_id:
        if clear_clicks > 0:
            mean_lists = []
            display_lists = []
            port_list_extra = []
            figure = create_fig(1)
    display_text = str(len(mean_lists)) + " groups collected"
    return figure, display_text, mean_lists, display_lists

# Calculate use collected data
@app.callback(
    Output('tab-button', 'value'),
    Output('static-data', 'children'),
    Output('result-number', 'children'),
    Input('calculate-button', 'n_clicks'),
    Input('calibrate-button-1', 'n_clicks'),
    Input('calibrate-button-2', 'n_clicks'),
    State('mean-lists', 'children'),
    State('static-data', 'children'))
def calcluate_action(calculate_clicks, calibrate_clicks_1, calibrate_clicks_2, mean_lists, static_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    valid = False
    tab_value = 'collect-tab'
    result_number = "1"
    if 'calculate-button' in changed_id and calculate_clicks > 0:
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
        result_number = "1"
    elif 'calibrate-button-1' in changed_id and calibrate_clicks_1 > 0:
        cols = [0, 3, 6, 9, 12, 15]
        data = {}
        for i in range(len(cols)):
            data[i] = [[]]
        for mean_list in mean_lists:
            for i in range(len(cols)):
                c = cols[i]
                data[i][0].append(mean_list[c:c + 3] + [1])
        estimate_center_radius(data)
        
        ref = None
        print('\nCalibration phase 1')
        print('==========')
        static_data = {}
        static_data["matrix"] = []
        static_data["old_data"] = []
        static_data["new_data"] = []
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
            #     [0, 0, 0], (-mat[:, 3] / scaling).tolist()))
            # print('corrected r = {:.5f}, std = {:.5f}, old r = {:.5f}, std = {:.5f}'.format(
            #     np.median(nr_list), np.std(nr_list), np.median(or_list), np.std(or_list)))
            static_data["matrix"].append(mat)
            static_data["old_data"].append(data_list)
            static_data["new_data"].append(transformed)
        result_number = "2"
        tab_value = 'result-tab'
        print(static_data)
    return tab_value, static_data, result_number

# Update result graph
@app.callback(
    Output('result-graph', 'figure'),
    Input('button', 'n_clicks'),
    State('result-number', 'children'),
    State('static-data', 'children'),)
    # State('result-graph', 'figure'))
def update_result_graph(button_clicks, result_number, data):
    figure = create_result_fig(result_number)
    print(result_number)
    if result_number == "1":
        figure["data"][0]["x"].extend(data[0])
        figure["data"][0]["y"].extend(data[1])
        figure["data"][0]["z"].extend(data[2])
        figure["data"][0]["marker"]["color"].extend(data[3])

    elif result_number == "2":
        print(figure["data"])
        print(figure["data"][0])
        for i in range(5):
            matrix = data["matrix"]
            old_data = data["old_data"]
            new_data = data["new_data"]
            for j in range(len(old_data[i])):
                # figure["data"][2 * i]["x"].append(old_data[i][j][0])
                # figure["data"][2 * i]["y"].append(old_data[i][j][1])
                # figure["data"][2 * i]["z"].append(old_data[i][j][2])
                # figure["data"][2 * i + 1]["x"].append(new_data[i][j][0])
                # figure["data"][2 * i + 1]["y"].append(new_data[i][j][1])
                # figure["data"][2 * i + 1]["z"].append(new_data[i][j][2])
                print(i)
                print(j)
                figure["data"][2 * i]["x"] = figure["data"][2 * i]["x"] + (old_data[i][j][0],)
                figure["data"][2 * i]["y"] = figure["data"][2 * i]["y"] + (old_data[i][j][1],)
                figure["data"][2 * i]["z"] = figure["data"][2 * i]["z"] + (old_data[i][j][2],)
                figure["data"][2 * i + 1]["x"] = figure["data"][2 * i + 1]["x"] + (new_data[i][j][0],)
                figure["data"][2 * i + 1]["y"] = figure["data"][2 * i + 1]["y"] + (new_data[i][j][1],)
                figure["data"][2 * i + 1]["z"] = figure["data"][2 * i + 1]["z"] + (new_data[i][j][2],)

                # figure["data"][2 * i]["marker"]["color"].extend(i)
                # figure["data"][2 * i + 1]["marker"]["color"].extend(i)
    return figure

# @app.callback(
#     Output('tab-button', 'value'),
    
#     State('mean-lists', 'children'),
#     State('tab-button', 'value'),
#     )
# def calibrate_action(cali_clicks_1, cali_clicks_2, mean_lists, tab_value):


def start_app():
    # app.run_server(debug=True)
    app.run_server()

if __name__ == '__main__':
    process = Process(target=start_app)
    process.start()
    
    # Open website automatically
    webbrowser.open('http://127.0.0.1:8050/', new=2)
    get_data()
