import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd


points = {'x':[1, 2, 3], 'y':[1, 2, 3], 'z':[1, 2, 3]}
df = pd.DataFrame(points)
num_of_points = 3
color = [1 for k in range(num_of_points)]
app = dash.Dash(__name__)
fig = px.scatter_3d(x=points['x'], y=points['y'], z=points['z'], color = color)
app.layout = html.Div([
    dcc.Graph(id="result-plot", figure=fig),
    html.Button(id="button", n_clicks=0)
])


@app.callback(
    Output("result-plot", "figure"),
    Input("button", "value"),
    State("result-plot", "figure"))
def update_bar_chart(slider_range, figure):
    # low, high = slider_range
    # mask = (df.petal_width > low) & (df.petal_width < high)
    x_list = [4, 5, 6, 7]
    # y_list = [1, 2, 3, 4]
    # z_list = [1, 2, 3, 4]
    print(figure)
    figure["data"][0]["x"].extend(x_list)
    figure["data"][0]["y"].extend(x_list)
    figure["data"][0]["z"].extend(x_list)
    # figure["data"][0]["hover_data"] = x_list
    # fig = px.scatter_3d(df[mask], 
    #     x='sepal_length', y='sepal_width', z='petal_width',
    #     color="species", hover_data=['petal_width'])
    return figure
app.run_server(debug=True)
