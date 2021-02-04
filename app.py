# importing libraries
import pandas as pd
import numpy as np
from operator import itemgetter
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# loading the data
covid_data = pd.read_excel('dataset.xlsx')
data_covid = pd.read_excel('dataset.xlsx')
colorscales = px.colors.named_colorscales()

for column in covid_data:
    if pd.api.types.is_numeric_dtype(covid_data[column]) and column != 'Patient ID':
        covid_data[column] = covid_data.groupby(
            ["Patient age quantile", "SARS-Cov-2 exam result", "Patient addmited to regular ward (1=yes, 0=no)",
             "Patient addmited to semi-intensive unit (1=yes, 0=no)",
             "Patient addmited to intensive care unit (1=yes, 0=no)"], as_index=False)[column].transform(
            lambda x: x.fillna(x.mean()))
covid_data = covid_data.fillna(0)

kwrds = ['not_detected', 'detected', 'absent', 'present', 'negative', 'positive']
for i in range(6):
    covid_data.replace(kwrds[i], int(i % 2), inplace=True)

con_attribute_list = [{'label': column, 'value': column, 'disabled': False} for column in covid_data if
                      not covid_data[column].isin([0, 1]).all()]
bin_attribute_list = [{'label': column, 'value': column, 'disabled': False} for column in covid_data if
                      covid_data[column].isin([0, 1]).all()]

# creating dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])


# contains the layout of the app
app.layout = html.Div([

    html.Div([  # header
        html.H1(children='JBI100 Visualization'),
        html.Div(children="Group 15 covid visualization tool")
    ], className='header'),

    html.Div([  # row

        dbc.Card(  # graph card
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([  # first graph
                        html.Div([
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Color Scale"),
                                    dcc.Dropdown(
                                        id='colorscale',
                                        options=[{"value": x, "label": x}
                                                 for x in colorscales],
                                        value='viridis'
                                    ),
                                    dcc.Graph(
                                        id='first_graph'
                                    )
                                ]), color='dark'
                            ),
                        ])
                    ], width=6, style={'height': '100%'}),
                    dbc.Col([  # second graph
                        html.Div([
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='second_graph'
                                    )
                                ]), color='dark'
                            ),
                        ])
                    ], width=6),
                ], align='center'),

                html.Br(),

                dbc.Row([
                    dbc.Col([  # third graph
                        html.Div([
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='third_graph'
                                    )
                                ]), color='dark'
                            ),
                        ])
                    ], width=6),
                    dbc.Col([  # fourth graph
                        html.Div([
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='fourth_graph'
                                    ),
                                    dcc.RangeSlider(
                                        id='my-range-slider',
                                        min=-5,
                                        max=5,
                                        step=0.1,
                                        value=[-1, 1],
                                        allowCross=False,
                                        marks={
                                            -1: {'label': '-1'},
                                            1: {'label': '1'}
                                        },
                                        tooltip={'always_visible': False, 'placement': 'bottom'}
                                    ),
                                ]), color='dark'
                            ),
                        ])
                    ], width=6),
                ], align='center'),

                html.Br(),

                dbc.Row([
                    dbc.Col([  # details table
                        html.Div([
                            dbc.Card(
                                dbc.CardBody([
                                    dash_table.DataTable(
                                        id='table',
                                        columns=[{"name": '', "id": ''}],
                                        data=[{}],
                                        style_cell={'color': 'black'}
                                    )
                                ]), color='dark'
                            ),
                        ])
                    ], width=12),
                ], align='center'),
            ]), color='dark'
        )
    ], style={'padding': '0px 0px 0px 0px'}, className="leftcolumn"),

    html.Div([  # right column
        dbc.Card(
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab(
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Checklist(  # checklist groups
                                    id='my_grouplist1',  # used to identify component in callback
                                    options=[  # create options for the group checklist
                                        {'label': 'Patient age quantile', 'value': 'Patient age quantile'},
                                        {'label': 'SARS-Cov-2 exam result', 'value': 'SARS-Cov-2 exam result'},
                                        {'label': 'Patient addmited to regular ward (1=yes, 0=no)',
                                         'value': 'Patient addmited to regular ward (1=yes, 0=no)'},
                                        {'label': 'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                                         'value': 'Patient addmited to semi-intensive unit (1=yes, 0=no)'},
                                        {'label': 'Patient addmited to intensive care unit (1=yes, 0=no)',
                                         'value': 'Patient addmited to intensive care unit (1=yes, 0=no)'}
                                    ],
                                    className='my_box_containerGroup',  # class of the container (div)
                                    inputClassName='my_box_input',  # class of the <input> checkbox element
                                    labelClassName='my_box_label',
                                    # class of the <label> that wraps the checkbox input and the option's label
                                ),

                                dcc.Checklist(  # checklist attributes
                                    id='my_checklist1',  # used to identify component in callback
                                    options=[  # using the covid data to create checklist with all attributes
                                        x for x in con_attribute_list
                                    ],
                                    className='my_box_container',  # class of the container (div)
                                    inputClassName='my_box_input',  # class of the <input> checkbox element
                                    labelClassName='my_box_label',
                                    # class of the <label> that wraps the checkbox input and the option's label
                                ),
                            ]), color='dark'  # end group list
                        ), label='1'
                    ),

                    dbc.Tab(
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Checklist(  # checklist attributes
                                    id='my_checklist2',  # used to identify component in callback
                                    options=[  # using the covid data to create checklist with all attributes
                                        x for x in bin_attribute_list
                                    ],
                                    className='my_box_container',  # class of the container (div)
                                    inputClassName='my_box_input',  # class of the <input> checkbox element
                                    labelClassName='my_box_label',
                                    # class of the <label> that wraps the checkbox input and the option's label
                                ),
                            ]), color='dark'  # end group list
                        ), label='2'
                    ),

                    dbc.Tab(
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Checklist(  # checklist groups
                                    id='my_grouplist3',  # used to identify component in callback
                                    options=[  # create options for the group checklist
                                        {'label': 'SARS-Cov-2 exam result', 'value': 'SARS-Cov-2 exam result'},
                                        {'label': 'Patient addmited to regular ward (1=yes, 0=no)',
                                         'value': 'Patient addmited to regular ward (1=yes, 0=no)'},
                                        {'label': 'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                                         'value': 'Patient addmited to semi-intensive unit (1=yes, 0=no)'},
                                        {'label': 'Patient addmited to intensive care unit (1=yes, 0=no)',
                                         'value': 'Patient addmited to intensive care unit (1=yes, 0=no)'}
                                    ],
                                    className='my_box_containerGroup',  # class of the container (div)
                                    inputClassName='my_box_input',  # class of the <input> checkbox element
                                    labelClassName='my_box_label',
                                    # class of the <label> that wraps the checkbox input and the option's label
                                ),

                                dcc.Checklist(  # checklist attributes
                                    id='my_checklist3',  # used to identify component in callback
                                    options=[  # using the covid data to create checklist with all attributes
                                        {'label': x, 'value': x, 'disabled': False} for x in covid_data.columns
                                    ],
                                    className='my_box_container',  # class of the container (div)
                                    inputClassName='my_box_input',  # class of the <input> checkbox element
                                    labelClassName='my_box_label',
                                    # class of the <label> that wraps the checkbox input and the option's label
                                ),
                            ]), color='dark'  # end group list
                        ), label='3'
                    ),

                    dbc.Tab([
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Checklist(  # checklist groups
                                    id='my_grouplist4',  # used to identify component in callback
                                    options=[  # create options for the group checklist
                                        {'label': 'Patient age quantile', 'value': 'Patient age quantile'},
                                        {'label': 'SARS-Cov-2 exam result', 'value': 'SARS-Cov-2 exam result'},
                                        {'label': 'Patient addmited to regular ward (1=yes, 0=no)',
                                         'value': 'Patient addmited to regular ward (1=yes, 0=no)'},
                                        {'label': 'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                                         'value': 'Patient addmited to semi-intensive unit (1=yes, 0=no)'},
                                        {'label': 'Patient addmited to intensive care unit (1=yes, 0=no)',
                                         'value': 'Patient addmited to intensive care unit (1=yes, 0=no)'}
                                    ],
                                    className='my_box_containerGroup',  # class of the container (div)
                                    inputClassName='my_box_input',  # class of the <input> checkbox element
                                    labelClassName='my_box_label',
                                    # class of the <label> that wraps the checkbox input and the option's label
                                ),

                                dcc.Checklist(  # checklist attributes
                                    id='my_checklist4',  # used to identify component in callback
                                    options=[  # using the covid data to create checklist with all attributes
                                        {'label': x, 'value': x, 'disabled': False} for x in covid_data.columns
                                    ],
                                    className='my_box_container',  # class of the container (div)
                                    inputClassName='my_box_input',  # class of the <input> checkbox element
                                    labelClassName='my_box_label',
                                    # class of the <label> that wraps the checkbox input and the option's label
                                ),
                            ]), color='dark'  # end group list
                        ),
                        html.Div([
                            dcc.Markdown(
                                'The pie chart displays the percentage of patients per age quantile that have an abnormal value for the selected measurement. The abnormality threshold is manually set by the user.'),
                        ], style={'padding': '10px 0px 0px 0px'})
                    ], label='4'
                    ),

                ])
            ]), color='dark'
        )
    ], style={'padding': '0px 0px 0px 0px', 'margin': 'auto'}, className="rightcolumn"),  # end right column div

])


@app.callback(
    [Output('table', 'columns'),
     Output('table', 'data'), ],

    [Input('first_graph', 'clickData'),
     Input('second_graph', 'clickData'),
     Input('third_graph', 'clickData'),
     Input('fourth_graph', 'clickData'),

     State('my_checklist1', 'value'),
     State('my_grouplist1', 'value'),
     State('my_checklist2', 'value'),
     State('my_checklist3', 'value'),
     State('my_grouplist3', 'value'),
     State('my_checklist4', 'value'),
     State('my_grouplist4', 'value')
     ],
    prevent_initial_call=True)
def show_details(click1, click2, click3, click4, attribute1, group1, attribute2, attribute3, group3, attribute4,
                 group4):
    ctx = dash.callback_context

    if ctx:
        point = ctx.triggered[0]['value']['points'][0]

        if ctx.triggered[0]['prop_id'] == 'first_graph.clickData':
            placeHolderDf = pd.DataFrame({
                'Patient ID': [data_covid['Patient ID'][point['pointNumber']]],
                group1[0]: [point['x']],
                attribute1[point['curveNumber']]: [point['y']],
                'Group mean': [
                    covid_data.groupby(['Patient age quantile'])[attribute1[point['curveNumber']]].mean()[point['x']]],
                'Group median': [
                    covid_data.groupby(['Patient age quantile'])[attribute1[point['curveNumber']]].median()[
                        point['x']]],
                'Group mode': [
                    covid_data.groupby(['Patient age quantile'])[attribute1[point['curveNumber']]].agg(pd.Series.mode)[
                        point['x']]]},

                columns=['Patient ID', group1[0], attribute1[point['curveNumber']], 'Group mean', 'Group median',
                         'Group mode'])
            return [{"name": i, "id": i} for i in placeHolderDf.columns], placeHolderDf.to_dict('records')

        elif ctx.triggered[0]['prop_id'] == 'second_graph.clickData':

            placeHolderDf = pd.DataFrame({
                'Attribute': [attribute2[0]],
                'Result': ['Negative', 'Positive'][point['curveNumber']],
                'Amount': [point['y']]},

                columns=['Attribute', 'Result', 'Amount'])
            return [{"name": i, "id": i} for i in placeHolderDf.columns], placeHolderDf.to_dict('records')

        elif ctx.triggered[0]['prop_id'] == 'third_graph.clickData':

            placeHolderDf = pd.DataFrame({
                'Patient ID': [data_covid['Patient ID'][point['pointNumber']]],
                'Attribute': [attribute3[point['curveNumber']]],
                attribute3[point['curveNumber']]: [point['y']],
                'Mean': [covid_data[point['x']].mean()],
                'Median': [covid_data[point['x']].median()],
                'Mode': [covid_data[point['x']].mode()]},

                columns=['Patient ID', 'Attribute', attribute3[point['curveNumber']], 'Mean', 'Median', 'Mode'])
            return [{"name": i, "id": i} for i in placeHolderDf.columns], placeHolderDf.to_dict('records')

        elif ctx.triggered[0]['prop_id'] == 'fourth_graph.clickData':

            placeHolderDf = pd.DataFrame({
                'Age quantile': [point['label']], 'Number of people': [point['value']],
                'Percentage': [point['percent'] * 100]},
                columns=['Age quantile', 'Number of people', 'Percentage'])

            return [{"name": i, "id": i} for i in placeHolderDf.columns], placeHolderDf.to_dict('records')


    else:
        return None, None



def make_scatter(indexlist, selectedpoints, my_checklist, group, selected, scale):
    fig = make_subplots(specs=[[{"secondary_y": False, 'type': 'scatter'}]])
    if my_checklist and group:
        if len(my_checklist) == 1:
            for attribute in my_checklist:
                fig.add_trace(go.Scatter(
                    name=attribute,
                    x=covid_data[group[0]],
                    y=covid_data[attribute],

                    marker=dict(
                        color=covid_data[attribute],
                        opacity=0.4,
                        colorbar=dict(title='Colorbar'),
                        colorscale=scale),

                    mode='markers',
                    hovertemplate='%s: ' % group[0] + '%{x}' + '<br>%s: ' % attribute + '%{y}' + '<extra></extra>')
                )
                fig.update_layout(
                    xaxis_title="{}".format(group[0]),
                    yaxis_title="{}".format(attribute)
                )
        else:
            for attribute in my_checklist:
                fig.add_trace(go.Scatter(
                    name=attribute,
                    x=covid_data[group[0]],
                    y=covid_data[attribute],
                    mode='markers',
                    hovertemplate='%s: ' % group[0] + '%{x}' + '<br>%s: ' % attribute + '%{y}' + '<extra></extra>')
                )
                fig.update_layout(
                    xaxis_title="{}".format(group[0]),
                    yaxis_title="{}".format(attribute)
                )





    fig.update_layout(margin={'t': 30, 'b': 0, 'r': 0, 'l': 0, 'pad': 0})
    fig.update_layout(hovermode='closest')
    fig.update_layout(clickmode='event+select', dragmode='select')

    fig.update_layout(title='Scatterplot')
    fig.update_layout(showlegend=True, legend=dict(orientation='h'))
    fig.update_layout(uirevision='constant')
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )

    if selected:
        fig.update_traces(customdata=indexlist,
                          marker={'opacity': 1}, selectedpoints=selectedpoints,
                          unselected={'marker': {'color': 'rgba(189, 195, 199, 0.3)'}})

    return (fig)
    # runs the dash app in debug mode.



def make_bar(indexlist, selectedpoints, my_checklist, selected):
    fig = make_subplots(specs=[[{"secondary_y": False, 'type': 'bar'}]])

    if my_checklist:
        binary_grouped = covid_data[['Patient age quantile', my_checklist[0]]]

        result = binary_grouped.groupby(
            'Patient age quantile').agg({my_checklist[0]: ['sum', 'count']})

        result = result[my_checklist[0]]
        result.rename(columns={'sum': 'positive'}, inplace=True)
        result.rename(columns={'count': 'total'}, inplace=True)
        result['negative'] = result['total'] - result['positive']

        positive_values = result['positive']
        negative_values = result['negative']
        index = result.index

        fig.add_trace(go.Bar(
            name='Negative values',
            x=index,
            y=negative_values,
            hovertemplate='Patient age quantile: ' + '%{x}' + '<br>Amount: ' + '%{y}' + '<extra></extra>'

        ))
        fig.add_trace(go.Bar(
            name='Positive values',
            x=index,
            y=positive_values,
            hovertemplate='Patient age quantile: ' + '%{x}' + '<br>Amount: ' + '%{y}' + '<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title = "Patient age quantile",
            yaxis_title = "{}".format(my_checklist[0])
        )

    fig.update_layout(barmode='group')
    fig.update_layout(margin={'t': 30, 'b': 0, 'r': 0, 'l': 0, 'pad': 0})
    fig.update_layout(hovermode='closest')
    fig.update_layout(clickmode='event+select', dragmode='select')

    fig.update_layout(title='Grouped bar chart')
    fig.update_layout(showlegend=True, legend=dict(orientation='h'))
    fig.update_layout(uirevision='constant')
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )

    if selected:
        fig.update_traces(customdata=indexlist,
                          marker={'opacity': 1}, selectedpoints=selectedpoints,
                          unselected={'marker': {'color': 'rgba(189, 195, 199, 0.3)'}})

    return (fig)



def make_violin(indexlist, selectedpoints, my_checklist, group, selected):
    fig = make_subplots(specs=[[{"secondary_y": False, 'type': 'box'}]])

    if my_checklist and group:
        fig.data = []
        for attribute in my_checklist:
            fig.add_trace(go.Violin(
                name=attribute,
                x=covid_data[group[0]],
                y=covid_data[attribute],
                box_visible=True,
                hovertemplate='%s: ' % group[0] + '%{x}' + '<br>%s: ' % attribute + '%{y}' + '<extra></extra>')
            )

            fig.update_layout(
                xaxis_title = "{}".format(group[0]),
                yaxis_title = "{}".format(attribute)
            )

    elif my_checklist and not group:
        fig.data = []
        for attribute in my_checklist:
            fig.add_trace(go.Violin(
                name=attribute,
                y=covid_data[attribute],
                box_visible=True,
                hovertemplate='%s: ' % attribute + '%{y}' + '<extra></extra>')
            )

            fig.update_layout(
                yaxis_title = "{}".format(attribute)
            )

    if selected:
        fig.update_traces(customdata=indexlist,
                          marker={'opacity': 1}, selectedpoints=selectedpoints,
                          unselected={'marker': {'color': 'rgba(189, 195, 199, 0.3)'}})

    fig.update_layout(margin={'t': 30, 'b': 0, 'r': 0, 'l': 0, 'pad': 0})
    fig.update_layout(hovermode='closest')
    fig.update_layout(clickmode='event+select', dragmode='select')
    fig.update_layout(title='Violin plot')
    fig.update_layout(showlegend=True, legend=dict(orientation='h'))
    fig.update_layout(uirevision='constant')
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    return (fig)


@app.callback(
    Output("fourth_graph", "figure"),
    [Input("my_checklist4", "value"),
     Input("my-range-slider", "value")],
)
def make_pie(my_checklist, slider):
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "pie"}]])
    if my_checklist:
        for attribute in my_checklist:
            if attribute in map(itemgetter('label'), bin_attribute_list):
                fig.data = []
                binary_dict = dict()
                for value in covid_data[attribute]:
                    if value in binary_dict.keys():
                        binary_dict[value] += 1
                    else:
                        binary_dict[value] = 1

                fig.add_trace(go.Pie(
                    name=attribute,
                    labels=list(binary_dict.keys()),
                    values=list(binary_dict.values()),
                    hovertemplate='Patient age quantile: ' + '%{label}' + '<extra></extra>'),
                    row=1, col=1
                )
            else:
                fig.data = []

                new_df = covid_data[['Patient age quantile', attribute]]

                abnormal_values = {'normal': 0, 'abnormal': []}

                for value in new_df.iterrows():

                    if value[1][1] > slider[1] or value[1][1] < slider[0]:

                        abnormal_values['abnormal'].append([value[1][0], value[1][1]])

                    else:
                        abnormal_values['normal'] += 1

                abnormal_perc = dict()

                for i in abnormal_values['abnormal']:

                    if i[:][0] not in abnormal_perc.keys():

                        abnormal_perc[i[:][0]] = 1

                    else:
                        abnormal_perc[i[:][0]] += 1

                fig.add_trace(go.Pie(
                    name=attribute,
                    labels=list(abnormal_perc.keys()),
                    values=list(abnormal_perc.values()),
                    hovertemplate='Patient age quantile: ' + '%{label}' + '<extra></extra>'),
                    row=1, col=1
                )

    fig.update_layout(margin={'t': 30, 'b': 0, 'r': 0, 'l': 0, 'pad': 0})
    fig.update_layout(hovermode='closest')
    fig.update_layout(clickmode='event+select', dragmode='select')
    fig.update_layout(title='Pie chart')
    fig.update_layout(showlegend=True, legend=dict(x=1, y=0.85))
    fig.update_layout(uirevision='constant')
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    return (fig)


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


@app.callback(
    Output('first_graph', 'figure'),
    Output('second_graph', 'figure'),
    Output('third_graph', 'figure'),
    Input('first_graph', 'selectedData'),
    Input('second_graph', 'selectedData'),
    Input('third_graph', 'selectedData'),
    Input("my_checklist1", "value"),
    Input("my_grouplist1", "value"),
    Input("colorscale", "value"),
    Input("my_checklist2", "value"),
    Input("my_checklist3", "value"),
    Input("my_grouplist3", "value")

)
def callback(selection1, selection2, selection3, my_checklist1, my_grouplist1, scale, my_checklist2, my_checklist3, my_grouplist3):
    indexlist = []
    for i in covid_data.index:
        indexlist.append(i)
    selectedpoints = []

    selected = False
    for selected_data in [selection1, selection2, selection3]:
        if selected_data and selected_data['points']:

            for p in selected_data['points']:
                selectedpoints.append(p['pointNumber'])

            selected = True
            selectedpoints = intersection(indexlist, selectedpoints)

    return [make_scatter(indexlist, selectedpoints, my_checklist1, my_grouplist1, selected, scale),
            make_bar(indexlist, selectedpoints, my_checklist2, selected),
            make_violin(indexlist, selectedpoints, my_checklist3, my_grouplist3, selected),
            ]


if __name__ == '__main__':
    app.run_server(debug=True)
