import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px

# loading the data

covid_data = pd.read_excel('dataset.xlsx')

# creating placeholder data for bottom table
placeHolderData = {
    'attribute 1': ['mean', 'median', 'mode'],
    'attribute 2': ['mean', 'median', 'mode'],
    'attribute 3': ['mean', 'median', 'mode']
}
# using example data for example graph
df = px.data.tips()
fig = px.bar(df, x="day", y="total_bill", color="smoker", barmode="group", facet_col="sex",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],
                              "smoker": ["Yes", "No"],
                              "sex": ["Male", "Female"]})

placeHolderDf = pd.DataFrame(placeHolderData, columns=['attribute 1', 'attribute 2', 'attribute 3'])

# creating dash app
app = dash.Dash(__name__)

# contaoins the layout of the app
app.layout = html.Div([

    html.Div([  # header
        html.H1(children='JBI100 Visualization'),
        html.Div(children="Group 15 covid visualization tool")
    ], className='header'),

    html.Div([  # navigation buttons
        html.A(children="Labels"),
        html.A(children="Options"),
        html.A(children="Data"),
        html.A(children="Export"),
    ], className='topnav'),

    html.Div([  # row
        html.Div([  # left column
            dcc.Graph(id='the_graph', figure=fig)
        ], className="leftcolumn"),

        html.Div([  # right column
            html.Div([  # card
                html.Div([  # group check list
                    dcc.Checklist(
                        id='my_grouplist',  # used to identify component in callback
                        options=[  # create options for the group checklist
                            {'label': 'group 1', 'value': 'group 1'},
                            {'label': 'group 2', 'value': 'group 2'},
                            {'label': 'group 3', 'value': 'group 3'},
                            {'label': 'group 4', 'value': 'group 4'},
                            {'label': 'group 5', 'value': 'group 5'},
                            {'label': 'group 6', 'value': 'group 6'}
                        ],
                        className='my_box_containerGroup',  # class of the container (div)
                        inputClassName='my_box_input',  # class of the <input> checkbox element
                        labelClassName='my_box_label',
                        # class of the <label> that wraps the checkbox input and the option's label
                    ),
                ]),  # end group list

                html.Div([  # checklist attributes
                    dcc.Checklist(
                        id='my_checklist',  # used to identify component in callback
                        options=[  # using the covid data to create checklist with all attributes
                            {'label': x, 'value': x, 'disabled': False}
                            for x in covid_data.columns
                        ],
                        className='my_box_container',  # class of the container (div)
                        inputClassName='my_box_input',  # class of the <input> checkbox element
                        labelClassName='my_box_label',
                        # class of the <label> that wraps the checkbox input and the option's label
                    ),
                ]),  # end checklist div
            ], className="card"),  # end card div
        ], className='"rightcolumn"'),  # end right column div
    ], className='row'),

    html.Div([  # create table at the bottom in the footer of the page
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in placeHolderDf.columns],
            data=placeHolderDf.to_dict('records'),
        ),

    ], className='footer'),  # end footer div

])

# runs the dash app in debug mode.

if __name__ == '__main__':
    app.run_server(debug=True)
