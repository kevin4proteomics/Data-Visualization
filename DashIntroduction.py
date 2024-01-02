from dash import Dash, html,dcc, Input, Output, State, dash_table
import mpld3
import dash_bootstrap_components as dbc
import seaborn as sns
import pandas as pd
import dash
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.resources import CDN
import io
import base64
import dash_bio as dashbio

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')
df_head = df.head(5)
for c in df.columns:
    print(c)

ave_Price = df_head['Price'].mean()
ave_GDP = df_head['GDP'].mean()
ave_Growth_Rate = df_head['Growth_Rate'].mean()
# scatter_fig = px.scatter(data_frame=df,
#                                 x='Vehicle_Type',
#                                 y='Price')

city_list = df['City'].unique().tolist()
city_dict = dict(zip(city_list, city_list))
print(city_dict)
x=[1,2,3,4,5]
# df = pd.DataFrame(x, columns=['x'])
# g=sns.lineplot(data=df, x='x', y='x')

navBar = dbc.NavbarSimple(
    brand="Dash Introduction",
    children=[
        html.Img(src="https://pngtree.com/free-png-vectors/cat",
                 height=20),
        html.A('Cat',
               href="https://pngtree.com/free")
    ],
)

cards=dbc.Row([
    dbc.Col(
        dbc.Card([
            html.H4('Avg. Price'),
            html.H5(f'{round(ave_Price,2)} dollars')
        ],
        body=True,
        style={'textAlign': 'center',
               'color':'white',},
        color='red')
    ),
    dbc.Col(
        dbc.Card([
            html.H4('Avg. GDP'),
            html.H5(f'{round(ave_GDP, 2)} dollars')
        ],
            body=True,
            style={'textAlign': 'center',
                   'color': 'white', },
            color='red')
    ),
    dbc.Col(
        dbc.Card([
            html.H4('Avg. Growth Rate'),
            html.H5(f'{round(ave_Growth_Rate, 2)} dollars')
        ],
            body=True,
            style={'textAlign': 'center',
                   'color': 'white', },
            color='red')
    )
])

app=Dash(__name__,
         external_stylesheets=[dbc.themes.SOLAR])


app.layout=html.Div(id='H',

    children=[
    navBar,
    html.H1('Dash Introduction',
            style={'textAlign': 'left',
                   'color': 'white'}),
        html.Br(),
        cards,
    html.Br(),
    html.Div([
        html.Strong('Vehicle Type' ),
        html.Br(),
        dcc.RadioItems(id='vehicle_id',
                        options=df['Vehicle_Type'].unique().tolist(),
                       style={'width': '100%', 'display': 'inline'},
                       labelStyle={'display': 'inline-block', 'width': '100', 'margin-right': '20px', 'margin-left': ''}
                       ),
        html.Br(),
        html.Strong('State'),

        dcc.Checklist(id='city_checklist',
                      options=city_dict,
                      labelStyle={'display': 'inline-block', 'width': '100', 'margin-right': '20px', 'margin-left': ''})
                 ]),
    dcc.RangeSlider(
        id='year_slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=[1980, 2000],
        marks={str(year): str(year)
               for year in range(df['Year'].min(), df['Year'].max()+1,10)}

    ),
    dbc.Row(
        [
        dbc.Col(html.Div(id='scatterplots')),
        dbc.Col(html.Div(id='scatterplots_px')),
        ]),

    html.P(['Dash Introduction',
            html.Br(),
            html.A('SJCC Link',
                   href='https://sjcc.edu',
                   target='_blank')
            ],
           style={'textAlign': 'left',
                  'color': 'gray'}),
    html.Div(id='state-price_map'),
    html.Br(),
    html.Div(id='table'),

    html.P(['Dash Introduction',
               html.Br()]),
    html.Button(id='refresh_button',
                n_clicks=0,
                children='Refresh',
                style={
                        'fontSize': "18px",  # controls the text size
                        'height': "50px",  # controls the height of the button
                        'width': "150px",  # controls the width of the button
                        'padding': "5px",  # adds space between the text and the border of the button
                        'borderRadius': 0.3,
                        'textAlign':'center',
                        'backgroundColor': 'red',
                        'color': 'white'

                    }
    ),
    html.Br(),

    # dcc.Dropdown(id='city_dropdown',
    #              options=city_dict,
    #              searchable=True),
    html.Br(),
             # className='chart-grid')
    dcc.Input(id='input_text',
              value='Change this text',
              type='text',
              style={'backgroundColor': 'white',
                     'color': 'gray'}),
    html.Div(id='output_text',
             children='')
])

#call back function
@app.callback(
    [Output('scatterplots', 'children'),
     Output('scatterplots_px', 'children'),
     Output('table', 'children'),
     Output('state-price_map', 'children')],
    [Input('refresh_button', 'n_clicks'),
     Input('year_slider', 'value'),
     State('city_checklist', 'value'),
     State('vehicle_id', 'value')]
)


def update_graph(button_click,year,selected_city, selected_vehicle):

    df = pd.read_csv(
        'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')

    title = ''
    if selected_city !=None:
        df = df[df['City'].isin( selected_city)]
        title += f'City: {selected_city}, '
    else:
        title += 'City: All, '

    if selected_vehicle != None:
        df = df[df['Vehicle_Type'] == (selected_vehicle)]
        title += f'Vehicle: {selected_vehicle}'
    else:
        title +='Vehicle: All'

    df = df[(df['Year'] >= year[0]) & (df['Year'] <= year[1])]
    print(df.head(5))
    # fig, ax = plt.subplots()
    #
    # ax.scatter(data=df, x='Year', y='Price')
    # # ax = sns.scatter(data=df, x='Year', y='Price', hue='City')
    # ax.set_xlabel('Year')
    # ax.set_ylabel('Price')
    # ax.set_title(title)
    # ax.grid(color='lightgray', alpha=0.7)
    #
    # html_matplotlib = mpld3.fig_to_html(fig)
    fig_growth = go.Figure(data=go.Scatter(
        x=df['Year'],
        y=df['Growth_Rate'],
        mode='markers',
        opacity=0.5,
        marker=dict(
            size=10,
            color='blue'
        )))
    fig_growth.update_layout(
        title_text = title
        )
    fig = go.Figure(data=go.Scatter(
    x=df['Year'],
    y=df['Price'],
    mode='markers',
    opacity=0.5,
    marker=dict(
        size=10,
        color='red'
    )))
    fig.update_layout(title_text=title,
                      hovermode='closest')
    # g = dcc.Graph(
    #     figure = html_matplotlib)
    df_head = df.head(100)
    df_filtered = dash_table.DataTable(columns=[{"name": i, "id": i} for i in df.columns],
                        data=df_head.to_dict('records'),
                        editable=True,
                        style_cell={
                                'whiteSpace': 'normal'

                            },
                        sort_action='native',
                        page_size=10,
                          # dropdown={
                          #     'Month':{
                          #         'options':[
                          #             {'label':i, 'value': i} for i in df['Month'].unique()
                          #         ]
                          #     }
                          # }

                                       )

    #map
    state_code = {
        'California': 'CA',
        'Illinois': 'IL',
        'New York': 'NY',
        'Georgia': 'GA'
    }
    ave_Price_state = df.groupby('City')['Price'].mean().reset_index()
    ave_Price_state['City'] = ave_Price_state['City'].apply(
        lambda x: state_code.get(x, x)
    )

    map_fig = px.choropleth(
        ave_Price_state,
        locations='City',
        locationmode='USA-states',
        color='Price',
        scope='usa',
        color_continuous_scale='reds'
    )
    print(ave_Price_state)
    return (dcc.Graph(figure=fig),dcc.Graph(figure=fig_growth),
            df_filtered, dcc.Graph(figure=map_fig))
#

@app.callback(
    Output('output_text', 'children'),
    Input('input_text', 'value'))

def update_output(selected_text):
    return f'Text: {selected_text}'





if __name__ == '__main__':
    app.run_server(debug=True)