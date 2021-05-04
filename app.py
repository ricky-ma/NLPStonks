import datetime
import math
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import yfinance as yf

from dash.dependencies import Input, Output
import plotly.express as px

from inference import load_headlines, predict


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

server = app.server

headlines_urls = load_headlines(22)
headlines = [x[0] for x in headlines_urls]
df = yf.Ticker('^DJI').history(period='max')
df['Date'] = df.index
most_recent = df["Close"][-1]
pred = predict(headlines)


# API Call to update news
def update_news():
    global pred, headlines_urls, headlines
    headlines_urls = load_headlines(22)
    headlines = [x[0] for x in headlines_urls]
    pred = predict(headlines)
    headlines_df = pd.DataFrame(data=headlines_urls, columns=['title', 'url'])
    max_rows = 22
    return html.Div(
        children=[
            html.P(className="p-news", children="Headlines"),
            html.P(
                className="p-news float-right",
                children="Last update : "
                + datetime.datetime.now().strftime("%H:%M:%S"),
            ),
            html.Table(
                className="table-news",
                children=[
                    html.Tr(
                        children=[
                            html.Td(
                                children=[
                                    html.A(
                                        className="td-link",
                                        children=headlines_df.iloc[i]["title"],
                                        href=headlines_df.iloc[i]["url"],
                                        target="_blank",
                                    )
                                ]
                            )
                        ]
                    )
                    for i in range(min(len(headlines_df), max_rows))
                ],
            ),
        ]
    )


# Callback to update news
@app.callback(Output("news", "children"), [Input("i_news", "n_intervals")])
def update_news_div(n):
    return update_news()


@app.callback(
    Output("time-series-chart", "figure"),
    [Input("i_news", "n_intervals")])
def display_time_series(n):
    fig = px.line(df, x='Date', y='Close')
    fig.update_xaxes(rangeslider_visible=True)
    return fig


# Display big numbers in readable format
def human_format(num):
    try:
        num = float(num)
        # If value is 0
        if num == 0:
            return 0
        # Else value is a number
        if num < 1000000:
            return num
        magnitude = int(math.log(num, 1000))
        mantissa = str(int(num / (1000 ** magnitude)))
        return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]
    except:
        return num


# Returns Top cell bar for header area
def get_top_bar_cell(cellTitle, cellValue):
    return html.Div(
        className="two-col",
        children=[
            html.P(className="p-top-bar", children=cellTitle),
            html.P(id=cellTitle, className="display-none", children=cellValue),
            html.P(children=human_format(cellValue)),
        ],
    )


# Returns HTML Top Bar for app layout
def get_top_bar(curr_closing, pred_closing, acc=0.956315):
    pred = 'UP' if pred_closing > curr_closing else 'DOWN'
    return [
        get_top_bar_cell("Today's Closing", curr_closing),
        get_top_bar_cell("Predicted Closing", pred_closing),
        get_top_bar_cell("Tomorrow's DJIA will go:", pred),
        get_top_bar_cell("Historical Accuracy", acc),
    ]


app.layout = html.Div(
    className='row',
    children=[
        # Interval component for graph updates
        dcc.Interval(id="i_news", interval=1 * 60000, n_intervals=0),

        # Left Panel Div
        html.Div(
            className="three columns div-left-panel",
            children=[
                # Div for Left Panel App Info
                html.Div(
                    className="div-info",
                    children=[
                        # html.Img(
                        #     className="logo", src=app.get_asset_url("dash-logo-new.png")
                        # ),
                        html.H2("NLP-STONKS", className="title-header"),
                        html.P(
                            """
                            This app continually queries the top news headlines from r/WorldNews to predict 
                            whether tomorrow's DJIA closing price will go up or down with ~96% accuracy. 
                            """
                        ),
                        # Div for News Headlines
                        html.Div(
                            className="div-news",
                            children=[html.Div(id="news", children=update_news())],
                        ),
                    ],
                ),
            ],
        ),
        # Right Panel Div
        html.Div(
            className="nine columns div-right-panel",
            children=[
                html.Div(
                    id="top_bar", className="row div-top-bar", children=get_top_bar(most_recent, pred)
                ),
                dcc.Graph(id="time-series-chart"),
            ],
        ),
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True)
