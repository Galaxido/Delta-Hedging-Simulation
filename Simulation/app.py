# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# Import Libraries
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import ssl


# Formula for Black-Scholes Call Price
def bs_call(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


# Formula for Black-Scholes Call Delta
def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# Function that obtains all tickers from S&P 500
def get_tickers():
    ssl._create_default_https_context = ssl._create_unverified_context
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df1 = table[0]
    df1.to_csv('S&P500-Info.csv')
    df1.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
    ticker = df1["Symbol"].sort_values().values
    return ticker


# Function that computes Option PnL, Hedging PnL and Total PnL
def PnL(data, volatility, log_moneyness):
    r = 0
    sigma = volatility

    # Let K = S*e^(-m) where m = log-moneyness
    K = data[0] * np.exp(-1 * log_moneyness)
    n = len(data)

    # Compute time-series given T where dt = daily
    df = pd.DataFrame((n - np.linspace(0, n, n)) / 252, columns=["Maturity"])

    # Input prices obtained from Yahoo Finance
    df["Stock Price"] = pd.DataFrame(data.values)

    # Compute Call price using BS and then Option PnL using cumulative returns
    df["Option Price"] = bs_call(df["Stock Price"], K, df["Maturity"], r, sigma)
    df["Option PnL"] = df["Option Price"].diff().cumsum()

    # Compute Delta using BS
    df["Delta"] = bs_delta(df["Stock Price"], K, df["Maturity"], r, sigma)

    # Since we rehedge daily, we obtain cash as proceeds which must grow at rate r (in our care r = 0)
    cash = np.zeros(len(data))
    cash[0] = df["Delta"].iloc[0] * df["Stock Price"].iloc[0]
    for i, val in enumerate(df["Delta"]):
        if i == len(data) - 1:
            break
        cash[i + 1] = cash[i] + (df["Delta"].iloc[i + 1] - df["Delta"].iloc[i]) * df["Stock Price"].iloc[i + 1]

    # Create Cash Proceeds Data Frame
    df["Cash"] = pd.DataFrame(cash)

    # Compute Hedge
    df["Hedge"] = -1 * df["Stock Price"] * df["Delta"] + df["Cash"]
    df["Hedge PnL"] = df["Hedge"].diff().cumsum()

    # Total portfolio consists of the original Call, Delta*Stock and Cash proceeds (Hedge)

    df["Total"] = df["Option Price"] + df["Hedge"]
    # Compute change in total PnL using cumulative returns
    df["Total PnL"] = df["Total"].diff().cumsum()

    # Set all PnL values at T = 0 to 1 -> We start off with 100%
    df["Option PnL"].iloc[0] = 0
    df["Hedge PnL"].iloc[0] = 0
    df["Total PnL"].iloc[0] = 0

    # Collect the dates of the Stock prices when we plot later
    df["Date"] = data.index

    return df[["Option PnL", "Hedge PnL", "Total PnL", "Date"]]


app = Dash(__name__)

# Obtain the Tickers
tickers = get_tickers()

# Set all possible parameters for period, volatility, and log-moneyness
period = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
vol_values = np.round(np.linspace(0.05, 0.5, 10), 2)
log_moneyness = np.round(np.linspace(-0.2, 0.2, 9), 2)

# Originally start with Microsoft, one year period, volatility = 0.2, and log-moneyness = 0 (ATM)
data = yf.download("MSFT", interval="1d", period="1y")["Adj Close"]

# Run unction that computes Option PnL, Hedging PnL and Total PnL
df2 = PnL(data, 0.2, 0)

# Plot the original PnL
fig = px.line(df2, x="Date", y=["Option PnL", "Hedge PnL", "Total PnL"], title='PnL Simulation')
fig.update_layout(

    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title={'text': "PnL Simulation", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
    legend_title="PnL",
    xaxis_title="Date",
    yaxis_title="PnL", )

app.layout = html.Div(children=[

    # Title
    html.H1(children='Delta Hedging Simulation',
            style={'textAlign': 'center', 'margin-top': '20px',
                   'margin-bottom': '0px', 'font-family': 'Arial, sans-serif'}),

    # Input for Ticker
    html.Div(children=[
        html.Br(),
        html.Label('Stock Ticker', style={'font-family': 'Arial, sans-serif'}),
        dcc.Dropdown(tickers, "MSFT", id='ticker', style={'font-family': 'Arial, sans-serif'}),

    ], style={'width': '48%', 'display': 'inline-block'}),

    # Input for Period
    html.Div(children=[
        html.Br(),
        html.Label('Period', style={'font-family': 'Arial, sans-serif'}),
        dcc.Dropdown(period, "1y", id='time', style={'font-family': 'Arial, sans-serif'}),

    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    # Output for Graph with the PnL
    html.Div(children=[
        dcc.Graph(id='graphic', figure=fig),

    ], style={'width': '100%', 'float': 'center', 'display': 'inline-block',
              'margin-top': '15px', 'margin-bottom': '-28px'}),

    html.Div(style={'textAlign': 'center'}, children=[

        # Slider for the Volatility
        html.Div(style={'width': '40%', 'display': 'inline-block'}, children=[
            html.H2(children='Volatility',
                    style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Slider(
                min=0.05,
                max=0.5,
                step=None,
                id='vol',
                value=0.2,
                marks={str(vol): str(vol) for vol in vol_values},
            ), ],
                 ),

        # Slider for the Log-Moneyness
        html.Div(style={'width': '40%', 'display': 'inline-block'}, children=[
            html.H2(children='Initial Log-Moneyness',
                    style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Slider(
                min=-0.2,
                max=0.2,
                step=None,
                id='logmoneyness',
                value=0,
                marks={str(moneyness): str(moneyness) for moneyness in log_moneyness},
            ), ],
                 ),
    ],
             ),
    # Outputs all the additional information
    html.Div(style={'textAlign': 'center', 'padding': '30px'}, children=[

        html.Div(id='stock-price-before', style={'padding': '10px', 'font-family': 'Arial, sans-serif'}),
        html.Div(id='strike', style={'padding': '10px', 'font-family': 'Arial, sans-serif'}),
        html.Div(id='option-price-before', style={'padding': '10px', 'font-family': 'Arial, sans-serif'}),
        html.Div(id='stock-price-after', style={'padding': '10px', 'font-family': 'Arial, sans-serif'}),
        html.Div(id='option-price-after', style={'padding': '10px', 'font-family': 'Arial, sans-serif'}),

    ],
             ),
])


# Inputs Ticker, Period, Volatility and Log-Moneyness and outputs PnL Graph
@app.callback(Output('graphic', 'figure'),
              Input('ticker', 'value'),
              Input('time', 'value'),
              Input('vol', 'value'),
              Input('logmoneyness', 'value'))
def update_graph(ticker, time, vol, logmoneyness):
    # Downloads data from Yahoo Finance
    data = yf.download(ticker, interval="1d", period=time)["Adj Close"]

    # Computes Option PnL, Hedge PnL, and Total PnL
    final_df = PnL(data, vol, logmoneyness)

    # Plot the new updated PnL
    fig = px.line(final_df, x="Date", y=["Option PnL", "Hedge PnL", "Total PnL"])
    fig.update_layout(

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title={'text': "PnL Simulation", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        legend_title="PnL",
        xaxis_title="Date",
        yaxis_title="PnL in Dollars ($)",
    )
    return fig


# Inputs Ticker, Period, Volatility and Log-Moneyness and outputs additional information
@app.callback(Output('stock-price-before', 'children'),
              Output('stock-price-after', 'children'),
              Output('option-price-before', 'children'),
              Output('option-price-after', 'children'),
              Output('strike', 'children'),
              Input('ticker', 'value'),
              Input('time', 'value'),
              Input('vol', 'value'),
              Input('logmoneyness', 'value'))
def callback_a(ticker, time, vol, logmoneyness):
    # Downloads data from Yahoo Finance
    data = yf.download(ticker, interval="1d", period=time)["Adj Close"]
    r = 0
    K = data[0] * np.exp(-1 * logmoneyness)
    T = len(data) / 252
    option_price = bs_call(data[0], K, T, r, vol)

    # Output some information
    return f"{ticker} price at beginning of simulation: {np.round(data[0], 2)}", \
           f"{ticker} price at end of simulation: {np.round(data[-1], 2)}", \
           f"Option price at beginning of simulation: {np.round(option_price, 2)}", \
           f"Option price at end of simulation: {np.round(max(data[-1] - K, 0), 2)}", \
           f"Strike price at beginning of simulation: {np.round(K, 2)}"


if __name__ == '__main__':
    app.run_server(debug=True)
