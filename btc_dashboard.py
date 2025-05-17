from typing import List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import datetime
from datetime import datetime, timedelta

def load_data() -> pd.DataFrame:
    """Load and prepare the BTC price data."""
    df = pd.read_csv("btc_usd_5y_hourly_kraken.csv", index_col="time", parse_dates=True)
    
    # Calculate technical indicators
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df['returns'] = df['close'].pct_change()
    
    return df

def create_candlestick_chart(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    show_sma20: bool = True,
    show_sma50: bool = True
) -> go.Figure:
    """Create an interactive candlestick chart with technical indicators."""
    # Filter data for the selected date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df.loc[mask]
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('BTC/USD Price', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=filtered_df.index,
            open=filtered_df['open'],
            high=filtered_df['high'],
            low=filtered_df['low'],
            close=filtered_df['close'],
            name='BTC/USD'
        ),
        row=1, col=1
    )
    
    # Add moving averages if enabled
    if show_sma20:
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df['SMA20'],
                name='SMA20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if show_sma50:
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df['SMA50'],
                name='SMA50',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Add volume bars
    colors = ['red' if row['close'] < row['open'] else 'green' 
              for _, row in filtered_df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=filtered_df.index,
            y=filtered_df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Interactive BTC/USD Price Chart',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the data
df = load_data()

# Define the app layout
app.layout = html.Div([
    html.H1("Bitcoin Price Analysis Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df.index[-30],
                end_date=df.index[-1],
                min_date_allowed=df.index[0],
                max_date_allowed=df.index[-1],
                style={'margin': '10px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Technical Indicators:"),
            dcc.Checklist(
                id='indicator-toggle',
                options=[
                    {'label': 'SMA20', 'value': 'SMA20'},
                    {'label': 'SMA50', 'value': 'SMA50'}
                ],
                value=['SMA20', 'SMA50'],
                style={'margin': '10px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),
        
        html.Div([
            html.Button("Download Chart", id="btn-download"),
            dcc.Download(id="download-chart")
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'})
    ], style={'margin': '20px'}),
    
    dcc.Graph(id='price-chart'),
    
    html.Div([
        html.H3("Chart Controls"),
        html.P("• Use the mouse wheel to zoom in/out"),
        html.P("• Click and drag to pan"),
        html.P("• Double click to reset the view"),
        html.P("• Hover over candlesticks for detailed information"),
    ], style={'margin': '20px', 'padding': '10px', 'backgroundColor': '#f8f9fa'})
])

@app.callback(
    Output('price-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('indicator-toggle', 'value')]
)
def update_chart(start_date: str, end_date: str, indicators: List[str]) -> go.Figure:
    """Update the chart based on user selections."""
    start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    return create_candlestick_chart(
        df=df,
        start_date=start,
        end_date=end,
        show_sma20='SMA20' in indicators,
        show_sma50='SMA50' in indicators
    )

@app.callback(
    Output("download-chart", "data"),
    Input("btn-download", "n_clicks"),
    State("price-chart", "figure"),
    prevent_initial_call=True
)
def download_chart(n_clicks: int, figure: Dict[str, Any]) -> Dict[str, Any]:
    """Download the current chart view as a PNG file."""
    if n_clicks is None:
        return dash.no_update
    
    fig = go.Figure(figure)
    return dcc.send_bytes(
        fig.to_image(format="png", engine="kaleido"),
        filename=f"btc_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

if __name__ == '__main__':
    app.run(debug=True) 