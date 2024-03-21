import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# Define a function to load and preprocess the data
def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.dropna(inplace=True)
    return df


# Function to load and preprocess the dataset
def load_and_preprocess_dataset(url):
    df = pd.read_csv(url)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.dropna(inplace=True)
    return df


# Define a function for plotting data
def plot_data(df, future_df=None):
    # Create traces
    trace1 = go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Actual')
    traces = [trace1]

    # If future_df is provided, add its data to the plot
    if future_df is not None:
        trace2 = go.Scatter(x=future_df['timestamp'], y=future_df['predicted_close'], mode='lines',
                            name='Predicted', line=dict(color='red', dash='dash'))
        traces.append(trace2)

    layout = go.Layout(title='Bitcoin Closing Prices and Predictions', xaxis=dict(title='Date'),
                       yaxis=dict(title='Price (USD)'), hovermode='closest')

    fig = go.Figure(data=traces, layout=layout)

    # Display the figure
    st.plotly_chart(fig, use_container_width=True)


# Function to plot data using Plotly
def plot_data_with_plotly(df, future_df=None):
    trace1 = go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Actual')
    traces = [trace1]
    if future_df is not None:
        trace2 = go.Scatter(x=future_df['timestamp'], y=future_df['predicted_close'], mode='lines',
                            name='Predicted', line=dict(color='red'))
        traces.append(trace2)
    layout = go.Layout(title='Bitcoin Closing Prices and Predictions', xaxis=dict(title='Date'),
                       yaxis=dict(title='Price (USD)'), hovermode='closest')
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def plot_daily_data(df):
    trace1 = go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Actual')
    traces = [trace1]
    layout = go.Layout(title='Bitcoin Closing Prices and Predictions', xaxis=dict(title='Date'),
                       yaxis=dict(title='Price (USD)'), hovermode='closest')
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)