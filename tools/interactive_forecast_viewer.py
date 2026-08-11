"""
Interactive Forecast Viewer

This Streamlit app allows you to:
1. Load a forecasting model from MLflow
2. Select a timestamp for prediction
3. Visualize forecast results
4. Navigate through time using Next/Previous buttons

Usage:
    streamlit run interactive_forecast_viewer.py

Requirements:
    pip install streamlit plotly
"""

import streamlit as st
import mlflow
import pandas as pd
import pathlib
import json
import datetime
import pytz
from dateutil import parser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from amp.mlflow_utils import ForecasterWrapper
from amp.utils import floor_time

# Page configuration
st.set_page_config(
    page_title="VTT Automated Modelling Pipeline: Interactive Forecast Viewer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 VTT Automated Modelling Pipeline: Interactive Forecast Viewer")

# Sidebar configuration
st.sidebar.header("Configuration")

# ========== MODEL SECTION ==========
st.sidebar.subheader("🤖 Model")

# MLflow configuration
mlflow_uri = st.sidebar.text_input(
    "MLflow URI",
    value="http://localhost:5000"
)

project_name = st.sidebar.text_input("Project Name", value="CIASEM")
task_name = st.sidebar.text_input("Task Name", value="ele_dh_power_forecast")

model_name = f"{project_name}_{task_name}_model"

# Initialize session state for model versions
if 'model_versions' not in st.session_state:
    st.session_state.model_versions = [1]
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'history_dfs' not in st.session_state:
    st.session_state.history_dfs = {}

# Version management
st.sidebar.write("**Model Versions:**")
for idx, ver in enumerate(st.session_state.model_versions):
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        new_ver = st.number_input(
            f"Version {idx + 1}",
            value=ver,
            min_value=1,
            key=f"version_input_{idx}",
            label_visibility="collapsed"
        )
        st.session_state.model_versions[idx] = new_ver
    with col2:
        if len(st.session_state.model_versions) > 1:
            if st.button("🗑️", key=f"remove_ver_{idx}", help="Remove version"):
                st.session_state.model_versions.pop(idx)
                version_key = f"v{ver}"
                if version_key in st.session_state.models:
                    del st.session_state.models[version_key]
                if version_key in st.session_state.predictions:
                    del st.session_state.predictions[version_key]
                if version_key in st.session_state.history_dfs:
                    del st.session_state.history_dfs[version_key]
                st.rerun()

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("➕ Compare Version", use_container_width=True):
        max_ver = max(st.session_state.model_versions)
        st.session_state.model_versions.append(max_ver + 1)
        st.rerun()
with col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.model_versions = [1]
        st.session_state.models = {}
        st.session_state.predictions = {}
        st.session_state.history_dfs = {}
        st.rerun()

st.sidebar.divider()

# ========== DATA SECTION ==========
st.sidebar.subheader("📁 Data")

# Data file selection
import os
from pathlib import Path

# Get current working directory for default
default_path = str(Path.cwd() / "data" / "hoas_example_data.csv")

data_file = st.sidebar.text_input(
    "Data File Path",
    value=default_path,
    help="Enter the full path to your CSV file or drag & drop below"
)

# Also provide file uploader as alternative
st.sidebar.caption("Or upload a file:")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=['csv'],
    help="Upload a CSV file with timestamp column",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Save to temporary location and use that path
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        data_file = tmp_file.name
    st.sidebar.success(f"✓ Using uploaded file: {uploaded_file.name}")


# Initialize session state for data and timestamps
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'modified_data' not in st.session_state:
    st.session_state.modified_data = None
if 'current_timestamp' not in st.session_state:
    st.session_state.current_timestamp = None
if 'actual_df' not in st.session_state:
    st.session_state.actual_df = None
if 'forecast_start' not in st.session_state:
    st.session_state.forecast_start = None
if 'current_input_df' not in st.session_state:
    st.session_state.current_input_df = None


def load_model(version):
    """Load specific model version from MLflow"""
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Search for registered model
        registered_models = mlflow.search_registered_models(
            filter_string=f"name='{model_name}'"
        )
        
        if not registered_models:
            st.error(f"No registered model found with name: {model_name}")
            return None
        
        # Load specific version
        try:
            model_uri = f"models:/{model_name}/{version}"
            loaded_model = ForecasterWrapper.load_model_with_metadata(model_uri)
            st.sidebar.success(f"✓ Version {version} loaded")
            return loaded_model
        except Exception as e:
            st.sidebar.error(f"Error loading version {version}: {e}")
            return None
            
    except Exception as e:
        st.sidebar.error(f"Error accessing MLflow: {e}")
        return None


def load_data(file_path):
    """Load data from CSV"""
    try:
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        
        # Localize to UTC if not already timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC)
        
        st.sidebar.success(f"✓ Data loaded: {len(df)} rows")
        
        # Store original data
        st.session_state.original_data = df.copy()
        st.session_state.modified_data = None
        
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return None


def make_prediction(model, data, timestamp, version_key, store_input=False):
    """Make prediction for a given timestamp and model version"""
    try:
        # Get model metadata
        max_lag, max_future = model.input_window
        data_freq = model.metadata.get_model_info().metadata.get("data_freq")
        if data_freq:
            try:
                data_freq = json.loads(data_freq)
            except (json.JSONDecodeError, ValueError):
                pass  # Already a number
        else:
            # Infer from data
            data_freq = int(pd.Timedelta(data.index[1] - data.index[0]).total_seconds() / 60)
        
        inputs = model.inputs
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except (json.JSONDecodeError, ValueError):
                pass  # Keep as string if parsing fails
        feature_types = list(inputs.keys()) if isinstance(inputs, dict) else list(inputs)
        
        # Floor timestamp to data frequency
        now = floor_time(timestamp, data_freq)
        
        # Calculate input window
        input_start = now - datetime.timedelta(minutes=abs(max_lag) * data_freq)
        input_end = now + datetime.timedelta(minutes=abs(max_future) * data_freq)
        
        # Select input data (use modified data if available)
        if st.session_state.modified_data is not None:
            input_df = st.session_state.modified_data.loc[input_start:input_end, feature_types]
        else:
            input_df = data.loc[input_start:input_end, feature_types]
        
        # Store input data for modification panel
        if store_input:
            st.session_state.current_input_df = input_df.copy()
        
        # Make prediction
        prediction = model.predict(input_df, params={'output_mode': 'single'})
        
        # Get last 10 steps of history for visualization
        history_steps = 10
        history_start_idx = max(0, len(input_df) - history_steps - abs(max_future))
        history_df = input_df.iloc[history_start_idx:len(input_df) - abs(max_future)]
        
        # Get real/actual values for comparison (forecast period only)
        forecast_start = now
        forecast_end = now + datetime.timedelta(minutes=abs(max_future) * data_freq)
        try:
            actual_df = data.loc[forecast_start:forecast_end]
        except:
            actual_df = None
        
        return prediction, history_df, actual_df, forecast_start, data_freq
    except Exception as e:
        st.error(f"Error making prediction for {version_key}: {e}")
        return None, None, None, None, None


def plot_forecast_comparison(history_dfs, predictions, actual_df, models_dict, forecast_start=None):
    """Create interactive plot comparing multiple model versions"""
    if not predictions or all(p is None for p in predictions.values()):
        return None
    
    # Get output columns from first valid prediction
    first_pred = next(p for p in predictions.values() if p is not None)
    output_cols = list(first_pred.columns)
    
    n_plots = len(output_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=output_cols,
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}] for _ in range(n_plots)]
    )
    
    # Color scheme for model versions
    model_colors = {
        0: '#636EFA',  # Blue
        1: '#EF553B',  # Red
        2: '#00CC96',  # Green
        3: '#AB63FA',  # Purple
        4: '#FFA15A',  # Orange
    }
    
    for idx, output_col in enumerate(output_cols, start=1):
        # Plot history (only once, from first model)
        if history_dfs and len(history_dfs) > 0:
            first_version_key = list(history_dfs.keys())[0]
        else:
            first_version_key = None
        
        if first_version_key and first_version_key in history_dfs and history_dfs[first_version_key] is not None:
            history_df = history_dfs[first_version_key]
            if output_col in history_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=history_df.index,
                        y=history_df[output_col],
                        name='History',
                        mode='lines+markers',
                        line=dict(color='gray', width=2),
                        marker=dict(size=6),
                        legendgroup=f'legend_{idx}',
                        legendgrouptitle_text=output_col,
                        showlegend=True
                    ),
                    row=idx,
                    col=1
                )
        
        # Plot each model version's forecast
        for model_idx, (version_key, prediction) in enumerate(predictions.items()):
            if prediction is not None and output_col in prediction.columns:
                color = model_colors[model_idx % len(model_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=prediction.index,
                        y=prediction[output_col],
                        name=f'{version_key}',
                        mode='lines+markers',
                        line=dict(color=color, width=2, dash='dot'),
                        marker=dict(size=8, symbol='diamond'),
                        legendgroup=f'legend_{idx}',
                        showlegend=True
                    ),
                    row=idx,
                    col=1
                )
        
        # Plot actual values
        if actual_df is not None and output_col in actual_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=actual_df.index,
                    y=actual_df[output_col],
                    name='Actual',
                    mode='lines+markers',
                    line=dict(color='#FDB462', width=3),  # Orange color for visibility
                    marker=dict(size=7, symbol='square'),
                    legendgroup=f'legend_{idx}',
                    showlegend=True,
                    opacity=0.9
                ),
                row=idx,
                col=1
            )
    
    # Add vertical lines at forecast start time for all subplots
    shapes = []
    if forecast_start is not None:
        for idx in range(1, n_plots + 1):
            # Calculate y position for this subplot (in paper coordinates)
            y_domain_start = 1 - (idx / n_plots) - (0.1 / n_plots)  # Account for spacing
            y_domain_end = 1 - ((idx - 1) / n_plots) - (0.1 / n_plots)
            
            shapes.append(
                dict(
                    type="line",
                    xref=f"x{idx if idx > 1 else ''}",
                    yref="paper",
                    x0=forecast_start,
                    y0=y_domain_start,
                    x1=forecast_start,
                    y1=y_domain_end,
                    line=dict(
                        color="rgba(128, 128, 128, 0.8)",
                        width=2,
                        dash="dash",
                    ),
                )
            )
    
    # Calculate default zoom range
    first_history = next((h for h in history_dfs.values() if h is not None), None)
    first_pred = next((p for p in predictions.values() if p is not None), None)
    
    if first_history is not None and first_pred is not None and len(first_history) > 0 and len(first_pred) > 0:
        zoom_start = first_history.index[0]
        zoom_end = first_pred.index[-1]
    else:
        zoom_start = zoom_end = None
    
    # Update layout
    layout_config = {
        'height': 350 * n_plots,
        'hovermode': 'x unified',
        'showlegend': True,
        'margin': dict(l=80, r=20, t=100, b=60),
        'shapes': shapes,
        'legend': dict(
            tracegroupgap=30,
            groupclick="toggleitem"
        )
    }
    
    if zoom_start is not None and zoom_end is not None:
        layout_config['xaxis'] = dict(range=[zoom_start, zoom_end])
    
    fig.update_layout(**layout_config)
    
    fig.update_xaxes(title_text="Time", row=n_plots, col=1)
    
    for i in range(1, n_plots + 1):
        fig.update_yaxes(title_text="Value", row=i, col=1, automargin=True)
    
    return fig


# Main UI
st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    st.info(f"📊 Comparing {len(st.session_state.model_versions)} model version(s): {', '.join([f'v{v}' for v in st.session_state.model_versions])}")

with col2:
    if st.button("🔄 Load Models & Data", use_container_width=True, type="primary"):
        with st.spinner("Loading models and data..."):
            # Load data first
            st.session_state.data = load_data(data_file)
            
            # Load all model versions
            st.session_state.models = {}
            for version in st.session_state.model_versions:
                model = load_model(version)
                if model:
                    version_key = f"v{version}"
                    st.session_state.models[version_key] = model
            
            if st.session_state.models and st.session_state.data is not None:
                # Set initial timestamp from first model
                first_model = next(iter(st.session_state.models.values()))
                max_lag = first_model.input_window[0]
                st.session_state.current_timestamp = st.session_state.data.index[abs(max_lag)]

# Show model info if loaded
if st.session_state.models:
    st.divider()
    with st.expander("📋 Loaded Models Information", expanded=False):
        for version_key, model in st.session_state.models.items():
            st.subheader(f"Model {version_key}")
            col1, col2 = st.columns(2)
            with col1:
                inputs = model.inputs
                if isinstance(inputs, str):
                    try:
                        inputs = json.loads(inputs)
                    except (json.JSONDecodeError, ValueError):
                        pass
                st.write("**Inputs:**", list(inputs.keys()) if isinstance(inputs, dict) else inputs)
                st.write("**Input Window:**", model.input_window)
                
                # Get features from metadata
                try:
                    metadata = model.metadata.get_model_info().metadata
                    if "features" in metadata:
                        features = metadata["features"]
                        if isinstance(features, str):
                            features = json.loads(features)
                        st.write("**Features:**", features)
                except Exception as e:
                    st.write("**Features:**", "N/A")
                    
            with col2:
                outputs = model.outputs
                if isinstance(outputs, str):
                    try:
                        outputs = json.loads(outputs)
                    except (json.JSONDecodeError, ValueError):
                        pass
                st.write("**Outputs:**", list(outputs.keys()) if isinstance(outputs, dict) else outputs)
            st.divider()

# Timestamp selection and navigation
if st.session_state.models and st.session_state.data is not None:
    st.divider()
    
    # Get data frequency from first model
    first_model = next(iter(st.session_state.models.values()))
    data_freq_meta = first_model.metadata.get_model_info().metadata.get("data_freq")
    if data_freq_meta:
        try:
            data_freq = json.loads(data_freq_meta)
        except (json.JSONDecodeError, ValueError):
            data_freq = data_freq_meta
    else:
        data_freq = int(pd.Timedelta(
            st.session_state.data.index[1] - st.session_state.data.index[0]
        ).total_seconds() / 60)
    
    st.subheader("Forecast Control")

    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        max_lag = abs(first_model.input_window[0])
        available_times = st.session_state.data.index[max_lag:-1]
        
        # Calculate the index for the selectbox
        if st.session_state.current_timestamp is None:
            default_idx = 0
        elif st.session_state.current_timestamp in available_times:
            default_idx = list(available_times).index(st.session_state.current_timestamp)
        else:
            default_idx = 0
        
        selected_time = st.selectbox(
            "Select Forecast Time",
            options=available_times,
            index=default_idx,
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Check if timestamp changed from selectbox
        if selected_time != st.session_state.current_timestamp:
            st.session_state.current_timestamp = selected_time
            # Auto-generate forecasts for all model versions
            st.session_state.predictions = {}
            st.session_state.history_dfs = {}
            for idx, (version_key, model) in enumerate(st.session_state.models.items()):
                store_input = (idx == 0)  # Only store input for first model
                prediction, history_df, actual_df, forecast_start, freq = make_prediction(
                    model,
                    st.session_state.data,
                    st.session_state.current_timestamp,
                    version_key,
                    store_input=store_input
                )
                st.session_state.predictions[version_key] = prediction
                st.session_state.history_dfs[version_key] = history_df
                if actual_df is not None:
                    st.session_state.actual_df = actual_df
                if forecast_start is not None:
                    st.session_state.forecast_start = forecast_start
    
    with col2:
        if st.button("◀ Previous", use_container_width=True):
            current_idx = list(available_times).index(st.session_state.current_timestamp)
            if current_idx > 0:
                new_timestamp = available_times[current_idx - 1]
                st.session_state.current_timestamp = new_timestamp
                # Auto-generate forecasts for all model versions
                st.session_state.predictions = {}
                st.session_state.history_dfs = {}
                for idx, (version_key, model) in enumerate(st.session_state.models.items()):
                    store_input = (idx == 0)  # Only store input for first model
                    prediction, history_df, actual_df, forecast_start, freq = make_prediction(
                        model,
                        st.session_state.data,
                        st.session_state.current_timestamp,
                        version_key,
                        store_input=store_input
                    )
                    st.session_state.predictions[version_key] = prediction
                    st.session_state.history_dfs[version_key] = history_df
                    if actual_df is not None:
                        st.session_state.actual_df = actual_df
                    if forecast_start is not None:
                        st.session_state.forecast_start = forecast_start
                st.rerun()
    
    with col3:
        if st.button("Next ▶", use_container_width=True):
            current_idx = list(available_times).index(st.session_state.current_timestamp)
            if current_idx < len(available_times) - 1:
                new_timestamp = available_times[current_idx + 1]
                st.session_state.current_timestamp = new_timestamp
                # Auto-generate forecasts for all model versions
                st.session_state.predictions = {}
                st.session_state.history_dfs = {}
                for idx, (version_key, model) in enumerate(st.session_state.models.items()):
                    store_input = (idx == 0)  # Only store input for first model
                    prediction, history_df, actual_df, forecast_start, freq = make_prediction(
                        model,
                        st.session_state.data,
                        st.session_state.current_timestamp,
                        version_key,
                        store_input=store_input
                    )
                    st.session_state.predictions[version_key] = prediction
                    st.session_state.history_dfs[version_key] = history_df
                    if actual_df is not None:
                        st.session_state.actual_df = actual_df
                    if forecast_start is not None:
                        st.session_state.forecast_start = forecast_start
                st.rerun()
    with col4:
        if st.button("🔮 Forecast All", type="primary", use_container_width=True):
            with st.spinner("Making predictions for all model versions..."):
                st.session_state.predictions = {}
                st.session_state.history_dfs = {}
                # Store input for first model
                first_version_key = list(st.session_state.models.keys())[0]
                for idx, (version_key, model) in enumerate(st.session_state.models.items()):
                    store_input = (idx == 0)  # Only store input for first model
                    prediction, history_df, actual_df, forecast_start, freq = make_prediction(
                        model,
                        st.session_state.data,
                        st.session_state.current_timestamp,
                        version_key,
                        store_input=store_input
                    )
                    st.session_state.predictions[version_key] = prediction
                    st.session_state.history_dfs[version_key] = history_df
                    if actual_df is not None:
                        st.session_state.actual_df = actual_df
                    if forecast_start is not None:
                        st.session_state.forecast_start = forecast_start
    
    # Data Modification Panel
    if st.session_state.current_input_df is not None:
        st.divider()
        with st.expander("🔧 Input Data Modification", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Reset Modifications", use_container_width=True):
                    st.session_state.modified_data = None
                    st.success("✓ Data reset to original")
                    st.rerun()
            
            # Get feature columns from input data
            feature_cols = list(st.session_state.current_input_df.columns)
            
            # Modification controls
            mod_col1, mod_col2, mod_col3 = st.columns([2, 2, 1])
            
            with mod_col1:
                selected_feature = st.selectbox(
                    "Select Feature to Modify",
                    options=feature_cols,
                    key="mod_feature"
                )
            
            with mod_col2:
                mod_type = st.selectbox(
                    "Modification Type",
                    options=["Multiply by", "Add offset", "Set to value"],
                    key="mod_type"
                )
            
            with mod_col3:
                if mod_type == "Multiply by":
                    mod_value = st.number_input("Factor", value=1.0, step=0.1, key="mod_value")
                elif mod_type == "Add offset":
                    mod_value = st.number_input("Offset", value=0.0, step=0.5, key="mod_value")
                else:  # Set to value
                    mod_value = st.number_input("Value", value=0.0, step=0.5, key="mod_value")
            
            col_apply, col_show = st.columns([1, 3])
            
            with col_apply:
                if st.button("✅ Apply Modification", use_container_width=True):
                    # Initialize modified data if not exists
                    if st.session_state.modified_data is None:
                        st.session_state.modified_data = st.session_state.data.copy()
                    
                    # Get the time range from current input
                    time_range = st.session_state.current_input_df.index
                    
                    # Apply modification
                    if mod_type == "Multiply by":
                        st.session_state.modified_data.loc[time_range, selected_feature] *= mod_value
                    elif mod_type == "Add offset":
                        st.session_state.modified_data.loc[time_range, selected_feature] += mod_value
                    else:  # Set to value
                        st.session_state.modified_data.loc[time_range, selected_feature] = mod_value
                    
                    st.success(f"✓ Modified {selected_feature}")
            
            # Show input data with modifications
            with st.expander("📋 Current Input Data", expanded=True):
                # Show comparison if data is modified
                if st.session_state.modified_data is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Input Data**")
                        original_input = st.session_state.original_data.loc[
                            st.session_state.current_input_df.index,
                            st.session_state.current_input_df.columns
                        ]
                        st.dataframe(original_input, use_container_width=True, height=300)
                    with col2:
                        st.write("**Modified Input Data** (🔴 = changed)")
                        modified_input = st.session_state.modified_data.loc[
                            st.session_state.current_input_df.index,
                            st.session_state.current_input_df.columns
                        ]
                        
                        # Highlight modified values
                        def highlight_modified(row):
                            orig_row = original_input.loc[row.name]
                            return ['background-color: rgba(255, 100, 100, 0.3)' if row[col] != orig_row[col] else '' 
                                    for col in row.index]
                        
                        styled_df = modified_input.style.apply(highlight_modified, axis=1)
                        st.dataframe(styled_df, use_container_width=True, height=300)
                else:
                    st.dataframe(st.session_state.current_input_df, use_container_width=True, height=300)
    
    # Display forecast comparison
    if st.session_state.predictions:
        st.divider()
        st.subheader("📈 Forecast Comparison")
        
        if st.session_state.modified_data is not None:
            st.info("🔴 Forecasts are based on **modified** input data")
        
        fig = plot_forecast_comparison(
            st.session_state.history_dfs,
            st.session_state.predictions,
            st.session_state.actual_df,
            st.session_state.models,
            st.session_state.forecast_start
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction data for each model version
        with st.expander("📊 Prediction Data (All Versions)", expanded=False):
            for version_key, prediction in st.session_state.predictions.items():
                if prediction is not None:
                    st.write(f"**Model {version_key} Forecast**")
                    st.dataframe(prediction, use_container_width=True)
                    st.divider()

else:
    st.info("Click 'Load Models & Data' to get started")

# Footer
st.divider()
st.caption(f"Interactive Forecast Viewer | Data frequency: {data_freq if st.session_state.models else 'N/A'} min | Model versions loaded: {len(st.session_state.models)}")
