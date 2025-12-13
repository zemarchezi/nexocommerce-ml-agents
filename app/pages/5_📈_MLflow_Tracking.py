"""
MLflow Tracking - View ML experiments and model performance
"""

import streamlit as st
import sys
from pathlib import Path
import mlflow
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="MLflow Tracking", page_icon="📈", layout="wide")

# Header
st.title("📈 MLflow Tracking")
st.markdown("Monitor ML experiments, model performance, and training metrics")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mlflow_uri = st.text_input(
        "MLflow Tracking URI",
        value="http://localhost:5000",
        help="MLflow tracking server URI"
    )
    
    st.markdown("---")
    
    st.header("📊 Quick Links")
    st.markdown(f"[Open MLflow UI]({mlflow_uri})")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Try to connect to MLflow
try:
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    st.success(f"✅ Connected to MLflow at {mlflow_uri}")
    
except Exception as e:
    st.error(f"❌ Could not connect to MLflow: {e}")
    st.info("💡 Make sure MLflow server is running: `mlflow ui --port 5000`")
    st.stop()

# Get experiments
try:
    experiments = client.search_experiments()
    
    if not experiments:
        st.warning("No experiments found in MLflow")
        st.info("Run the training pipeline to create experiments: `python src/pipeline/training_pipeline.py`")
        st.stop()
    
    # Experiment selector
    experiment_names = [exp.name for exp in experiments if exp.name != "Default"]
    
    if not experiment_names:
        st.warning("No custom experiments found")
        st.stop()
    
    selected_experiment = st.selectbox(
        "Select Experiment",
        experiment_names,
        index=0
    )
    
    # Get experiment details
    experiment = client.get_experiment_by_name(selected_experiment)
    
    st.markdown("---")
    
    # Experiment Info
    st.header(f"🔬 Experiment: {selected_experiment}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Experiment ID", experiment.experiment_id)
    
    with col2:
        st.metric("Artifact Location", experiment.artifact_location.split("/")[-1] + "...")
    
    with col3:
        st.metric("Lifecycle Stage", experiment.lifecycle_stage)
    
    st.markdown("---")
    
    # Get runs for this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=100
    )
    
    if not runs:
        st.warning("No runs found for this experiment")
        st.stop()
    
    st.success(f"Found {len(runs)} runs")
    
    # Runs Overview
    st.header("🏃 Runs Overview")
    
    # Create runs dataframe
    runs_data = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id[:8] + "...",
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit='ms'),
            "duration": (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else 0,
        }
        
        # Add metrics
        for key, value in run.data.metrics.items():
            run_data[key] = value
        
        # Add params
        for key, value in run.data.params.items():
            run_data[f"param_{key}"] = value
        
        runs_data.append(run_data)
    
    runs_df = pd.DataFrame(runs_data)
    
    # Display runs table
    st.subheader("📋 All Runs")
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Metrics Comparison
    st.header("📊 Metrics Comparison")
    
    # Get available metrics
    metric_columns = [col for col in runs_df.columns if col not in ['run_id', 'status', 'start_time', 'duration'] and not col.startswith('param_')]
    
    if metric_columns:
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            metric_columns,
            default=metric_columns[:3] if len(metric_columns) >= 3 else metric_columns
        )
        
        if selected_metrics:
            # Create comparison chart
            fig = go.Figure()
            
            for metric in selected_metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=runs_df['run_id'],
                    y=runs_df[metric],
                    text=runs_df[metric].round(4),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Metrics Comparison Across Runs",
                xaxis_title="Run ID",
                yaxis_title="Metric Value",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics found in runs")
    
    st.markdown("---")
    
    # Model Performance Over Time
    st.header("📈 Performance Over Time")
    
    if 'accuracy' in runs_df.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=runs_df['start_time'],
            y=runs_df['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=10)
        ))
        
        if 'f1_score' in runs_df.columns:
            fig.add_trace(go.Scatter(
                x=runs_df['start_time'],
                y=runs_df['f1_score'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='#3498db', width=2),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Score",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Best Run Analysis
    st.header("🏆 Best Run Analysis")
    
    if 'accuracy' in runs_df.columns:
        best_run_idx = runs_df['accuracy'].idxmax()
        best_run = runs[best_run_idx]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Best Accuracy",
                f"{runs_df.loc[best_run_idx, 'accuracy']:.4f}"
            )
        
        with col2:
            if 'precision' in runs_df.columns:
                st.metric(
                    "Precision",
                    f"{runs_df.loc[best_run_idx, 'precision']:.4f}"
                )
        
        with col3:
            if 'recall' in runs_df.columns:
                st.metric(
                    "Recall",
                    f"{runs_df.loc[best_run_idx, 'recall']:.4f}"
                )
        
        with col4:
            if 'f1_score' in runs_df.columns:
                st.metric(
                    "F1 Score",
                    f"{runs_df.loc[best_run_idx, 'f1_score']:.4f}"
                )
        
        # Best run parameters
        st.subheader("⚙️ Best Run Parameters")
        
        params = best_run.data.params
        if params:
            param_df = pd.DataFrame([params]).T
            param_df.columns = ['Value']
            st.dataframe(param_df, use_container_width=True)
        else:
            st.info("No parameters logged for best run")
        
        # Best run artifacts
        st.subheader("📦 Best Run Artifacts")
        
        try:
            artifacts = client.list_artifacts(best_run.info.run_id)
            if artifacts:
                artifact_names = [artifact.path for artifact in artifacts]
                st.write("Available artifacts:")
                for artifact in artifact_names:
                    st.write(f"- {artifact}")
            else:
                st.info("No artifacts found for best run")
        except Exception as e:
            st.warning(f"Could not load artifacts: {e}")
    
    st.markdown("---")
    
    # Hyperparameter Analysis
    st.header("🔧 Hyperparameter Analysis")
    
    param_columns = [col for col in runs_df.columns if col.startswith('param_')]
    
    if param_columns and 'accuracy' in runs_df.columns:
        selected_param = st.selectbox(
            "Select parameter to analyze",
            param_columns
        )
        
        if selected_param:
            # Create scatter plot
            fig = px.scatter(
                runs_df,
                x=selected_param,
                y='accuracy',
                size='duration' if 'duration' in runs_df.columns else None,
                color='status',
                hover_data=['run_id'],
                title=f"Accuracy vs {selected_param.replace('param_', '')}"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Registry
    st.header("📚 Model Registry")
    
    try:
        registered_models = client.search_registered_models()
        
        if registered_models:
            st.success(f"Found {len(registered_models)} registered models")
            
            for model in registered_models:
                with st.expander(f"📦 {model.name}", expanded=False):
                    st.write(f"**Description:** {model.description or 'No description'}")
                    st.write(f"**Latest Version:** {model.latest_versions[0].version if model.latest_versions else 'N/A'}")
                    
                    # Get model versions
                    versions = client.search_model_versions(f"name='{model.name}'")
                    
                    if versions:
                        version_data = []
                        for version in versions:
                            version_data.append({
                                "Version": version.version,
                                "Stage": version.current_stage,
                                "Run ID": version.run_id[:8] + "...",
                                "Created": pd.to_datetime(version.creation_timestamp, unit='ms')
                            })
                        
                        version_df = pd.DataFrame(version_data)
                        st.dataframe(version_df, use_container_width=True, hide_index=True)
        else:
            st.info("No registered models found")
            st.write("Models can be registered through the MLflow UI or programmatically")
    
    except Exception as e:
        st.warning(f"Could not load model registry: {e}")
    
    st.markdown("---")
    
    # Export Options
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export runs as CSV
        csv_export = runs_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Runs CSV",
            data=csv_export,
            file_name=f"mlflow_runs_{selected_experiment}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export best run info
        if 'accuracy' in runs_df.columns:
            best_run_info = {
                "experiment": selected_experiment,
                "run_id": best_run.info.run_id,
                "metrics": dict(best_run.data.metrics),
                "params": dict(best_run.data.params)
            }
            
            import json
            json_export = json.dumps(best_run_info, indent=2)
            
            st.download_button(
                label="📥 Download Best Run JSON",
                data=json_export,
                file_name=f"best_run_{selected_experiment}.json",
                mime="application/json",
                use_container_width=True
            )

except Exception as e:
    st.error(f"❌ Error loading MLflow data: {e}")
    import traceback
    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("💡 Tip: Use MLflow UI for more detailed experiment tracking and model management")