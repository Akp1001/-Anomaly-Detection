import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from final_anomaly_code import MergedAnomalyDetectorGPT

# Streamlit Page Config
st.set_page_config(page_title="Anomaly Detection Dashboard", page_icon="📊", layout="wide")
st.title("📊 Advanced Anomaly Detection Dashboard")
st.write("Upload training and test datasets, run anomaly detection, and visualize results.")

# File Upload Section
st.header("📁 Upload Datasets")
col1, col2 = st.columns(2)

with col1:
    train_file = st.file_uploader("Upload Training CSV", type="csv", key="train")
with col2:
    test_file = st.file_uploader("Upload Test CSV", type="csv", key="test")

if train_file and test_file:
    train_df = pd.read_csv(train_file, delimiter=";")
    test_df = pd.read_csv(test_file, delimiter=";")
    st.success(f"✅ Loaded {len(train_df)} training and {len(test_df)} test samples")

    if st.checkbox("🔍 Show Training Data Preview"):
        st.dataframe(train_df.head())
    if st.checkbox("🔍 Show Test Data Preview"):
        st.dataframe(test_df.head())

    epochs = st.slider("Neural Network Epochs", 10, 100, 50)

    if st.button("🚀 Run Full Analysis"):
        with st.spinner("Running anomaly detection..."):
            train_df.to_csv("temp_train.csv", sep=";", index=False)
            test_df.to_csv("temp_test.csv", sep=";", index=False)

            detector = MergedAnomalyDetectorGPT()
            results = detector.run_pipeline(
                train_path="temp_train.csv",
                test_path="temp_test.csv",
                epochs=epochs,
                show_plots=False,
                plots_dir="plots"
            )

            # ✅ Force exactly 14 features (remove Throughput_Ratio if present)
            if "Throughput_Ratio" in detector.features:
                detector.features.remove("Throughput_Ratio")
                feature_idx = [i for i, f in enumerate(detector.features)]
                detector.X = detector.X[:, feature_idx]
                detector.X_test = detector.X_test[:, feature_idx]
                detector.X_scaled = detector.scaler.fit_transform(detector.X)
                detector.X_test_scaled = detector.scaler.transform(detector.X_test)

            st.session_state["detector"] = detector
            st.session_state["results"] = results

# Results Section
if "detector" in st.session_state:
    detector = st.session_state["detector"]
    results = st.session_state["results"]

    st.header("📊 Results & Insights")

    # Show total number of features used (always 14)
    st.info(f"🔑 **Total Features Used:** {len(detector.features)} ({', '.join(detector.features)})")

    # Summary Metrics
    st.subheader("📈 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)

    total_samples = len(detector.X_test)
    predictions = pd.read_csv("anomaly_predictions.csv")["Unusual"]
    anomaly_count = int(predictions.sum())
    anomaly_rate = (anomaly_count / total_samples) * 100

    with col1:
        st.metric("Best Model", detector.best_model_name.replace("_", " ").title())
    with col2:
        st.metric("Total Samples", total_samples)
    with col3:
        st.metric("Anomalies Found", anomaly_count)
    with col4:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

    # Model Performance Table
    st.subheader("🏆 Model Performance")
    performance_data = []
    for model, info in detector.models.items():
        performance_data.append({
            "Model": model.replace("_", " ").title(),
            "F1 Score": f"{info['f1_score']:.4f}",
            "Precision": f"{info['precision']:.4f}",
            "Recall": f"{info['recall']:.4f}"
        })
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)

    # Plotly Model Performance Chart
    st.subheader("📊 Model Performance (Plotly)")
    models = [d["Model"] for d in performance_data]
    f1_scores = [float(d["F1 Score"]) for d in performance_data]
    precisions = [float(d["Precision"]) for d in performance_data]
    recalls = [float(d["Recall"]) for d in performance_data]

    fig_performance = go.Figure()
    fig_performance.add_trace(go.Bar(name="F1 Score", x=models, y=f1_scores))
    fig_performance.add_trace(go.Bar(name="Precision", x=models, y=precisions))
    fig_performance.add_trace(go.Bar(name="Recall", x=models, y=recalls))
    fig_performance.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode="group"
    )
    st.plotly_chart(fig_performance, use_container_width=True)

    # Prediction Distribution Pie Chart
    st.subheader("📊 Prediction Distribution")
    normal_count = total_samples - anomaly_count
    fig_pie = go.Figure(data=[go.Pie(labels=["Normal", "Anomaly"],
                                     values=[normal_count, anomaly_count],
                                     hole=0.3)])
    fig_pie.update_layout(title="Prediction Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Radar Chart for Model Comparison
    st.subheader("🧭 Model Performance Radar Chart")
    metrics = ['F1 Score', 'Precision', 'Recall']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ['blue', 'red', 'green', 'orange']

    for i, model in enumerate(models):
        values = [f1_scores[i], precisions[i], recalls[i]]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar Chart", fontsize=14, fontweight="bold")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    st.pyplot(fig)

    # Display All Saved Plots from 'plots/' Directory
    st.subheader("📈 Detailed Visualizations")
    plot_files = sorted([f for f in os.listdir("plots") if f.endswith(".png")])
    if plot_files:
        for pf in plot_files:
            st.image(os.path.join("plots", pf), caption=pf, use_container_width=True)
    else:
        st.info("No plots found in 'plots/' directory. (Try rerunning the pipeline)")

    # ROC & PR Curves
    st.subheader("📉 ROC & Precision-Recall Curves")
    detector.evaluate_all_models()  # generates ROC & PR plots
    st.pyplot(plt.gcf())

    # Download Section
    st.subheader("💾 Download Results")
    col1, col2 = st.columns(2)
    with col1:
        with open("anomaly_predictions.csv", "rb") as f:
            st.download_button(
                "📥 Download Predictions CSV",
                f,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    with col2:
        import json
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_model": detector.best_model_name,
            "total_samples": total_samples,
            "anomalies_detected": anomaly_count,
            "anomaly_rate": f"{anomaly_rate:.2f}%",
            "model_performance": performance_data,
            "total_features": len(detector.features),
            "features_used": detector.features
        }
        st.download_button(
            "📊 Download Report JSON",
            data=json.dumps(report, indent=2),
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • Enhanced Anomaly Detection Dashboard")
