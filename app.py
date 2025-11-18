# app.py
import streamlit as st
import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import streamlit.components.v1 as components


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RiskSentinel | AI Audit", layout="wide")

st.title("🛡️ RiskSentinel: Automated ML Stress-Tester")
st.markdown("""
**Objective:** Detect **Data Drift** and **Model Decay** in production environments.  
Upload your baseline data (Reference) and your new incoming data (Current) to generate a compliance report.
""")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("1. Upload Data Assets")
ref_file = st.sidebar.file_uploader("Upload Reference Data (Training)", type=['csv'])
curr_file = st.sidebar.file_uploader("Upload Current Data (Production)", type=['csv'])
model_file = st.sidebar.file_uploader("Upload Model (.pkl) [Optional]", type=['pkl'])

st.sidebar.header("2. Configure Audit")
check_drift = st.sidebar.checkbox("Check for Data Drift", value=True)
check_target = st.sidebar.checkbox("Check for Target Drift", value=False)

# --- MAIN LOGIC ---
if ref_file and curr_file:
    # Load Data
    ref_data = pd.read_csv(ref_file)
    curr_data = pd.read_csv(curr_file)

    st.success("✅ Data Loaded Successfully")

    # Display Preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reference Distribution")
        st.write(ref_data.head(3))
    with col2:
        st.subheader("Current Distribution")
        st.write(curr_data.head(3))

    # Button to Run Audit
    if st.button("🚀 Run Risk Audit"):
        with st.spinner("Running Statistical Tests (KS-Test, Jensen-Shannon)..."):
            
            # 1. Setup Evidently Report
            metrics_list = []
            if check_drift:
                metrics_list.append(DataDriftPreset())
            #if check_target:
                #metrics_list.append(TargetDriftPreset())
            
            report = Report(metrics=metrics_list)
            
            # 2. Calculate Drift
            # (If model is provided, we could add prediction drift here too)
            report.run(reference_data=ref_data, current_data=curr_data)
            
            # 3. Save and Display
            report.save_html("report.html")
            
            # Read HTML file and display in Streamlit
            with open("report.html", "r", encoding='utf-8') as f:
                html_content = f.read()
            
            st.header("📋 Compliance Report")
            components.html(html_content, height=1000, scrolling=True)
            
            st.download_button(
                label="📥 Download Audit Report (HTML)",
                data=html_content,
                file_name="risk_sentinel_audit.html",
                mime="text/html"
            )

else:
    st.info("👈 Please upload 'reference.csv' and 'current.csv' to begin.")