"""
Streamlit Web Application for Real-Time Customer Complaint Classification & Routing.
"""

import streamlit as st
import pandas as pd
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import ComplaintClassifierService

st.set_page_config(
    page_title="AI Customer Complaint Classifier",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def get_classifier():
    return ComplaintClassifierService()

try:
    classifier = get_classifier()
except Exception as e:
    st.error(f"Failed to load classifier: {e}")
    st.stop()

st.title("🛡️ Automated Customer Complaint Classifier & Triaging")
st.markdown(
    "NLP-powered complaint categorization, confidence scoring, and urgency escalation detection."
)

tab1, tab2, tab3 = st.tabs(["📝 Single Ticket Classification", "📂 Batch Upload (CSV)", "ℹ️ Dataset & Model Info"])

with tab1:
    st.subheader("Analyze Individual Complaint")
    sample_text = (
        "I noticed unauthorized transactions on my debit card from an unknown ATM. "
        "I contacted customer service to reverse the fees, but they refused to help and closed my claim."
    )
    user_complaint = st.text_area(
        "Enter Complaint Narrative:",
        value=sample_text,
        height=150
    )

    if st.button("Classify Complaint", type="primary"):
        if not user_complaint.strip():
            st.warning("Please enter complaint text.")
        else:
            with st.spinner("Analyzing text..."):
                res = classifier.predict(user_complaint)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Predicted Category")
                st.info(f"**{res['predicted_category']}**")
                st.metric("Confidence Score", f"{res['confidence']*100:.1f}%")

            with col2:
                st.markdown("### Escalation Priority")
                urgency = res["urgency"]
                if urgency["priority"] == "CRITICAL / HIGH":
                    st.error(f"🚨 **{urgency['priority']}**")
                elif urgency["priority"] == "MEDIUM":
                    st.warning(f"⚠️ **{urgency['priority']}**")
                else:
                    st.success(f"✅ **{urgency['priority']}**")

                if urgency["triggers"]:
                    st.write(f"Trigger terms detected: `{', '.join(urgency['triggers'])}`")

            st.markdown("### Top Candidate Probabilities")
            df_candidates = pd.DataFrame(res["top_candidates"])
            df_candidates["confidence_pct"] = df_candidates["confidence"] * 100
            st.bar_chart(df_candidates.set_index("category")["confidence_pct"])

with tab2:
    st.subheader("Bulk Complaint Classification")
    st.markdown("Upload a CSV file containing customer complaints to classify them in bulk.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("File Preview:", df_upload.head(3))
        text_cols = df_upload.columns.tolist()
        selected_col = st.selectbox("Select the column containing the complaint narrative:", text_cols)

        if st.button("Run Batch Classification"):
            with st.spinner(f"Classifying {len(df_upload)} complaints..."):
                results = classifier.predict_batch(df_upload[selected_col].astype(str).tolist())
                df_upload["Predicted_Category"] = [r.get("predicted_category", "Error") for r in results]
                df_upload["Confidence"] = [r.get("confidence", 0.0) for r in results]
                df_upload["Priority"] = [r.get("urgency", {}).get("priority", "NORMAL") for r in results]

            st.success("Batch classification complete!")
            st.dataframe(df_upload[[selected_col, "Predicted_Category", "Confidence", "Priority"]].head(20))

            csv_data = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("Download Annotated CSV", csv_data, "classified_complaints.csv", "text/csv")

with tab3:
    st.subheader("Model Taxonomy & Performance")
    st.markdown("""
    - **Dataset**: Consumer Complaint Database (CFPB / Kaggle)
    - **Total Records in Database**: 1,280,000+
    - **Training Set**: Stratified & Deduplicated Balanced Sample (44,178 complaints)
    - **Model Architecture**: TF-IDF (Unigrams + Bigrams, Sublinear TF) + Calibrated Logistic Regression
    - **Test Accuracy**: **83.1%**
    - **Macro F1 Score**: **83.0%**
    - **Categories**:
        1. Mortgages & Home Loans
        2. Credit Cards & Prepaid
        3. Debt Collection
        4. Credit Reporting
        5. Retail Banking & Accounts
        6. Student Loans
        7. Personal & Consumer Loans
        8. Vehicle Loans
        9. Money Transfers & Digital Currency
    """)
