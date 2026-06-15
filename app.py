import streamlit as st
import pandas as pd
import json
import random
from predict import SupportIntegrityAuditor

st.set_page_config(
    page_title="SIA Dashboard | Operations Portal",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_cached_auditor():
    try:
        return SupportIntegrityAuditor()
    except Exception as e:
        st.error(f"Failed to access local fine-tuned weights folder: {e}")
        return None

auditor = get_cached_auditor()

st.title("🤖 Support Integrity Auditor (SIA)")
st.markdown("Automated triage optimization engine producing authentic multi-layered Evidence Dossiers matching notebook standards.")
st.write("---")

tab1, tab2 = st.tabs(["🎯 Single Ticket Audit", "📂 Bulk Log File Pipeline"])

# --- TAB 1: SINGLE TICKET INFERENCE ---
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 CRM Operational Log Data Specification")
        
        category = st.selectbox("Operational Category", ["General Inquiry", "Technical", "Billing", "Account"])
        text_input = st.text_area("Ticket Text Content String", placeholder="Paste incoming conversation text body...", height=110)
        
        c1, c2 = st.columns(2)
        with c1:
            priority = st.selectbox("Assigned Priority Level", ["Low", "Medium", "High", "Critical"])
            domain = st.selectbox("Customer Domain Tier", ["Enterprise", "SME", "Free Tier"])
        with c2:
            channel = st.selectbox("Inbound Support Channel", ["Web Form", "Chat", "Email", "API"])
            res_hours = st.number_input("Resolution Time (Hours)", min_value=1, max_value=720, value=24)
            
        run_single = st.button("Generate Evidence Dossier", type="primary")

    with col2:
        st.subheader("📄 Generated Analytical Evidence Dossier")
        if run_single:
            if not text_input.strip():
                st.warning("Please provide input ticket text context to evaluate.")
            elif auditor is None:
                st.error("Model Pipeline Engine Offline.")
            else:
                generated_id = f"TKT-{random.randint(100000, 999999)}"
                
                is_mismatch, conf, dossier = auditor.audit_ticket(
                    ticket_id=generated_id,
                    category=category,
                    text=text_input,
                    human_priority=priority,
                    customer_domain=domain,
                    support_channel=channel,
                    resolution_time_hrs=res_hours
                )
                
                # Fetch specific custom descriptive metrics out of metadata layer
                audit_label = dossier["metadata"]["consensus_severity_label"]
                
                if audit_label == "Hidden Crisis":
                    st.error(f"🚨 MISMATCH FLAG: {audit_label.upper()} (Ensemble Verification: {dossier['confidence']})")
                elif audit_label == "False Alarm":
                    st.warning(f"⚠️ MISMATCH FLAG: {audit_label.upper()} (Ensemble Verification: {dossier['confidence']})")
                else:
                    st.success(f"✅ UNIFORMITY VERIFIED: {audit_label.upper()} (Ensemble Verification: {dossier['confidence']})")
                
                st.markdown(f"**Text Analysis Interpretation:**\n> {dossier['constraint_analysis']}")
                
                st.write("---")
                with st.status("Review Dossier JSON Elements", expanded=True):
                    st.json(dossier)
                    
                st.download_button(
                    "📥 Save Exportable Evidence Dossier File (.json)",
                    data=json.dumps(dossier, indent=4),
                    file_name=f"SIA_Dossier_{generated_id}.json",
                    mime="application/json",
                    use_container_width=True
                )

# --- TAB 2: BATCH SPREADSHEET INFERENCE ---
with tab2:
    st.subheader("📥 Upload Operational Log Archives")
    uploaded_file = st.file_uploader("Upload data manifest spreadsheet", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"File parsed successfully. Found {len(df)} lines to crosscheck.")
        st.dataframe(df.head(3), use_container_width=True)
        
        st.write("---")
        st.markdown("#### 🛠️ Configure Source Target Data Column Mapping")
        avail_cols = list(df.columns)
        
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            col_id = st.selectbox("Ticket ID Column Name", avail_cols, index=0)
            col_cat = st.selectbox("Category Column Name", avail_cols, index=min(5, len(avail_cols)-1))
            col_text = st.selectbox("Text Content Column Name", avail_cols, index=min(4, len(avail_cols)-1))
        with mc2:
            col_priority = st.selectbox("Priority Column Name", avail_cols, index=min(6, len(avail_cols)-1))
            col_domain = st.selectbox("Customer Domain Column Name", avail_cols, index=min(16, len(avail_cols)-1))
        with mc3:
            col_channel = st.selectbox("Support Channel Column Name", avail_cols, index=min(7, len(avail_cols)-1))
            col_time = st.selectbox("Resolution Time Column Name", avail_cols, index=min(9, len(avail_cols)-1))
            
        if st.button("Execute Multi-Layered Batch Audit Pipeline", type="primary"):
            if auditor is None:
                st.error("Inference pipeline broken.")
            else:
                with st.spinner("Processing analytical matrices across row registers..."):
                    processed_df, full_dossier_list = auditor.audit_batch(
                        df=df, id_col=col_id, cat_col=col_cat, text_col=col_text,
                        priority_col=col_priority, domain_col=col_domain,
                        channel_col=col_channel, time_col=col_time
                    )
                
                alert_count = int((processed_df["SIA_Verdict"] == "MISMATCH_ALERT").sum())
                
                st.write("---")
                st.subheader("📈 Execution Pipeline Evaluation Summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Rows Parsed", f"{len(processed_df)} entries")
                m2.metric("Mismatches Uncovered", f"{alert_count} cases")
                m3.metric("Systemic Deviation Rate", f"{(alert_count / len(processed_df)) * 100:.2f}%")
                
                st.dataframe(processed_df, use_container_width=True)
                
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "📥 Save Audited Analysis Manifest (CSV)",
                        data=processed_df.to_csv(index=False).encode('utf-8'),
                        file_name="SIA_Batch_Summary_Output.csv", mime="text/csv", use_container_width=True
                    )
                with dl2:
                    st.download_button(
                        "📥 Save Detailed Portfolio Package (JSON)",
                        data=json.dumps(full_dossier_list, indent=4),
                        file_name="SIA_Batch_Evidence_Dossiers.json", mime="application/json", use_container_width=True
                    )
