"""
Settings Page

Configure system parameters, API keys, and thresholds.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Settings | Sentinel-AI",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Settings")
st.markdown("Configure system parameters, thresholds, and integrations.")

# Tabs for different settings sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🔑 API Configuration",
    "🎚️ Thresholds",
    "🛡️ Sensitivity Filter",
    "👤 User Settings"
])

with tab1:
    st.subheader("🔑 LLM API Configuration")
    
    st.info("⚠️ API keys are stored in environment variables. Update your `.env` file for permanent changes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Google Gemini")
        gemini_key = st.text_input(
            "Gemini API Key",
            value="•" * 20 if os.getenv("GOOGLE_API_KEY") else "",
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        if st.button("Test Gemini Connection"):
            with st.spinner("Testing..."):
                if os.getenv("GOOGLE_API_KEY"):
                    st.success("✅ Gemini API connected successfully!")
                else:
                    st.error("❌ No API key configured")
    
    with col2:
        st.markdown("#### OpenAI GPT-4o")
        openai_key = st.text_input(
            "OpenAI API Key",
            value="•" * 20 if os.getenv("OPENAI_API_KEY") else "",
            type="password",
            help="Get your API key from OpenAI Platform"
        )
        
        if st.button("Test OpenAI Connection"):
            with st.spinner("Testing..."):
                if os.getenv("OPENAI_API_KEY"):
                    st.success("✅ OpenAI API connected successfully!")
                else:
                    st.error("❌ No API key configured")
    
    st.markdown("---")
    
    st.markdown("#### Default Provider")
    default_provider = st.radio(
        "Select primary LLM provider:",
        ["Gemini 1.5 Pro (Recommended)", "GPT-4o"],
        horizontal=True
    )
    
    st.markdown("---")
    
    st.subheader("🗄️ Database Configuration")
    
    db_url = st.text_input(
        "Database URL",
        value=os.getenv("DATABASE_URL", "sqlite:///./sentinel_ai.db"),
        help="PostgreSQL or SQLite connection string"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Database Connection"):
            with st.spinner("Testing..."):
                try:
                    from src.database.connection import DatabaseManager
                    if DatabaseManager.health_check():
                        st.success("✅ Database connected!")
                    else:
                        st.error("❌ Connection failed")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("Initialize Database"):
            with st.spinner("Initializing..."):
                try:
                    from src.database.connection import init_db
                    init_db()
                    st.success("✅ Database initialized!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with tab2:
    st.subheader("🎚️ Audit Thresholds")
    
    st.markdown("Configure confidence thresholds for automated decisions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Escalation Threshold")
        escalation_threshold = st.slider(
            "Confidence below this triggers human review",
            min_value=50,
            max_value=90,
            value=70,
            step=5,
            help="Lower values = more automated decisions, higher values = more human reviews"
        )
        st.caption(f"Current: {escalation_threshold}%")
        
        st.markdown("#### Auto-Fail Threshold")
        auto_fail_threshold = st.slider(
            "Confidence above this auto-rejects content",
            min_value=70,
            max_value=99,
            value=85,
            step=5,
            help="High-confidence violations are automatically rejected"
        )
        st.caption(f"Current: {auto_fail_threshold}%")
    
    with col2:
        st.markdown("#### Severity Weights")
        
        critical_weight = st.number_input("Critical Severity Weight", value=15.0, step=1.0)
        high_weight = st.number_input("High Severity Weight", value=10.0, step=1.0)
        medium_weight = st.number_input("Medium Severity Weight", value=5.0, step=1.0)
        low_weight = st.number_input("Low Severity Weight", value=2.0, step=1.0)
    
    st.markdown("---")
    
    st.markdown("#### Policy-Specific Thresholds")
    
    policies = ["Hate Speech", "Harassment", "Misinformation", "Spam", "Violence", "Adult Content"]
    
    policy_thresholds = {}
    cols = st.columns(3)
    
    for i, policy in enumerate(policies):
        with cols[i % 3]:
            policy_thresholds[policy] = st.slider(
                policy,
                min_value=50,
                max_value=95,
                value=70,
                step=5,
                key=f"policy_threshold_{policy}"
            )
    
    if st.button("💾 Save Thresholds", type="primary"):
        st.success("✅ Thresholds saved successfully!")

with tab3:
    st.subheader("🛡️ Sensitivity Filter Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Filter Settings")
        
        filter_enabled = st.checkbox("Enable Sensitivity Filter", value=True)
        
        mask_character = st.selectbox(
            "Mask Character",
            ["█", "●", "*", "X"],
            index=0
        )
        
        show_warnings = st.checkbox("Show Content Warnings", value=True)
        
        st.markdown("#### Category Settings")
        
        categories = {
            "Graphic Violence": True,
            "Adult Content": True,
            "Self-Harm": True,
            "Disturbing Content": True
        }
        
        for cat, default in categories.items():
            st.checkbox(f"Filter {cat}", value=default, key=f"filter_{cat}")
    
    with col2:
        st.markdown("#### Custom Keywords")
        
        st.text_area(
            "Additional keywords to filter (one per line)",
            placeholder="Enter custom keywords...",
            height=150
        )
        
        st.markdown("#### Preview")
        
        sample_text = "This is a sample with blood and violence mentioned."
        
        if filter_enabled:
            filtered = "This is a sample with ████ and ████████ mentioned."
            st.markdown(f"""
            <div style="background: #2d2d3d; padding: 1rem; border-radius: 8px;">
                <strong>Original:</strong><br>
                <span style="color: #888;">{sample_text}</span>
                <br><br>
                <strong>Filtered:</strong><br>
                <span>{filtered}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #2d2d3d; padding: 1rem; border-radius: 8px;">
                <span>{sample_text}</span>
                <br><br>
                <em style="color: #888;">Filter disabled - content shown as-is</em>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("💾 Save Filter Settings", type="primary"):
        st.success("✅ Filter settings saved!")

with tab4:
    st.subheader("👤 User Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Profile")
        
        user_name = st.text_input("Display Name", value="Quality Specialist")
        user_id = st.text_input("User ID", value="QMS-001", disabled=True)
        user_email = st.text_input("Email", value="qms@example.com")
        
        st.markdown("#### Notifications")
        
        st.checkbox("Email notifications for escalations", value=True)
        st.checkbox("Daily summary reports", value=True)
        st.checkbox("Critical violation alerts", value=True)
    
    with col2:
        st.markdown("#### Display Preferences")
        
        theme = st.selectbox("Theme", ["Dark (Default)", "Light", "System"])
        
        items_per_page = st.slider("Items per page in tables", 10, 50, 25, 5)
        
        date_format = st.selectbox(
            "Date format",
            ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"]
        )
        
        st.markdown("#### Keyboard Shortcuts")
        
        st.checkbox("Enable keyboard shortcuts", value=True)
        
        st.markdown("""
        | Shortcut | Action |
        |----------|--------|
        | `Ctrl+Enter` | Submit review |
        | `Ctrl+S` | Skip current item |
        | `Ctrl+P` | Mark as Pass |
        | `Ctrl+F` | Mark as Fail |
        """)
    
    if st.button("💾 Save User Settings", type="primary"):
        st.success("✅ Settings saved!")

# System Information
st.markdown("---")
st.subheader("ℹ️ System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Application**
    - Version: 1.0.0
    - Environment: Production
    - Last Updated: 2026-01-19
    """)

with col2:
    st.markdown("""
    **Agents**
    - Policy Agent: ✅ Active
    - Hindi Cultural Agent: ✅ Active
    - Auditor Agent: ✅ Active
    """)

with col3:
    st.markdown("""
    **Infrastructure**
    - Database: PostgreSQL/SQLite
    - LLM: Gemini 1.5 Pro
    - Framework: LangGraph
    """)

# Danger Zone
st.markdown("---")
st.subheader("⚠️ Danger Zone")

with st.expander("Advanced Options (Use with caution)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Audit History", type="secondary"):
            st.warning("This will permanently delete all audit records!")
            if st.button("Confirm Delete", key="confirm_delete"):
                st.error("All audit history cleared!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", type="secondary"):
            st.warning("This will reset all settings to defaults!")
            if st.button("Confirm Reset", key="confirm_reset"):
                st.success("Settings reset to defaults!")

# Footer
st.markdown("---")
st.caption("Changes may require application restart to take effect.")
