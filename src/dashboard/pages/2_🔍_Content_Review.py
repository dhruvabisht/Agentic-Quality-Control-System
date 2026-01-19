"""
Content Review Page - Human-in-the-Loop Interface

Allows Quality Measurement Specialists to review escalated content,
override agent decisions, and provide feedback for model retraining.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Content Review | Sentinel-AI",
    page_icon="🔍",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .content-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .content-text {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        white-space: pre-wrap;
        margin: 1rem 0;
    }
    .cot-step {
        background: rgba(8, 102, 255, 0.1);
        border-left: 3px solid #0866FF;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .verdict-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .override-section {
        background: rgba(247, 185, 40, 0.1);
        border: 1px solid #F7B928;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Content Review")
st.markdown("Review escalated content and provide human judgment for complex cases.")

# Mock pending reviews data
if "pending_reviews" not in st.session_state:
    st.session_state.pending_reviews = [
        {
            "id": "ESC-001",
            "content_id": "CNT-8845",
            "content_text": "Yaar ये लोग are so annoying! Inko देश से बाहर निकालो, they don't belong here।",
            "language": "Hinglish",
            "content_type": "comment",
            "original_verdict": "escalate",
            "confidence": 68.2,
            "violations": ["Potential hate speech"],
            "chain_of_thought": [
                {"step": 1, "type": "observation", "content": "Policy Analysis: Found 1 potential violation with severity 'high'. Language: Hinglish content detected. Sentiment: hostile."},
                {"step": 2, "type": "analysis", "content": "- Hate Speech/Calls for Exclusion: Matched keywords ['निकालो', 'don't belong']. This indicates potential advocacy for exclusion."},
                {"step": 3, "type": "inference", "content": "Code-mixed content requires cultural context analysis. Hostile sentiment increases concern."},
                {"step": 4, "type": "decision", "content": "Confidence (68.2%) below threshold (70%). Human review required."}
            ],
            "cultural_context": "Content uses Hinglish (Hindi-English code-mixing). Contains exclusionary language targeting a group.",
            "created_at": "2026-01-19 18:15:00"
        },
        {
            "id": "ESC-002",
            "content_id": "CNT-8842",
            "content_text": "ये छोटी जात के लोग हमारी बस्ती में नहीं रह सकते। इनको यहाँ से भगाओ।",
            "language": "Hindi",
            "content_type": "post",
            "original_verdict": "escalate",
            "confidence": 72.5,
            "violations": ["Caste-based discrimination", "Hate speech"],
            "chain_of_thought": [
                {"step": 1, "type": "observation", "content": "Policy Analysis: Found 2 potential violations with severity 'critical'. Language: Hindi. Sentiment: hostile."},
                {"step": 2, "type": "analysis", "content": "- Caste-based Discrimination: Matched keywords ['छोटी जात']. This is treated as hate speech in Indian context."},
                {"step": 3, "type": "inference", "content": "Caste-based content in Hindi is highly sensitive. Critical severity requires confirmation."},
                {"step": 4, "type": "decision", "content": "Critical violation detected. Human confirmation required before removal."}
            ],
            "cultural_context": "Contains caste-based slurs which are illegal under Indian law. Highly offensive in cultural context.",
            "created_at": "2026-01-19 18:10:00"
        },
        {
            "id": "ESC-003",
            "content_id": "CNT-8839",
            "content_text": "BJP waalon को वोट दो नहीं तो देश बर्बाद हो जाएगा। Congress is destroying everything!",
            "language": "Hinglish",
            "content_type": "post",
            "original_verdict": "escalate",
            "confidence": 58.3,
            "violations": ["Political content - unclear violation"],
            "chain_of_thought": [
                {"step": 1, "type": "observation", "content": "Policy Analysis: Found 0 clear violations. Language: Hinglish. Sentiment: negative. Cultural sensitivity: politics."},
                {"step": 2, "type": "analysis", "content": "- Political content detected but no clear policy violation. Strong opinion but within bounds."},
                {"step": 3, "type": "inference", "content": "Political speech is protected. No hate speech or misinformation detected."},
                {"step": 4, "type": "decision", "content": "Low confidence (58.3%). Content may be political opinion rather than violation."}
            ],
            "cultural_context": "Political content referencing Indian political parties. Strong opinion but not necessarily violating.",
            "created_at": "2026-01-19 18:05:00"
        }
    ]

# Sidebar filters
with st.sidebar:
    st.subheader("🔧 Filters")
    
    language_filter = st.multiselect(
        "Language",
        ["All", "English", "Hindi", "Hinglish"],
        default=["All"]
    )
    
    severity_filter = st.multiselect(
        "Severity",
        ["All", "Critical", "High", "Medium", "Low"],
        default=["All"]
    )
    
    st.markdown("---")
    
    st.subheader("📊 Queue Stats")
    st.metric("Pending Reviews", len(st.session_state.pending_reviews))
    st.metric("Reviewed Today", 23)
    st.metric("Avg Review Time", "2.3 min")

# Main content area
tab1, tab2 = st.tabs(["📋 Pending Reviews", "✅ Completed Reviews"])

with tab1:
    if not st.session_state.pending_reviews:
        st.success("🎉 All caught up! No pending reviews.")
    else:
        st.info(f"📌 {len(st.session_state.pending_reviews)} items pending review")
        
        for i, review in enumerate(st.session_state.pending_reviews):
            with st.expander(
                f"**{review['id']}** | {review['content_id']} | {review['language']} | "
                f"Confidence: {review['confidence']}%",
                expanded=(i == 0)
            ):
                # Content display
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📝 Content")
                    st.markdown(f"""
                    <div class="content-text">
                    {review['content_text']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Type:** {review['content_type']} | **Language:** {review['language']}")
                
                with col2:
                    st.markdown("### ⚠️ Agent Analysis")
                    
                    # Original verdict
                    verdict_color = {
                        "pass": "#31A24C",
                        "fail": "#FA383E",
                        "escalate": "#F7B928"
                    }
                    st.markdown(f"""
                    <div style="background: {verdict_color[review['original_verdict']]}; 
                                padding: 0.5rem 1rem; border-radius: 20px; text-align: center; 
                                font-weight: 600; color: {'white' if review['original_verdict'] != 'escalate' else '#1a1a1a'};">
                        {review['original_verdict'].upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Confidence", f"{review['confidence']}%")
                    
                    st.markdown("**Detected Violations:**")
                    for v in review.get('violations', []):
                        st.markdown(f"- {v}")
                
                # Chain of Thought
                st.markdown("### 🧠 Chain of Thought Reasoning")
                for step in review.get('chain_of_thought', []):
                    st.markdown(f"""
                    <div class="cot-step">
                        <strong>Step {step['step']} ({step['type']}):</strong><br>
                        {step['content']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cultural Context
                if review.get('cultural_context'):
                    st.markdown("### 🌍 Cultural Context")
                    st.info(review['cultural_context'])
                
                # Human Decision Section
                st.markdown("---")
                st.markdown("### 👤 Human Review Decision")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    final_decision = st.radio(
                        "Your Decision:",
                        ["✅ Approve (Pass)", "❌ Reject (Fail)", "⏸️ Need More Info"],
                        key=f"decision_{review['id']}",
                        horizontal=True
                    )
                
                with col2:
                    override_reason = st.selectbox(
                        "Reason (if overriding):",
                        [
                            "Not applicable",
                            "Context misunderstood by AI",
                            "Cultural context changes meaning",
                            "Satire/Irony not detected",
                            "Policy interpretation differs",
                            "Edge case - needs policy update",
                            "Other"
                        ],
                        key=f"reason_{review['id']}"
                    )
                
                # Feedback
                feedback = st.text_area(
                    "Additional Notes (for model retraining):",
                    placeholder="Explain your reasoning, especially if overriding the AI decision...",
                    key=f"feedback_{review['id']}"
                )
                
                flag_for_training = st.checkbox(
                    "🏷️ Flag for model retraining",
                    key=f"training_{review['id']}"
                )
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("📤 Submit Review", key=f"submit_{review['id']}", type="primary"):
                        # Process the review
                        st.success(f"✅ Review submitted for {review['id']}")
                        # In production, this would save to database
                        st.session_state.pending_reviews.remove(review)
                        st.rerun()
                
                with col2:
                    if st.button("⏭️ Skip for Now", key=f"skip_{review['id']}"):
                        st.info("Skipped. Will appear at end of queue.")

with tab2:
    st.subheader("Recently Completed Reviews")
    
    completed_df = pd.DataFrame({
        "Review ID": ["ESC-098", "ESC-097", "ESC-096", "ESC-095"],
        "Content ID": ["CNT-8840", "CNT-8838", "CNT-8835", "CNT-8832"],
        "Original": ["Escalate", "Escalate", "Fail", "Escalate"],
        "Final": ["Pass", "Fail", "Fail", "Pass"],
        "Overridden": ["Yes", "No", "No", "Yes"],
        "Reviewed By": ["QMS-001", "QMS-002", "QMS-001", "QMS-003"],
        "Time": ["10 min ago", "25 min ago", "1 hour ago", "2 hours ago"]
    })
    
    st.dataframe(
        completed_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Overridden": st.column_config.TextColumn("Overridden", width="small")
        }
    )
    
    st.markdown("---")
    
    # Override statistics
    st.subheader("📊 Override Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reviews Today", "23")
    with col2:
        st.metric("Overrides", "7", "30.4%")
    with col3:
        st.metric("Flagged for Training", "4")

# Footer
st.markdown("---")
st.caption("Human reviews are logged for quality assurance and model improvement.")
