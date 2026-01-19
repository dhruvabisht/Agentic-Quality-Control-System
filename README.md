# 🛡️ Sentinel-AI

**Agentic Content Quality Audit System for Meta's Trust and Safety Operations**

Sentinel-AI is a full-stack multi-agent system that audits Hindi and English social media content against Community Standards and Advertiser Policies using advanced LLM-powered reasoning.

## 🌟 Features

- **Multi-Agent Workflow**: Policy Agent, Hindi Cultural Agent, and Auditor Agent working in concert
- **Chain-of-Thought Reasoning**: Transparent decision-making with detailed explanations
- **Hindi Language Support**: Regional nuances, slang, and cultural context analysis
- **Human-in-the-Loop Interface**: Streamlit dashboard for Quality Measurement Specialists
- **Comprehensive Audit Trail**: PostgreSQL-backed history with KPI tracking
- **Market Insight Reports**: Trend analysis for Hindi-language policy violations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Content Input (Hindi/English)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Policy Agent                            │
│              (Rule Retrieval from JSON DB)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Hindi Cultural Agent                        │
│           (Regional Nuance & Sentiment Analysis)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Auditor Agent                            │
│           (Chain-of-Thought Final Verdict)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                        │
│              (HITL Review & Override)                        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (or use SQLite fallback)
- Gemini API key or OpenAI API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd "Agentic Quality Control System"
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URL
   ```

5. **Initialize database:**
   ```bash
   python -c "from src.database.connection import init_db; init_db()"
   ```

6. **Launch dashboard:**
   ```bash
   streamlit run src/dashboard/app.py
   ```

## 📁 Project Structure

```
├── data/
│   ├── community_standards.json   # Policy rules database
│   └── sample_content.json        # Test content samples
├── src/
│   ├── agents/                    # Multi-agent system
│   │   ├── policy_agent.py
│   │   ├── hindi_cultural_agent.py
│   │   ├── auditor_agent.py
│   │   └── orchestrator.py
│   ├── database/                  # Data layer
│   │   ├── models.py
│   │   └── connection.py
│   ├── llm/                       # LLM integration
│   │   └── provider.py
│   ├── features/                  # Advanced features
│   │   ├── sensitivity_filter.py
│   │   └── market_insights.py
│   └── dashboard/                 # Streamlit UI
│       ├── app.py
│       └── pages/
└── tests/                         # Test suite
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📊 KPI Metrics Tracked

- Audit accuracy rate
- Escalation frequency
- Human override rate
- Average confidence score
- Violation category distribution
- Response time metrics

## 📄 License

Internal Use Only - Meta Trust & Safety Division

---

Built with ❤️ for Trust & Safety
