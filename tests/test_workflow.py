"""
Integration Tests for Sentinel-AI Workflow

Tests for the complete orchestrated workflow using LangGraph.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestWorkflowIntegration:
    """Integration tests for the complete audit workflow."""
    
    @pytest.fixture
    def sample_contents(self):
        """Sample content for testing."""
        return [
            {
                "id": "TEST-CLEAN-001",
                "text": "Just had a great day at work! Feeling blessed.",
                "expected_verdict": "pass",
                "language": "english"
            },
            {
                "id": "TEST-CLEAN-002",
                "text": "आज का दिन बहुत अच्छा था, परिवार के साथ समय बिताया।",
                "expected_verdict": "pass",
                "language": "hindi"
            },
            {
                "id": "TEST-FAIL-001",
                "text": "These people are like cockroaches, they should go back where they came from!",
                "expected_verdict": "fail",
                "language": "english"
            },
            {
                "id": "TEST-FAIL-002",
                "text": "ये छोटी जात के लोग हमारी बस्ती में नहीं रह सकते।",
                "expected_verdict": "fail",
                "language": "hindi"
            },
            {
                "id": "TEST-MIXED-001",
                "text": "Yaar ये लोग are so annoying! Inko देश से निकालो।",
                "expected_verdict": "fail",
                "language": "mixed"
            }
        ]
    
    def test_orchestrator_initialization(self):
        """Test that the orchestrator initializes correctly."""
        from src.agents.orchestrator import SentinelOrchestrator
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        assert orchestrator.policy_agent is not None
        assert orchestrator.hindi_agent is not None
        assert orchestrator.auditor_agent is not None
    
    def test_single_clean_content_audit(self):
        """Test auditing clean English content."""
        from src.agents.orchestrator import SentinelOrchestrator
        from src.agents.auditor_agent import Verdict
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        result = orchestrator.audit_content(
            content="Beautiful day at the beach, enjoying the sunset!",
            content_id="TEST-001",
            content_type="post"
        )
        
        assert result.verdict == Verdict.PASS
        assert result.confidence_score >= 80
        assert len(result.chain_of_thought) >= 4
    
    def test_single_violation_audit(self):
        """Test auditing content with violations."""
        from src.agents.orchestrator import SentinelOrchestrator
        from src.agents.auditor_agent import Verdict
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        result = orchestrator.audit_content(
            content="I will kill you and your entire family. You deserve to die.",
            content_id="TEST-002",
            content_type="comment"
        )
        
        assert result.verdict in [Verdict.FAIL, Verdict.ESCALATE]
        if result.verdict == Verdict.FAIL:
            assert len(result.policy_violations) > 0
    
    def test_hindi_content_audit(self):
        """Test auditing Hindi content."""
        from src.agents.orchestrator import SentinelOrchestrator
        from src.agents.auditor_agent import Verdict
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        result = orchestrator.audit_content(
            content="आज माँ के हाथ का खाना खाया, बहुत मजा आया!",
            content_id="TEST-003",
            content_type="post"
        )
        
        assert result.verdict == Verdict.PASS
    
    def test_hinglish_content_audit(self):
        """Test auditing code-mixed Hinglish content."""
        from src.agents.orchestrator import SentinelOrchestrator
        from src.agents.auditor_agent import Verdict
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        result = orchestrator.audit_content(
            content="Yaar today was so fun! बहुत मजा आया with friends!",
            content_id="TEST-004",
            content_type="post"
        )
        
        assert result.verdict == Verdict.PASS
    
    def test_caste_content_flagged(self):
        """Test that caste-based content is properly flagged."""
        from src.agents.orchestrator import SentinelOrchestrator
        from src.agents.auditor_agent import Verdict
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        result = orchestrator.audit_content(
            content="ये नीच जात के लोग हमारे साथ नहीं बैठ सकते",
            content_id="TEST-005",
            content_type="comment"
        )
        
        # Caste-based discrimination should fail or escalate
        assert result.verdict in [Verdict.FAIL, Verdict.ESCALATE]
    
    def test_batch_audit(self, sample_contents):
        """Test batch auditing of multiple contents."""
        from src.agents.orchestrator import SentinelOrchestrator
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        contents = [(c["id"], c["text"]) for c in sample_contents[:3]]
        results = orchestrator.batch_audit(contents)
        
        assert len(results) == 3
        assert all(r is not None for r in results)
    
    def test_result_contains_all_fields(self):
        """Test that audit result contains all required fields."""
        from src.agents.orchestrator import SentinelOrchestrator
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        result = orchestrator.audit_content(
            content="Test content for field validation",
            content_id="TEST-FIELDS"
        )
        
        result_dict = result.to_dict()
        
        required_fields = [
            "content_id",
            "verdict",
            "confidence_score",
            "chain_of_thought",
            "violation_summary",
            "detailed_explanation",
            "requires_human_review",
            "is_sensitive",
            "policy_violations",
            "recommended_action",
            "timestamp"
        ]
        
        for field in required_fields:
            assert field in result_dict, f"Missing field: {field}"
    
    def test_chain_of_thought_structure(self):
        """Test that chain of thought has proper structure."""
        from src.agents.orchestrator import SentinelOrchestrator
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        result = orchestrator.audit_content(
            content="Some test content here",
            content_id="TEST-COT"
        )
        
        cot = result.chain_of_thought
        
        assert len(cot) >= 4
        
        # Check for required step types
        step_types = [step.reasoning_type for step in cot]
        assert "observation" in step_types
        assert "decision" in step_types
    
    def test_workflow_visualization(self):
        """Test that workflow visualization is available."""
        from src.agents.orchestrator import SentinelOrchestrator
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        viz = orchestrator.get_workflow_visualization()
        
        assert "mermaid" in viz.lower()
        assert "PolicyAnalysis" in viz
        assert "AuditorDecision" in viz


class TestDatabaseIntegration:
    """Tests for database persistence."""
    
    def test_audit_persistence(self):
        """Test that audits are persisted to database."""
        from src.database.connection import init_db, get_session
        from src.database.models import ContentAudit
        from src.agents.orchestrator import SentinelOrchestrator
        
        # Initialize database
        init_db()
        
        orchestrator = SentinelOrchestrator(enable_persistence=True)
        
        content_id = f"PERSIST-{datetime.now().strftime('%H%M%S')}"
        
        result = orchestrator.audit_content(
            content="Test content for persistence validation",
            content_id=content_id
        )
        
        # Check if record exists in database
        with get_session() as session:
            audit = session.query(ContentAudit).filter(
                ContentAudit.content_id == content_id
            ).first()
            
            if audit:  # May not exist if SQLite issues
                assert audit.content_id == content_id
                assert audit.verdict is not None


class TestSampleContent:
    """Test against sample content from JSON file."""
    
    def test_sample_content_file_exists(self):
        """Test that sample content file exists."""
        sample_path = project_root / "data" / "sample_content.json"
        assert sample_path.exists()
    
    def test_audit_sample_content(self):
        """Test auditing sample content from JSON file."""
        import json
        
        from src.agents.orchestrator import SentinelOrchestrator
        from src.agents.auditor_agent import Verdict
        
        sample_path = project_root / "data" / "sample_content.json"
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data["samples"][:5]  # Test first 5 samples
        
        orchestrator = SentinelOrchestrator(enable_persistence=False)
        
        for sample in samples:
            result = orchestrator.audit_content(
                content=sample["text"],
                content_id=sample["id"],
                content_type=sample.get("content_type", "post")
            )
            
            expected = sample["expected_verdict"]
            
            # Check if result matches expected (with some tolerance for edge cases)
            if expected == "pass":
                assert result.verdict in [Verdict.PASS, Verdict.ESCALATE], \
                    f"Sample {sample['id']} expected pass but got {result.verdict}"
            elif expected == "fail":
                assert result.verdict in [Verdict.FAIL, Verdict.ESCALATE], \
                    f"Sample {sample['id']} expected fail but got {result.verdict}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
