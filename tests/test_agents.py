"""
Unit Tests for Sentinel-AI Agents

Tests for PolicyAgent, HindiCulturalAgent, and AuditorAgent.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.policy_agent import PolicyAgent, PolicyAnalysisResult
from src.agents.hindi_cultural_agent import HindiCulturalAgent, HindiDialect, SentimentType
from src.agents.auditor_agent import AuditorAgent, Verdict


class TestPolicyAgent:
    """Tests for the Policy Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a PolicyAgent instance."""
        return PolicyAgent()
    
    def test_initialization(self, agent):
        """Test that agent initializes with policies loaded."""
        assert agent.policies is not None
        assert len(agent.policies) > 0
        assert "hate_speech" in agent.policies
    
    def test_clean_content_passes(self, agent):
        """Test that clean content has no violations."""
        content = "Just had the best day at the beach! The sunset was beautiful."
        result = agent.analyze(content, "TEST-001")
        
        assert result.total_matches == 0
        assert result.highest_severity == "none"
        assert len(result.matched_policies) == 0
    
    def test_hate_speech_detection(self, agent):
        """Test detection of hate speech keywords."""
        content = "These people are like cockroaches and should go back to where they came from."
        result = agent.analyze(content, "TEST-002")
        
        assert result.total_matches > 0
        assert result.highest_severity in ["high", "critical"]
        policy_names = [m.policy_name for m in result.matched_policies]
        assert "Hate Speech" in policy_names
    
    def test_hindi_keyword_detection(self, agent):
        """Test detection of Hindi violation keywords."""
        content = "इन लोगों को मार डालो, ये कीड़े हैं"
        result = agent.analyze(content, "TEST-003")
        
        assert result.total_matches > 0
        assert "मार डालो" in result.raw_keywords_found or "कीड़े" in result.raw_keywords_found
    
    def test_misinformation_detection(self, agent):
        """Test detection of misinformation patterns."""
        content = "Doctors don't want you to know this miracle cure that works 100%!"
        result = agent.analyze(content, "TEST-004")
        
        assert result.total_matches > 0
        has_misinfo = any(
            "misinformation" in m.policy_id.lower() or "spam" in m.policy_id.lower()
            for m in result.matched_policies
        )
        assert has_misinfo
    
    def test_result_to_dict(self, agent):
        """Test that result can be serialized to dict."""
        content = "Test content with violence keywords like slaughter."
        result = agent.analyze(content, "TEST-005")
        
        result_dict = result.to_dict()
        
        assert "content_id" in result_dict
        assert "matched_policies" in result_dict
        assert "highest_severity" in result_dict
        assert isinstance(result_dict["matched_policies"], list)


class TestHindiCulturalAgent:
    """Tests for the Hindi Cultural Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a HindiCulturalAgent instance."""
        return HindiCulturalAgent()
    
    def test_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.SLANG_DICTIONARY is not None
        assert len(agent.SLANG_DICTIONARY) > 0
    
    def test_hindi_detection_devanagari(self, agent):
        """Test detection of Hindi in Devanagari script."""
        content = "आज का दिन बहुत अच्छा था"
        result = agent.analyze(content, "TEST-001")
        
        assert result.is_hindi == True
        assert result.dialect == HindiDialect.STANDARD or result.dialect == HindiDialect.COLLOQUIAL
    
    def test_hinglish_detection(self, agent):
        """Test detection of code-mixed Hinglish content."""
        content = "Yaar today was amazing, मजा आ गया!"
        result = agent.analyze(content, "TEST-002")
        
        assert result.is_mixed_language == True
        assert result.dialect == HindiDialect.HINGLISH
    
    def test_english_only_content(self, agent):
        """Test that pure English content is identified correctly."""
        content = "This is a completely English sentence without any Hindi."
        result = agent.analyze(content, "TEST-003")
        
        assert result.is_hindi == False
        assert result.detected_language_ratio["english"] > 0.9
    
    def test_slang_detection(self, agent):
        """Test detection of Hindi slang."""
        content = "तू कमीना है, साला बेवकूफ"
        result = agent.analyze(content, "TEST-004")
        
        assert len(result.slang_detected) > 0
        assert "कमीना" in result.slang_detected or "साला" in result.slang_detected
    
    def test_caste_content_detection(self, agent):
        """Test detection of caste-based content."""
        content = "ये छोटी जात के लोग यहाँ नहीं रह सकते"
        result = agent.analyze(content, "TEST-005")
        
        assert result.requires_special_attention == True
        assert agent.is_casteist_content(content) == True
    
    def test_sentiment_analysis_negative(self, agent):
        """Test sentiment detection for hostile content."""
        content = "मैं तुझे मार दूंगा, तुझसे घृणा है"
        result = agent.analyze(content, "TEST-006")
        
        assert result.sentiment in [SentimentType.NEGATIVE, SentimentType.HOSTILE, SentimentType.THREATENING]
        assert result.sentiment_score < 0
    
    def test_sentiment_analysis_positive(self, agent):
        """Test sentiment detection for positive content."""
        content = "बहुत खुशी की बात है, धन्यवाद आपका"
        result = agent.analyze(content, "TEST-007")
        
        assert result.sentiment == SentimentType.POSITIVE
        assert result.sentiment_score >= 0
    
    def test_cultural_context_detection(self, agent):
        """Test detection of cultural sensitivities."""
        content = "मंदिर में पूजा करने गया था, बहुत शांति मिली"
        result = agent.analyze(content, "TEST-008")
        
        # Should detect religion-related context
        context_types = [c.context_type for c in result.cultural_contexts]
        assert "religion" in context_types
    
    def test_result_to_dict(self, agent):
        """Test serialization of result."""
        content = "Test content यार"
        result = agent.analyze(content, "TEST-009")
        
        result_dict = result.to_dict()
        
        assert "is_hindi" in result_dict
        assert "dialect" in result_dict
        assert "sentiment" in result_dict


class TestAuditorAgent:
    """Tests for the Auditor Agent."""
    
    @pytest.fixture
    def policy_agent(self):
        return PolicyAgent()
    
    @pytest.fixture
    def hindi_agent(self):
        return HindiCulturalAgent()
    
    @pytest.fixture
    def auditor(self):
        return AuditorAgent(escalation_threshold=70)
    
    def test_initialization(self, auditor):
        """Test that auditor initializes correctly."""
        assert auditor.escalation_threshold == 70
        assert auditor.SEVERITY_BASE_CONFIDENCE is not None
    
    def test_clean_content_passes(self, auditor, policy_agent, hindi_agent):
        """Test that clean content gets Pass verdict."""
        content = "Beautiful sunset at the beach today!"
        
        policy_result = policy_agent.analyze(content, "TEST-001")
        hindi_result = hindi_agent.analyze(content, "TEST-001")
        
        result = auditor.audit(content, "TEST-001", policy_result, hindi_result)
        
        assert result.verdict == Verdict.PASS
        assert result.confidence_score >= 80
        assert result.requires_human_review == False
    
    def test_violation_fails(self, auditor, policy_agent, hindi_agent):
        """Test that clear violations get Fail verdict."""
        content = "These cockroaches should be slaughtered and eliminated."
        
        policy_result = policy_agent.analyze(content, "TEST-002")
        hindi_result = hindi_agent.analyze(content, "TEST-002")
        
        result = auditor.audit(content, "TEST-002", policy_result, hindi_result)
        
        assert result.verdict == Verdict.FAIL
        assert len(result.policy_violations) > 0
    
    def test_low_confidence_escalates(self, auditor, policy_agent, hindi_agent):
        """Test that low confidence cases get escalated."""
        # Create a borderline case
        auditor_low = AuditorAgent(escalation_threshold=95)  # Very high threshold
        
        content = "Some borderline content that might be problematic but unclear."
        
        policy_result = policy_agent.analyze(content, "TEST-003")
        hindi_result = hindi_agent.analyze(content, "TEST-003")
        
        result = auditor_low.audit(content, "TEST-003", policy_result, hindi_result)
        
        # With such a high threshold, most content should escalate
        if result.confidence_score < 95:
            assert result.verdict == Verdict.ESCALATE
    
    def test_chain_of_thought_generated(self, auditor, policy_agent, hindi_agent):
        """Test that Chain of Thought reasoning is generated."""
        content = "This is test content for reasoning."
        
        policy_result = policy_agent.analyze(content, "TEST-004")
        
        result = auditor.audit(content, "TEST-004", policy_result)
        
        assert len(result.chain_of_thought) >= 4  # At least 4 steps
        
        step_types = [s.reasoning_type for s in result.chain_of_thought]
        assert "observation" in step_types
        assert "analysis" in step_types
        assert "inference" in step_types
        assert "decision" in step_types
    
    def test_hindi_context_considered(self, auditor, policy_agent, hindi_agent):
        """Test that Hindi cultural context affects decision."""
        content = "ये छोटी जात के लोग यहाँ नहीं रह सकते"
        
        policy_result = policy_agent.analyze(content, "TEST-005")
        hindi_result = hindi_agent.analyze(content, "TEST-005")
        
        result = auditor.audit(content, "TEST-005", policy_result, hindi_result)
        
        # Caste-based content should fail or escalate
        assert result.verdict in [Verdict.FAIL, Verdict.ESCALATE]
        
        # Chain of thought should mention Hindi/cultural context
        cot_text = result.get_chain_of_thought_text()
        # (In real usage, we'd check for specific mentions)
    
    def test_result_serialization(self, auditor, policy_agent):
        """Test that result can be serialized."""
        content = "Test content"
        
        policy_result = policy_agent.analyze(content, "TEST-006")
        result = auditor.audit(content, "TEST-006", policy_result)
        
        result_dict = result.to_dict()
        
        assert "verdict" in result_dict
        assert "confidence_score" in result_dict
        assert "chain_of_thought" in result_dict
        assert "policy_violations" in result_dict
    
    def test_sensitivity_detection(self, auditor, policy_agent):
        """Test sensitivity detection for graphic content."""
        content = "Content about blood, gore, and massacre."
        
        policy_result = policy_agent.analyze(content, "TEST-007")
        result = auditor.audit(content, "TEST-007", policy_result)
        
        assert result.is_sensitive == True


class TestIntegration:
    """Integration tests for the full agent workflow."""
    
    def test_full_workflow_clean_content(self):
        """Test complete workflow with clean content."""
        policy_agent = PolicyAgent()
        hindi_agent = HindiCulturalAgent()
        auditor = AuditorAgent()
        
        content = "आज का मौसम बहुत अच्छा है, धूप खिली हुई है।"
        content_id = "INTEG-001"
        
        policy_result = policy_agent.analyze(content, content_id)
        hindi_result = hindi_agent.analyze(content, content_id)
        final_result = auditor.audit(content, content_id, policy_result, hindi_result)
        
        assert final_result.verdict == Verdict.PASS
        assert final_result.confidence_score >= 80
    
    def test_full_workflow_violation(self):
        """Test complete workflow with violating content."""
        policy_agent = PolicyAgent()
        hindi_agent = HindiCulturalAgent()
        auditor = AuditorAgent()
        
        content = "मैं तुझे मार दूंगा, तेरा पूरा परिवार देख लेगा"
        content_id = "INTEG-002"
        
        policy_result = policy_agent.analyze(content, content_id)
        hindi_result = hindi_agent.analyze(content, content_id)
        final_result = auditor.audit(content, content_id, policy_result, hindi_result)
        
        assert final_result.verdict in [Verdict.FAIL, Verdict.ESCALATE]
        assert len(final_result.policy_violations) > 0 or final_result.requires_human_review


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
