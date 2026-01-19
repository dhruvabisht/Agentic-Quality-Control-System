"""
Auditor Agent for Sentinel-AI

Performs Chain-of-Thought reasoning to provide final verdict.
Synthesizes policy analysis and cultural context into actionable decisions.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .policy_agent import PolicyAnalysisResult
from .hindi_cultural_agent import HindiAnalysisResult, SentimentType


class Verdict(str, Enum):
    """Possible audit verdicts"""
    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"


@dataclass
class ChainOfThoughtStep:
    """A single step in the reasoning chain"""
    step_number: int
    reasoning_type: str  # "observation", "analysis", "inference", "decision"
    content: str
    confidence_delta: float = 0.0  # How this step affects confidence


@dataclass
class AuditResult:
    """Complete audit result with Chain-of-Thought reasoning"""
    content_id: str
    content_text: str
    verdict: Verdict
    confidence_score: float  # 0-100
    chain_of_thought: List[ChainOfThoughtStep]
    violation_summary: str
    detailed_explanation: str
    requires_human_review: bool
    is_sensitive: bool
    sensitivity_category: Optional[str]
    policy_violations: List[Dict[str, Any]]
    recommended_action: str
    processing_time_ms: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "content_text": self.content_text[:200] + "..." if len(self.content_text) > 200 else self.content_text,
            "verdict": self.verdict.value,
            "confidence_score": self.confidence_score,
            "chain_of_thought": [
                {
                    "step": s.step_number,
                    "type": s.reasoning_type,
                    "content": s.content,
                    "confidence_delta": s.confidence_delta
                }
                for s in self.chain_of_thought
            ],
            "violation_summary": self.violation_summary,
            "detailed_explanation": self.detailed_explanation,
            "requires_human_review": self.requires_human_review,
            "is_sensitive": self.is_sensitive,
            "sensitivity_category": self.sensitivity_category,
            "policy_violations": self.policy_violations,
            "recommended_action": self.recommended_action,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp
        }
    
    def get_chain_of_thought_text(self) -> str:
        """Get the full chain of thought as readable text."""
        lines = ["## Chain of Thought Reasoning\n"]
        for step in self.chain_of_thought:
            lines.append(f"**Step {step.step_number} ({step.reasoning_type}):**")
            lines.append(f"{step.content}")
            if step.confidence_delta != 0:
                lines.append(f"_Confidence adjustment: {step.confidence_delta:+.1f}%_")
            lines.append("")
        return "\n".join(lines)


class AuditorAgent:
    """
    Auditor Agent: Performs Chain-of-Thought reasoning for final verdict.
    
    Responsibilities:
    - Synthesize policy and cultural analysis
    - Apply Chain-of-Thought reasoning
    - Generate final verdict (Pass/Fail/Escalate)
    - Calculate confidence score
    - Provide detailed violation explanation
    - Apply sensitivity filtering
    """
    
    # Confidence thresholds
    DEFAULT_ESCALATION_THRESHOLD = 70
    DEFAULT_HIGH_CONFIDENCE = 85
    
    # Severity to base confidence mapping
    SEVERITY_BASE_CONFIDENCE = {
        "critical": 90,
        "high": 85,
        "medium": 75,
        "low": 65,
        "none": 50
    }
    
    # Actions by severity
    SEVERITY_ACTIONS = {
        "critical": "Immediate removal required",
        "high": "Content should be removed",
        "medium": "Content should be reviewed and potentially removed",
        "low": "Warning or reduced visibility",
        "none": "No action required"
    }
    
    def __init__(
        self, 
        llm_provider=None,
        escalation_threshold: int = None,
        sensitivity_filter=None
    ):
        """
        Initialize the Auditor Agent.
        
        Args:
            llm_provider: Optional LLMProvider for enhanced reasoning
            escalation_threshold: Confidence threshold for escalation
            sensitivity_filter: Optional SensitivityFilter instance
        """
        self.llm = llm_provider
        self.escalation_threshold = escalation_threshold or self.DEFAULT_ESCALATION_THRESHOLD
        self.sensitivity_filter = sensitivity_filter
    
    def _create_observation_step(
        self, 
        step_num: int, 
        policy_result: PolicyAnalysisResult,
        hindi_result: Optional[HindiAnalysisResult]
    ) -> ChainOfThoughtStep:
        """Create the initial observation step."""
        observations = []
        
        if policy_result.total_matches > 0:
            observations.append(
                f"Policy Analysis: Found {policy_result.total_matches} potential violations "
                f"with highest severity '{policy_result.highest_severity}'."
            )
            policies = set(m.policy_name for m in policy_result.matched_policies)
            observations.append(f"Affected policies: {', '.join(policies)}")
        else:
            observations.append("Policy Analysis: No keyword-based policy violations detected.")
        
        if hindi_result:
            if hindi_result.is_hindi or hindi_result.is_mixed_language:
                observations.append(
                    f"Language: {hindi_result.dialect.value} content detected. "
                    f"Sentiment: {hindi_result.sentiment.value}."
                )
            if hindi_result.cultural_contexts:
                contexts = [c.context_type for c in hindi_result.cultural_contexts]
                observations.append(f"Cultural sensitivities detected: {', '.join(contexts)}")
            if hindi_result.slang_detected:
                observations.append(f"Slang terms found: {len(hindi_result.slang_detected)}")
        
        return ChainOfThoughtStep(
            step_number=step_num,
            reasoning_type="observation",
            content=" ".join(observations),
            confidence_delta=0.0
        )
    
    def _create_analysis_step(
        self,
        step_num: int,
        policy_result: PolicyAnalysisResult,
        hindi_result: Optional[HindiAnalysisResult]
    ) -> ChainOfThoughtStep:
        """Create the analysis step with reasoning."""
        analysis_points = []
        confidence_delta = 0.0
        
        if policy_result.matched_policies:
            # Analyze each violation
            for match in policy_result.matched_policies:
                analysis_points.append(
                    f"- {match.policy_name}/{match.rule_name}: "
                    f"Matched keywords {match.matched_keywords}. "
                    f"This indicates potential {match.description.lower()}"
                )
                
                # Adjust confidence based on severity
                if match.severity == "critical":
                    confidence_delta += 15
                elif match.severity == "high":
                    confidence_delta += 10
                elif match.severity == "medium":
                    confidence_delta += 5
        
        if hindi_result and hindi_result.requires_special_attention:
            analysis_points.append(
                "- Hindi Cultural Analysis flags this content for special attention "
                "due to sensitive cultural context or hostile sentiment."
            )
            confidence_delta += 10
            
            if hindi_result.sentiment in [SentimentType.HOSTILE, SentimentType.THREATENING]:
                analysis_points.append(
                    f"- Sentiment analysis indicates {hindi_result.sentiment.value} tone, "
                    "which increases violation likelihood."
                )
                confidence_delta += 5
        
        if not analysis_points:
            analysis_points.append(
                "- No significant policy violations or cultural concerns identified."
            )
            confidence_delta -= 10
        
        return ChainOfThoughtStep(
            step_number=step_num,
            reasoning_type="analysis",
            content="\n".join(analysis_points),
            confidence_delta=confidence_delta
        )
    
    def _create_inference_step(
        self,
        step_num: int,
        policy_result: PolicyAnalysisResult,
        hindi_result: Optional[HindiAnalysisResult],
        current_confidence: float
    ) -> ChainOfThoughtStep:
        """Create the inference step."""
        inferences = []
        confidence_delta = 0.0
        
        # Check for multiple violation types (compounds severity)
        policy_types = set(m.policy_id for m in policy_result.matched_policies)
        if len(policy_types) > 1:
            inferences.append(
                f"Multiple policy categories affected ({len(policy_types)}), "
                "indicating compound violation."
            )
            confidence_delta += 5
        
        # Check for regional/cultural amplification
        regional_violations = [m for m in policy_result.matched_policies if m.is_regional]
        if regional_violations and hindi_result and hindi_result.is_hindi:
            inferences.append(
                "Regional violation in Hindi content - cultural context increases severity."
            )
            confidence_delta += 5
        
        # Check keyword density
        if len(policy_result.raw_keywords_found) > 3:
            inferences.append(
                f"High density of violation indicators ({len(policy_result.raw_keywords_found)} keywords)."
            )
            confidence_delta += 5
        
        # Consider context mitigation
        if hindi_result and hindi_result.sentiment == SentimentType.POSITIVE:
            if policy_result.total_matches < 2:
                inferences.append(
                    "Positive sentiment context may mitigate apparent violations - "
                    "could be ironic or quoted content."
                )
                confidence_delta -= 10
        
        if not inferences:
            inferences.append("Standard analysis applies without additional contextual factors.")
        
        return ChainOfThoughtStep(
            step_number=step_num,
            reasoning_type="inference",
            content=" ".join(inferences),
            confidence_delta=confidence_delta
        )
    
    def _create_decision_step(
        self,
        step_num: int,
        verdict: Verdict,
        confidence: float,
        reasoning: str
    ) -> ChainOfThoughtStep:
        """Create the final decision step."""
        return ChainOfThoughtStep(
            step_number=step_num,
            reasoning_type="decision",
            content=f"**Verdict: {verdict.value.upper()}** (Confidence: {confidence:.1f}%)\n{reasoning}",
            confidence_delta=0.0
        )
    
    def _determine_verdict(
        self,
        policy_result: PolicyAnalysisResult,
        hindi_result: Optional[HindiAnalysisResult],
        confidence: float
    ) -> tuple:
        """Determine the final verdict based on analysis."""
        
        # No violations = Pass
        if policy_result.total_matches == 0:
            if hindi_result and hindi_result.requires_special_attention:
                # Cultural concern without policy violation - Escalate
                return Verdict.ESCALATE, "Cultural sensitivity detected without clear policy violation. Human review recommended."
            return Verdict.PASS, "No policy violations detected. Content is compliant."
        
        # Check severity
        severity = policy_result.highest_severity
        
        # Critical or High severity with good confidence = Fail
        if severity in ["critical", "high"] and confidence >= 60:
            violation_names = ", ".join(set(m.policy_name for m in policy_result.matched_policies))
            return Verdict.FAIL, f"Clear violation of {violation_names} with {severity} severity."
        
        # Low confidence = Escalate
        if confidence < self.escalation_threshold:
            return Verdict.ESCALATE, f"Confidence ({confidence:.1f}%) below threshold ({self.escalation_threshold}%). Human review required."
        
        # Medium severity with good confidence = Fail
        if severity == "medium" and confidence >= 70:
            violation_names = ", ".join(set(m.policy_name for m in policy_result.matched_policies))
            return Verdict.FAIL, f"Violation of {violation_names} detected with medium severity."
        
        # Low severity = usually Pass with warning, but check cultural context
        if severity == "low":
            if hindi_result and hindi_result.requires_special_attention:
                return Verdict.ESCALATE, "Low severity violation combined with cultural sensitivity. Review recommended."
            return Verdict.PASS, "Only low-severity concerns detected. Content marginally compliant."
        
        # Default to Escalate for uncertain cases
        return Verdict.ESCALATE, "Analysis inconclusive. Human judgment required."
    
    def _check_sensitivity(self, content: str, policy_result: PolicyAnalysisResult) -> tuple:
        """Check if content is sensitive and needs filtering."""
        is_sensitive = False
        category = None
        
        # Check for graphic violence
        violence_policies = [m for m in policy_result.matched_policies if "violence" in m.policy_id.lower()]
        if violence_policies:
            is_sensitive = True
            category = "graphic_violence"
        
        # Check for adult content
        adult_policies = [m for m in policy_result.matched_policies if "adult" in m.policy_id.lower()]
        if adult_policies:
            is_sensitive = True
            category = "adult_content"
        
        # Use sensitivity filter if available
        if self.sensitivity_filter:
            filter_result = self.sensitivity_filter.check(content)
            if filter_result.get("is_sensitive"):
                is_sensitive = True
                category = filter_result.get("category", category)
        
        return is_sensitive, category
    
    def _generate_violation_summary(self, policy_result: PolicyAnalysisResult) -> str:
        """Generate a concise violation summary."""
        if not policy_result.matched_policies:
            return "No violations detected."
        
        policies = set(m.policy_name for m in policy_result.matched_policies)
        return f"Violations: {', '.join(policies)}. Severity: {policy_result.highest_severity}."
    
    def _generate_detailed_explanation(
        self,
        policy_result: PolicyAnalysisResult,
        hindi_result: Optional[HindiAnalysisResult],
        verdict: Verdict
    ) -> str:
        """Generate detailed explanation of the audit decision."""
        parts = []
        
        parts.append(f"## Audit Decision: {verdict.value.upper()}\n")
        
        if policy_result.matched_policies:
            parts.append("### Policy Violations Detected:\n")
            for match in policy_result.matched_policies:
                parts.append(f"- **{match.policy_name}** ({match.rule_name})")
                parts.append(f"  - Severity: {match.severity}")
                parts.append(f"  - Matched content: {', '.join(match.matched_keywords)}")
                parts.append(f"  - Description: {match.description}")
                parts.append("")
        else:
            parts.append("### No Policy Violations Detected\n")
        
        if hindi_result and (hindi_result.is_hindi or hindi_result.is_mixed_language):
            parts.append("### Cultural Analysis (Hindi Content):\n")
            parts.append(f"- Language: {hindi_result.dialect.value}")
            parts.append(f"- Sentiment: {hindi_result.sentiment.value} (score: {hindi_result.sentiment_score:.2f})")
            if hindi_result.cultural_contexts:
                parts.append("- Cultural sensitivities:")
                for ctx in hindi_result.cultural_contexts:
                    parts.append(f"  - {ctx.context_type}: {ctx.description}")
            if hindi_result.slang_detected:
                parts.append(f"- Slang detected: {', '.join(hindi_result.slang_detected)}")
            parts.append("")
        
        return "\n".join(parts)
    
    def audit(
        self,
        content: str,
        content_id: str,
        policy_result: PolicyAnalysisResult,
        hindi_result: Optional[HindiAnalysisResult] = None
    ) -> AuditResult:
        """
        Perform complete audit with Chain-of-Thought reasoning.
        
        Args:
            content: The original content text
            content_id: Identifier for the content
            policy_result: Result from PolicyAgent
            hindi_result: Optional result from HindiCulturalAgent
            
        Returns:
            Complete AuditResult with verdict and reasoning
        """
        import time
        start_time = time.time()
        
        chain_of_thought = []
        step_num = 1
        
        # Step 1: Observation
        observation = self._create_observation_step(step_num, policy_result, hindi_result)
        chain_of_thought.append(observation)
        step_num += 1
        
        # Step 2: Analysis
        analysis = self._create_analysis_step(step_num, policy_result, hindi_result)
        chain_of_thought.append(analysis)
        step_num += 1
        
        # Calculate initial confidence
        base_confidence = self.SEVERITY_BASE_CONFIDENCE.get(
            policy_result.highest_severity, 50
        )
        
        if policy_result.total_matches == 0:
            base_confidence = 90  # High confidence in Pass for clean content
        
        # Apply confidence adjustments from steps
        confidence = base_confidence
        for step in chain_of_thought:
            confidence = min(100, max(0, confidence + step.confidence_delta))
        
        # Step 3: Inference
        inference = self._create_inference_step(step_num, policy_result, hindi_result, confidence)
        chain_of_thought.append(inference)
        confidence = min(100, max(0, confidence + inference.confidence_delta))
        step_num += 1
        
        # Determine verdict
        verdict, verdict_reasoning = self._determine_verdict(policy_result, hindi_result, confidence)
        
        # Step 4: Decision
        decision = self._create_decision_step(step_num, verdict, confidence, verdict_reasoning)
        chain_of_thought.append(decision)
        
        # Check sensitivity
        is_sensitive, sensitivity_category = self._check_sensitivity(content, policy_result)
        
        # Generate summaries
        violation_summary = self._generate_violation_summary(policy_result)
        detailed_explanation = self._generate_detailed_explanation(policy_result, hindi_result, verdict)
        
        # Determine if human review is needed
        requires_human = verdict == Verdict.ESCALATE or confidence < self.escalation_threshold
        
        # Get recommended action
        recommended_action = self.SEVERITY_ACTIONS.get(
            policy_result.highest_severity if verdict == Verdict.FAIL else "none",
            "Review and determine appropriate action"
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare policy violations for storage
        policy_violations = [
            {
                "policy_id": m.policy_id,
                "policy_name": m.policy_name,
                "rule_id": m.rule_id,
                "severity": m.severity,
                "matched_keywords": m.matched_keywords
            }
            for m in policy_result.matched_policies
        ]
        
        return AuditResult(
            content_id=content_id,
            content_text=content,
            verdict=verdict,
            confidence_score=confidence,
            chain_of_thought=chain_of_thought,
            violation_summary=violation_summary,
            detailed_explanation=detailed_explanation,
            requires_human_review=requires_human,
            is_sensitive=is_sensitive,
            sensitivity_category=sensitivity_category,
            policy_violations=policy_violations,
            recommended_action=recommended_action,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def audit_with_llm(
        self,
        content: str,
        content_id: str,
        policy_result: PolicyAnalysisResult,
        hindi_result: Optional[HindiAnalysisResult] = None
    ) -> AuditResult:
        """
        Perform audit with LLM-enhanced reasoning.
        Falls back to rule-based if LLM unavailable.
        """
        if not self.llm:
            return self.audit(content, content_id, policy_result, hindi_result)
        
        # Use LLM for enhanced reasoning
        # ... LLM enhancement logic would go here ...
        
        # For now, use rule-based as base
        return self.audit(content, content_id, policy_result, hindi_result)
