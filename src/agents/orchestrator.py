"""
Sentinel Orchestrator - LangGraph-based Agent Workflow

Coordinates the multi-agent workflow:
1. Policy Agent → Rule retrieval
2. Hindi Cultural Agent → Cultural analysis (if Hindi detected)
3. Auditor Agent → Chain-of-Thought verdict

Uses LangGraph for workflow orchestration and state management.
"""

import os
from typing import TypedDict, Optional, Annotated, Sequence
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .policy_agent import PolicyAgent, PolicyAnalysisResult
from .hindi_cultural_agent import HindiCulturalAgent, HindiAnalysisResult
from .auditor_agent import AuditorAgent, AuditResult, Verdict
from ..database.models import ContentAudit, PolicyViolation, VerdictType, ContentLanguage, ViolationCategory
from ..database.connection import get_session


# State definition for the workflow
class AuditState(TypedDict):
    """State passed through the agent workflow"""
    content_id: str
    content_text: str
    content_type: str
    
    # Agent outputs
    policy_result: Optional[dict]
    hindi_result: Optional[dict]
    audit_result: Optional[dict]
    
    # Workflow control
    is_hindi_content: bool
    workflow_complete: bool
    error: Optional[str]
    
    # Metadata
    start_time: str
    end_time: Optional[str]


class SentinelOrchestrator:
    """
    Main orchestrator for the Sentinel-AI multi-agent workflow.
    
    Coordinates:
    - PolicyAgent: Policy rule matching
    - HindiCulturalAgent: Cultural context analysis
    - AuditorAgent: Final verdict generation
    
    Workflow:
    1. Content → Policy Analysis
    2. Policy Analysis → Hindi Analysis (if Hindi detected)
    3. All Analysis → Auditor → Final Verdict
    4. Save to Database
    """
    
    def __init__(
        self,
        policy_path: Optional[str] = None,
        escalation_threshold: int = 70,
        enable_persistence: bool = True
    ):
        """
        Initialize the Sentinel Orchestrator.
        
        Args:
            policy_path: Path to community_standards.json
            escalation_threshold: Confidence threshold for escalation
            enable_persistence: Whether to save results to database
        """
        # Initialize agents
        self.policy_agent = PolicyAgent(policy_path)
        self.hindi_agent = HindiCulturalAgent()
        self.auditor_agent = AuditorAgent(escalation_threshold=escalation_threshold)
        
        self.enable_persistence = enable_persistence
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        print("✅ Sentinel Orchestrator initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph
        workflow = StateGraph(AuditState)
        
        # Add nodes
        workflow.add_node("policy_analysis", self._policy_analysis_node)
        workflow.add_node("hindi_analysis", self._hindi_analysis_node)
        workflow.add_node("auditor_decision", self._auditor_decision_node)
        workflow.add_node("save_results", self._save_results_node)
        
        # Define edges
        workflow.set_entry_point("policy_analysis")
        
        # Conditional edge: Hindi analysis only if Hindi content detected
        workflow.add_conditional_edges(
            "policy_analysis",
            self._route_after_policy,
            {
                "hindi_analysis": "hindi_analysis",
                "auditor_decision": "auditor_decision"
            }
        )
        
        workflow.add_edge("hindi_analysis", "auditor_decision")
        workflow.add_edge("auditor_decision", "save_results")
        workflow.add_edge("save_results", END)
        
        # Compile with memory checkpointer
        checkpointer = MemorySaver()
        compiled = workflow.compile(checkpointer=checkpointer)
        
        return compiled
    
    def _route_after_policy(self, state: AuditState) -> str:
        """Route to Hindi analysis or directly to Auditor."""
        if state.get("is_hindi_content", False):
            return "hindi_analysis"
        return "auditor_decision"
    
    def _policy_analysis_node(self, state: AuditState) -> AuditState:
        """Node: Run policy analysis."""
        try:
            content = state["content_text"]
            content_id = state["content_id"]
            
            # Run policy analysis
            policy_result = self.policy_agent.analyze(content, content_id)
            
            # Check if content is Hindi
            hindi_check = self.hindi_agent.analyze(content, content_id)
            is_hindi = hindi_check.is_hindi or hindi_check.is_mixed_language
            
            return {
                **state,
                "policy_result": policy_result.to_dict(),
                "is_hindi_content": is_hindi,
                "error": None
            }
        except Exception as e:
            return {
                **state,
                "error": f"Policy analysis failed: {str(e)}"
            }
    
    def _hindi_analysis_node(self, state: AuditState) -> AuditState:
        """Node: Run Hindi cultural analysis."""
        try:
            content = state["content_text"]
            content_id = state["content_id"]
            
            # Run Hindi analysis
            hindi_result = self.hindi_agent.analyze(content, content_id)
            
            return {
                **state,
                "hindi_result": hindi_result.to_dict(),
                "error": None
            }
        except Exception as e:
            return {
                **state,
                "error": f"Hindi analysis failed: {str(e)}"
            }
    
    def _auditor_decision_node(self, state: AuditState) -> AuditState:
        """Node: Run auditor for final decision."""
        try:
            content = state["content_text"]
            content_id = state["content_id"]
            
            # Reconstruct results from state
            policy_result = self._reconstruct_policy_result(state.get("policy_result", {}))
            hindi_result = self._reconstruct_hindi_result(state.get("hindi_result"))
            
            # Run audit
            audit_result = self.auditor_agent.audit(
                content=content,
                content_id=content_id,
                policy_result=policy_result,
                hindi_result=hindi_result
            )
            
            return {
                **state,
                "audit_result": audit_result.to_dict(),
                "end_time": datetime.utcnow().isoformat(),
                "workflow_complete": True,
                "error": None
            }
        except Exception as e:
            return {
                **state,
                "error": f"Auditor decision failed: {str(e)}"
            }
    
    def _save_results_node(self, state: AuditState) -> AuditState:
        """Node: Save audit results to database."""
        if not self.enable_persistence:
            return state
        
        try:
            audit_data = state.get("audit_result", {})
            if not audit_data:
                return state
            
            # Check if database is configured
            import os
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                print("⚠️ DATABASE_URL not configured, skipping save")
                return state
            
            with get_session() as session:
                # Map verdict
                verdict_map = {
                    "pass": VerdictType.PASS,
                    "fail": VerdictType.FAIL,
                    "escalate": VerdictType.ESCALATE
                }
                
                # Determine language
                hindi_result = state.get("hindi_result") or {}
                if hindi_result.get("is_mixed_language"):
                    language = ContentLanguage.MIXED
                elif hindi_result.get("is_hindi"):
                    language = ContentLanguage.HINDI
                else:
                    language = ContentLanguage.ENGLISH
                
                # Create audit record
                audit = ContentAudit(
                    content_id=state["content_id"],
                    content_text=state["content_text"],
                    content_type=state.get("content_type", "post"),
                    language=language,
                    verdict=verdict_map.get(audit_data.get("verdict", "pass"), VerdictType.PASS),
                    confidence_score=audit_data.get("confidence_score", 0),
                    policy_agent_output=state.get("policy_result"),
                    hindi_cultural_agent_output=hindi_result if hindi_result else None,
                    auditor_agent_output=audit_data,
                    chain_of_thought=audit_data.get("detailed_explanation", ""),
                    is_sensitive=audit_data.get("is_sensitive", False),
                    sensitivity_category=audit_data.get("sensitivity_category"),
                    requires_human_review=audit_data.get("requires_human_review", False),
                    is_reviewed=False
                )
                session.add(audit)
                session.flush()  # Get the ID
                
                # Add violations
                for violation in audit_data.get("policy_violations", []):
                    # Map category
                    category_map = {
                        "hate_speech": ViolationCategory.HATE_SPEECH,
                        "graphic_violence": ViolationCategory.GRAPHIC_VIOLENCE,
                        "harassment": ViolationCategory.HARASSMENT,
                        "misinformation": ViolationCategory.MISINFORMATION,
                        "adult_content": ViolationCategory.ADULT_CONTENT,
                        "spam": ViolationCategory.SPAM,
                        "advertiser_policies": ViolationCategory.ADVERTISER_VIOLATION
                    }
                    
                    policy_id = violation.get("policy_id", "").lower()
                    category = ViolationCategory.OTHER
                    for key, cat in category_map.items():
                        if key in policy_id:
                            category = cat
                            break
                    
                    pv = PolicyViolation(
                        audit_id=audit.id,
                        category=category,
                        policy_name=violation.get("policy_name", "Unknown"),
                        policy_rule_id=violation.get("rule_id"),
                        severity=violation.get("severity", "medium"),
                        matched_keywords=violation.get("matched_keywords", []),
                        is_hindi_specific=state.get("is_hindi_content", False)
                    )
                    session.add(pv)
            
            return state
        except Exception as e:
            # Don't fail the workflow if save fails - just log and continue
            print(f"⚠️ Failed to save results (non-fatal): {e}")
            return state
    
    def _reconstruct_policy_result(self, data: dict) -> PolicyAnalysisResult:
        """Reconstruct PolicyAnalysisResult from dict."""
        from .policy_agent import PolicyMatch
        
        matches = []
        for m in data.get("matched_policies", []):
            matches.append(PolicyMatch(
                policy_id=m.get("policy_id", ""),
                policy_name=m.get("policy_name", ""),
                rule_id=m.get("rule_id", ""),
                rule_name=m.get("rule_name", ""),
                severity=m.get("severity", "medium"),
                matched_keywords=m.get("matched_keywords", []),
                description=m.get("description", ""),
                is_regional=m.get("is_regional", False)
            ))
        
        return PolicyAnalysisResult(
            content_id=data.get("content_id", ""),
            matched_policies=matches,
            highest_severity=data.get("highest_severity", "none"),
            total_matches=data.get("total_matches", 0),
            analysis_notes=data.get("analysis_notes", ""),
            raw_keywords_found=data.get("raw_keywords_found", [])
        )
    
    def _reconstruct_hindi_result(self, data: Optional[dict]) -> Optional[HindiAnalysisResult]:
        """Reconstruct HindiAnalysisResult from dict."""
        if not data:
            return None
        
        from .hindi_cultural_agent import CulturalContext, HindiDialect, SentimentType
        
        contexts = []
        for c in data.get("cultural_contexts", []):
            contexts.append(CulturalContext(
                context_type=c.get("type", ""),
                description=c.get("description", ""),
                sensitivity_level=c.get("sensitivity_level", "low"),
                related_policies=c.get("related_policies", [])
            ))
        
        try:
            dialect = HindiDialect(data.get("dialect", "hinglish"))
        except ValueError:
            dialect = HindiDialect.HINGLISH
        
        try:
            sentiment = SentimentType(data.get("sentiment", "neutral"))
        except ValueError:
            sentiment = SentimentType.NEUTRAL
        
        return HindiAnalysisResult(
            content_id=data.get("content_id", ""),
            is_hindi=data.get("is_hindi", False),
            is_mixed_language=data.get("is_mixed_language", False),
            dialect=dialect,
            detected_language_ratio=data.get("detected_language_ratio", {}),
            sentiment=sentiment,
            sentiment_score=data.get("sentiment_score", 0.0),
            cultural_contexts=contexts,
            slang_detected=data.get("slang_detected", []),
            transliteration_notes=data.get("transliteration_notes", ""),
            analysis_summary=data.get("analysis_summary", ""),
            requires_special_attention=data.get("requires_special_attention", False)
        )
    
    def audit_content(
        self,
        content: str,
        content_id: Optional[str] = None,
        content_type: str = "post"
    ) -> AuditResult:
        """
        Run the complete audit workflow on content.
        
        Args:
            content: The text content to audit
            content_id: Optional ID for the content (auto-generated if not provided)
            content_type: Type of content (post, comment, ad, story)
            
        Returns:
            AuditResult with verdict and full analysis
        """
        import uuid
        
        if not content_id:
            content_id = f"CONTENT-{uuid.uuid4().hex[:8].upper()}"
        
        # Initialize state
        initial_state: AuditState = {
            "content_id": content_id,
            "content_text": content,
            "content_type": content_type,
            "policy_result": None,
            "hindi_result": None,
            "audit_result": None,
            "is_hindi_content": False,
            "workflow_complete": False,
            "error": None,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None
        }
        
        # Run workflow
        config = {"configurable": {"thread_id": content_id}}
        final_state = self.workflow.invoke(initial_state, config)
        
        # Check for errors
        if final_state.get("error"):
            raise RuntimeError(f"Workflow error: {final_state['error']}")
        
        # Reconstruct and return audit result
        audit_data = final_state.get("audit_result", {})
        
        from .auditor_agent import ChainOfThoughtStep
        
        cot_steps = []
        for s in audit_data.get("chain_of_thought", []):
            cot_steps.append(ChainOfThoughtStep(
                step_number=s.get("step", 0),
                reasoning_type=s.get("type", ""),
                content=s.get("content", ""),
                confidence_delta=s.get("confidence_delta", 0.0)
            ))
        
        return AuditResult(
            content_id=audit_data.get("content_id", content_id),
            content_text=content,
            verdict=Verdict(audit_data.get("verdict", "pass")),
            confidence_score=audit_data.get("confidence_score", 0),
            chain_of_thought=cot_steps,
            violation_summary=audit_data.get("violation_summary", ""),
            detailed_explanation=audit_data.get("detailed_explanation", ""),
            requires_human_review=audit_data.get("requires_human_review", False),
            is_sensitive=audit_data.get("is_sensitive", False),
            sensitivity_category=audit_data.get("sensitivity_category"),
            policy_violations=audit_data.get("policy_violations", []),
            recommended_action=audit_data.get("recommended_action", ""),
            processing_time_ms=audit_data.get("processing_time_ms", 0),
            timestamp=audit_data.get("timestamp", datetime.utcnow().isoformat())
        )
    
    def batch_audit(
        self,
        contents: list,
        content_type: str = "post"
    ) -> list:
        """
        Audit multiple pieces of content.
        
        Args:
            contents: List of (content_id, content_text) tuples or just content strings
            content_type: Type of content
            
        Returns:
            List of AuditResults
        """
        results = []
        
        for item in contents:
            if isinstance(item, tuple):
                content_id, content_text = item
            else:
                content_id = None
                content_text = item
            
            try:
                result = self.audit_content(content_text, content_id, content_type)
                results.append(result)
            except Exception as e:
                print(f"⚠️ Failed to audit content: {e}")
                results.append(None)
        
        return results
    
    def get_workflow_visualization(self) -> str:
        """Get a Mermaid diagram of the workflow."""
        return """
```mermaid
stateDiagram-v2
    [*] --> PolicyAnalysis
    PolicyAnalysis --> HindiAnalysis: Hindi detected
    PolicyAnalysis --> AuditorDecision: English only
    HindiAnalysis --> AuditorDecision
    AuditorDecision --> SaveResults
    SaveResults --> [*]
    
    state PolicyAnalysis {
        [*] --> LoadPolicies
        LoadPolicies --> MatchKeywords
        MatchKeywords --> [*]
    }
    
    state HindiAnalysis {
        [*] --> DetectLanguage
        DetectLanguage --> AnalyzeSentiment
        AnalyzeSentiment --> FindCulturalContext
        FindCulturalContext --> [*]
    }
    
    state AuditorDecision {
        [*] --> Observation
        Observation --> Analysis
        Analysis --> Inference
        Inference --> Decision
        Decision --> [*]
    }
```
        """
