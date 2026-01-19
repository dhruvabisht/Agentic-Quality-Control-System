"""
Policy Agent for Sentinel-AI

Retrieves and matches content against Community Standards rules.
Performs keyword matching, pattern detection, and policy identification.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PolicyMatch:
    """Represents a matched policy rule"""
    policy_id: str
    policy_name: str
    rule_id: str
    rule_name: str
    severity: str
    matched_keywords: List[str] = field(default_factory=list)
    description: str = ""
    is_regional: bool = False
    region: Optional[str] = None


@dataclass
class PolicyAnalysisResult:
    """Result of policy analysis on content"""
    content_id: str
    matched_policies: List[PolicyMatch]
    highest_severity: str
    total_matches: int
    analysis_notes: str
    raw_keywords_found: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "matched_policies": [
                {
                    "policy_id": m.policy_id,
                    "policy_name": m.policy_name,
                    "rule_id": m.rule_id,
                    "rule_name": m.rule_name,
                    "severity": m.severity,
                    "matched_keywords": m.matched_keywords,
                    "description": m.description,
                    "is_regional": m.is_regional
                }
                for m in self.matched_policies
            ],
            "highest_severity": self.highest_severity,
            "total_matches": self.total_matches,
            "analysis_notes": self.analysis_notes,
            "raw_keywords_found": self.raw_keywords_found
        }


class PolicyAgent:
    """
    Policy Agent: Retrieves and matches content against Community Standards.
    
    Responsibilities:
    - Load policy rules from JSON database
    - Perform keyword matching on content
    - Identify applicable policies and their severity
    - Support both English and Hindi keywords
    """
    
    SEVERITY_ORDER = ["low", "medium", "high", "critical"]
    
    def __init__(self, policy_path: Optional[str] = None):
        """
        Initialize the Policy Agent.
        
        Args:
            policy_path: Path to community_standards.json. 
                        Defaults to data/community_standards.json
        """
        if policy_path is None:
            # Default path relative to project root
            base_path = Path(__file__).parent.parent.parent
            policy_path = base_path / "data" / "community_standards.json"
        
        self.policy_path = Path(policy_path)
        self.policies: Dict[str, Any] = {}
        self.severity_levels: Dict[str, Any] = {}
        self._load_policies()
    
    def _load_policies(self) -> None:
        """Load policies from JSON file."""
        try:
            with open(self.policy_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.policies = data.get("policies", {})
            self.severity_levels = data.get("severity_levels", {})
            self.regional_considerations = data.get("regional_considerations", {})
            
            print(f"✅ Policy Agent loaded {len(self.policies)} policy categories")
        except FileNotFoundError:
            print(f"⚠️ Policy file not found: {self.policy_path}")
            self.policies = {}
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing policy file: {e}")
            self.policies = {}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching (lowercase, strip extra spaces)."""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def _extract_keywords_from_rule(self, rule: Dict) -> tuple:
        """Extract English and Hindi keywords from a rule."""
        keywords_en = rule.get("keywords_en", [])
        keywords_hi = rule.get("keywords_hi", [])
        patterns = rule.get("patterns", [])
        
        # Filter out redacted placeholders
        keywords_en = [k for k in keywords_en if not k.startswith("[REDACTED")]
        keywords_hi = [k for k in keywords_hi if not k.startswith("[REDACTED")]
        
        return keywords_en, keywords_hi, patterns
    
    def _match_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find keywords that match in the text."""
        normalized_text = self._normalize_text(text)
        matched = []
        
        for keyword in keywords:
            if not keyword or keyword.startswith("[REDACTED"):
                continue
            
            # Create pattern that matches whole words or phrases
            pattern = re.escape(keyword.lower())
            if re.search(pattern, normalized_text):
                matched.append(keyword)
        
        return matched
    
    def analyze(self, content: str, content_id: str = "unknown") -> PolicyAnalysisResult:
        """
        Analyze content against all policy rules.
        
        Args:
            content: The text content to analyze
            content_id: Identifier for the content
            
        Returns:
            PolicyAnalysisResult with all matched policies
        """
        matched_policies: List[PolicyMatch] = []
        all_keywords_found: List[str] = []
        highest_severity = "low"
        
        for policy_key, policy in self.policies.items():
            rules = policy.get("rules", [])
            
            for rule in rules:
                keywords_en, keywords_hi, patterns = self._extract_keywords_from_rule(rule)
                
                # Match English keywords
                matched_en = self._match_keywords(content, keywords_en)
                # Match Hindi keywords  
                matched_hi = self._match_keywords(content, keywords_hi)
                # Match patterns
                matched_patterns = self._match_keywords(content, patterns)
                
                all_matched = matched_en + matched_hi + matched_patterns
                
                if all_matched:
                    severity = rule.get("severity", "medium")
                    
                    match = PolicyMatch(
                        policy_id=policy.get("id", policy_key),
                        policy_name=policy.get("name", policy_key),
                        rule_id=rule.get("rule_id", "unknown"),
                        rule_name=rule.get("name", "Unknown Rule"),
                        severity=severity,
                        matched_keywords=all_matched,
                        description=rule.get("description", ""),
                        is_regional=rule.get("is_regional", False),
                        region=rule.get("region")
                    )
                    matched_policies.append(match)
                    all_keywords_found.extend(all_matched)
                    
                    # Update highest severity
                    if self._compare_severity(severity, highest_severity) > 0:
                        highest_severity = severity
        
        # Generate analysis notes
        if matched_policies:
            notes = f"Found {len(matched_policies)} policy matches across {len(set(m.policy_id for m in matched_policies))} categories."
            if any(m.is_regional for m in matched_policies):
                notes += " Includes region-specific violations."
        else:
            notes = "No policy violations detected through keyword matching."
        
        return PolicyAnalysisResult(
            content_id=content_id,
            matched_policies=matched_policies,
            highest_severity=highest_severity if matched_policies else "none",
            total_matches=len(matched_policies),
            analysis_notes=notes,
            raw_keywords_found=list(set(all_keywords_found))
        )
    
    def _compare_severity(self, sev1: str, sev2: str) -> int:
        """Compare two severity levels. Returns >0 if sev1 > sev2."""
        try:
            idx1 = self.SEVERITY_ORDER.index(sev1.lower())
            idx2 = self.SEVERITY_ORDER.index(sev2.lower())
            return idx1 - idx2
        except ValueError:
            return 0
    
    def get_policy_by_id(self, policy_id: str) -> Optional[Dict]:
        """Get a specific policy by its ID."""
        for policy_key, policy in self.policies.items():
            if policy.get("id") == policy_id or policy_key == policy_id:
                return policy
        return None
    
    def get_all_policy_names(self) -> List[str]:
        """Get list of all policy names."""
        return [p.get("name", k) for k, p in self.policies.items()]
    
    def get_severity_threshold(self, severity: str) -> int:
        """Get the confidence threshold for a severity level."""
        level = self.severity_levels.get(severity, {})
        return level.get("confidence_threshold", 70)
