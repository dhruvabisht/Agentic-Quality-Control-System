"""
Sensitivity Filter for Sentinel-AI

Filters and masks sensitive/graphic content for display.
Provides content warnings and safe viewing options.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class SensitivityCategory(str, Enum):
    """Categories of sensitive content"""
    GRAPHIC_VIOLENCE = "graphic_violence"
    ADULT_CONTENT = "adult_content"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    DISTURBING = "disturbing"
    NONE = "none"


@dataclass
class SensitivityResult:
    """Result of sensitivity check"""
    is_sensitive: bool
    category: SensitivityCategory
    severity: str  # low, medium, high, critical
    matched_patterns: List[str]
    warning_message: str
    masked_content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_sensitive": self.is_sensitive,
            "category": self.category.value,
            "severity": self.severity,
            "matched_patterns": self.matched_patterns,
            "warning_message": self.warning_message,
            "masked_content": self.masked_content
        }


class SensitivityFilter:
    """
    Sensitivity Filter for graphic content simulation.
    
    Features:
    - Keyword-based sensitivity detection
    - Configurable severity levels
    - Content masking for safe display
    - Content warnings generation
    """
    
    # Sensitivity patterns by category
    SENSITIVITY_PATTERNS = {
        SensitivityCategory.GRAPHIC_VIOLENCE: {
            "keywords_en": [
                "blood", "gore", "dismember", "decapitate", "mutilate",
                "kill", "murder", "slaughter", "massacre", "torture",
                "stab", "shoot", "execution", "beheading"
            ],
            "keywords_hi": [
                "खून", "हत्या", "कत्ल", "मार डालो", "काट डालो",
                "गोली", "तलवार", "जान से मार"
            ],
            "severity_keywords": {
                "critical": ["beheading", "dismember", "execution", "gore"],
                "high": ["murder", "massacre", "torture", "slaughter"],
                "medium": ["kill", "stab", "shoot", "blood"]
            }
        },
        SensitivityCategory.ADULT_CONTENT: {
            "keywords_en": [
                # Redacted for appropriateness
            ],
            "keywords_hi": [
                # Redacted for appropriateness
            ],
            "severity_keywords": {
                "critical": [],
                "high": [],
                "medium": []
            }
        },
        SensitivityCategory.SELF_HARM: {
            "keywords_en": [
                "kill myself", "suicide", "end my life", "cut myself",
                "self-harm", "want to die", "no reason to live"
            ],
            "keywords_hi": [
                "आत्महत्या", "खुदकुशी", "मर जाना चाहता", "जीना नहीं चाहता"
            ],
            "severity_keywords": {
                "critical": ["suicide method", "kill myself tonight"],
                "high": ["kill myself", "end my life"],
                "medium": ["want to die", "self-harm"]
            }
        },
        SensitivityCategory.DISTURBING: {
            "keywords_en": [
                "corpse", "dead body", "accident scene", "graphic injury",
                "severe trauma", "disfigured"
            ],
            "keywords_hi": [
                "लाश", "मृत शरीर", "घायल", "दुर्घटना"
            ],
            "severity_keywords": {
                "high": ["corpse", "disfigured"],
                "medium": ["dead body", "accident scene"]
            }
        }
    }
    
    # Warning messages by category
    WARNING_MESSAGES = {
        SensitivityCategory.GRAPHIC_VIOLENCE: "⚠️ This content contains graphic violence or violent imagery.",
        SensitivityCategory.ADULT_CONTENT: "🔞 This content may contain adult or sexually explicit material.",
        SensitivityCategory.SELF_HARM: "⚠️ This content discusses self-harm or suicide. If you're struggling, please seek help.",
        SensitivityCategory.DISTURBING: "⚠️ This content may contain disturbing imagery.",
        SensitivityCategory.HATE_SPEECH: "⚠️ This content contains hate speech or discriminatory language.",
        SensitivityCategory.NONE: ""
    }
    
    def __init__(self, enabled: bool = True, mask_character: str = "█"):
        """
        Initialize the Sensitivity Filter.
        
        Args:
            enabled: Whether filtering is enabled
            mask_character: Character used to mask sensitive content
        """
        self.enabled = enabled
        self.mask_character = mask_character
    
    def _find_matches(self, text: str, keywords: List[str]) -> List[str]:
        """Find matching keywords in text."""
        text_lower = text.lower()
        matches = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches.append(keyword)
        
        return matches
    
    def _determine_severity(
        self, 
        matches: List[str], 
        severity_keywords: Dict[str, List[str]]
    ) -> str:
        """Determine severity based on matched keywords."""
        matches_lower = [m.lower() for m in matches]
        
        for severity in ["critical", "high", "medium", "low"]:
            severity_list = severity_keywords.get(severity, [])
            for kw in severity_list:
                if kw.lower() in matches_lower:
                    return severity
        
        return "low" if matches else "none"
    
    def _mask_content(self, text: str, keywords: List[str]) -> str:
        """Mask sensitive keywords in content."""
        masked = text
        
        for keyword in keywords:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            replacement = self.mask_character * len(keyword)
            masked = pattern.sub(replacement, masked)
        
        return masked
    
    def check(self, content: str) -> SensitivityResult:
        """
        Check content for sensitivity.
        
        Args:
            content: The text content to check
            
        Returns:
            SensitivityResult with detection details
        """
        if not self.enabled:
            return SensitivityResult(
                is_sensitive=False,
                category=SensitivityCategory.NONE,
                severity="none",
                matched_patterns=[],
                warning_message=""
            )
        
        all_matches = []
        detected_category = SensitivityCategory.NONE
        highest_severity = "none"
        severity_order = ["none", "low", "medium", "high", "critical"]
        
        # Check each category
        for category, patterns in self.SENSITIVITY_PATTERNS.items():
            keywords_en = patterns.get("keywords_en", [])
            keywords_hi = patterns.get("keywords_hi", [])
            severity_keywords = patterns.get("severity_keywords", {})
            
            matches = self._find_matches(content, keywords_en + keywords_hi)
            
            if matches:
                all_matches.extend(matches)
                severity = self._determine_severity(matches, severity_keywords)
                
                # Update if higher severity
                if severity_order.index(severity) > severity_order.index(highest_severity):
                    highest_severity = severity
                    detected_category = category
        
        # Generate result
        is_sensitive = len(all_matches) > 0
        warning = self.WARNING_MESSAGES.get(detected_category, "")
        
        masked_content = None
        if is_sensitive and all_matches:
            masked_content = self._mask_content(content, all_matches)
        
        return SensitivityResult(
            is_sensitive=is_sensitive,
            category=detected_category,
            severity=highest_severity,
            matched_patterns=list(set(all_matches)),
            warning_message=warning,
            masked_content=masked_content
        )
    
    def get_safe_display(self, content: str, show_warning: bool = True) -> Dict[str, Any]:
        """
        Get content safe for display with appropriate warnings.
        
        Args:
            content: The original content
            show_warning: Whether to include warning message
            
        Returns:
            Dictionary with display content and metadata
        """
        result = self.check(content)
        
        if not result.is_sensitive:
            return {
                "content": content,
                "is_masked": False,
                "warning": None,
                "can_reveal": False
            }
        
        return {
            "content": result.masked_content or content,
            "is_masked": True,
            "warning": result.warning_message if show_warning else None,
            "can_reveal": True,
            "original_length": len(content),
            "severity": result.severity,
            "category": result.category.value
        }
    
    def add_custom_keywords(
        self, 
        category: SensitivityCategory, 
        keywords: List[str],
        severity: str = "medium"
    ) -> None:
        """
        Add custom keywords to a sensitivity category.
        
        Args:
            category: The category to add keywords to
            keywords: List of keywords to add
            severity: Severity level for these keywords
        """
        if category not in self.SENSITIVITY_PATTERNS:
            self.SENSITIVITY_PATTERNS[category] = {
                "keywords_en": [],
                "keywords_hi": [],
                "severity_keywords": {}
            }
        
        # Add to English keywords (can be extended for Hindi)
        self.SENSITIVITY_PATTERNS[category]["keywords_en"].extend(keywords)
        
        # Add to severity keywords
        if severity not in self.SENSITIVITY_PATTERNS[category]["severity_keywords"]:
            self.SENSITIVITY_PATTERNS[category]["severity_keywords"][severity] = []
        self.SENSITIVITY_PATTERNS[category]["severity_keywords"][severity].extend(keywords)
