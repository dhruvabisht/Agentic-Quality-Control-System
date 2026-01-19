"""
Hindi Cultural Agent for Sentinel-AI

Analyzes content for regional nuances, slang, and context specific to the Indian market.
Provides high-accuracy Hindi sentiment and policy analysis.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class HindiDialect(str, Enum):
    """Hindi dialect/variation types"""
    STANDARD = "standard_hindi"
    HINGLISH = "hinglish"  # Hindi-English code-mixing
    COLLOQUIAL = "colloquial"
    REGIONAL = "regional"


class SentimentType(str, Enum):
    """Sentiment categories"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    HOSTILE = "hostile"
    THREATENING = "threatening"


@dataclass
class CulturalContext:
    """Cultural context detected in content"""
    context_type: str
    description: str
    sensitivity_level: str  # low, medium, high
    related_policies: List[str] = field(default_factory=list)


@dataclass
class HindiAnalysisResult:
    """Result of Hindi cultural analysis"""
    content_id: str
    is_hindi: bool
    is_mixed_language: bool
    dialect: HindiDialect
    detected_language_ratio: Dict[str, float]  # {"hindi": 0.6, "english": 0.4}
    sentiment: SentimentType
    sentiment_score: float  # -1.0 to 1.0
    cultural_contexts: List[CulturalContext]
    slang_detected: List[str]
    transliteration_notes: str
    analysis_summary: str
    requires_special_attention: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "is_hindi": self.is_hindi,
            "is_mixed_language": self.is_mixed_language,
            "dialect": self.dialect.value,
            "detected_language_ratio": self.detected_language_ratio,
            "sentiment": self.sentiment.value,
            "sentiment_score": self.sentiment_score,
            "cultural_contexts": [
                {
                    "type": c.context_type,
                    "description": c.description,
                    "sensitivity_level": c.sensitivity_level,
                    "related_policies": c.related_policies
                }
                for c in self.cultural_contexts
            ],
            "slang_detected": self.slang_detected,
            "transliteration_notes": self.transliteration_notes,
            "analysis_summary": self.analysis_summary,
            "requires_special_attention": self.requires_special_attention
        }


class HindiCulturalAgent:
    """
    Hindi Cultural Agent: Analyzes content for regional and cultural nuances.
    
    Responsibilities:
    - Detect Hindi content and code-mixing patterns
    - Analyze regional slang and colloquialisms
    - Identify cultural context and sentiment
    - Flag region-specific sensitivities
    - Provide transliteration support
    """
    
    # Common Hindi slang and colloquialisms (with meanings)
    SLANG_DICTIONARY = {
        # Negative/Offensive slang
        "कमीना": {"meaning": "bastard/scoundrel", "severity": "medium"},
        "हरामी": {"meaning": "illegitimate/scoundrel", "severity": "medium"},
        "साला": {"meaning": "brother-in-law (derogatory)", "severity": "low"},
        "बेवकूफ": {"meaning": "fool/idiot", "severity": "low"},
        "गधा": {"meaning": "donkey (calling someone stupid)", "severity": "low"},
        "उल्लू": {"meaning": "owl (calling someone fool)", "severity": "low"},
        "नालायक": {"meaning": "worthless/useless", "severity": "medium"},
        "चुटिया": {"meaning": "offensive term", "severity": "high"},
        "भड़वा": {"meaning": "offensive term", "severity": "high"},
        "रंडी": {"meaning": "offensive term", "severity": "high"},
        
        # Caste-related slurs (highly sensitive)
        "चमार": {"meaning": "caste slur", "severity": "critical", "is_casteist": True},
        "भंगी": {"meaning": "caste slur", "severity": "critical", "is_casteist": True},
        "छोटी जात": {"meaning": "lower caste (derogatory)", "severity": "critical", "is_casteist": True},
        "नीच जात": {"meaning": "lower caste (derogatory)", "severity": "critical", "is_casteist": True},
        "अछूत": {"meaning": "untouchable (derogatory)", "severity": "critical", "is_casteist": True},
        
        # Common internet slang (Hinglish)
        "yaar": {"meaning": "friend/dude", "severity": "none"},
        "bhai": {"meaning": "brother/bro", "severity": "none"},
        "desi": {"meaning": "Indian/local", "severity": "none"},
        "jugaad": {"meaning": "creative hack", "severity": "none"},
        "bakwas": {"meaning": "nonsense/rubbish", "severity": "low"},
        "timepass": {"meaning": "passing time", "severity": "none"},
        "tension mat le": {"meaning": "don't worry", "severity": "none"},
    }
    
    # Cultural sensitivity topics
    CULTURAL_SENSITIVITIES = {
        "caste": {
            "keywords": ["जात", "जाति", "ब्राह्मण", "क्षत्रिय", "वैश्य", "शूद्र", "दलित", "आरक्षण", "caste", "reservation"],
            "sensitivity": "high",
            "description": "Caste-related discussion - requires careful analysis"
        },
        "religion": {
            "keywords": ["हिंदू", "मुस्लिम", "सिख", "ईसाई", "मंदिर", "मस्जिद", "गुरुद्वारा", "चर्च", "hindu", "muslim", "temple", "mosque"],
            "sensitivity": "high",
            "description": "Religious discussion - potential for communal tension"
        },
        "cow": {
            "keywords": ["गाय", "गौ", "गौमाता", "गौरक्षा", "beef", "cow", "गौहत्या"],
            "sensitivity": "high",
            "description": "Cow-related content - highly sensitive in India"
        },
        "politics": {
            "keywords": ["मोदी", "राहुल", "भाजपा", "कांग्रेस", "आप", "BJP", "Congress", "AAP", "चुनाव", "वोट"],
            "sensitivity": "medium",
            "description": "Political content - may be contentious"
        },
        "regional": {
            "keywords": ["बिहारी", "मद्रासी", "भैया", "पंजाबी", "बंगाली", "UP", "Bihar"],
            "sensitivity": "medium",
            "description": "Regional stereotypes - potential prejudice"
        }
    }
    
    # Negative sentiment indicators
    NEGATIVE_INDICATORS = [
        "मार", "काट", "जला", "नफरत", "घृणा", "मर जा", "भाग जा", "निकल", 
        "hate", "kill", "die", "destroy", "hurt", "attack"
    ]
    
    # Positive sentiment indicators
    POSITIVE_INDICATORS = [
        "प्यार", "खुशी", "अच्छा", "बढ़िया", "शानदार", "मज़ा", "धन्यवाद",
        "love", "happy", "great", "amazing", "beautiful", "thanks"
    ]
    
    def __init__(self, llm_provider=None):
        """
        Initialize the Hindi Cultural Agent.
        
        Args:
            llm_provider: Optional LLMProvider for enhanced analysis
        """
        self.llm = llm_provider
    
    def _detect_hindi(self, text: str) -> bool:
        """Detect if text contains Hindi (Devanagari script)."""
        # Devanagari Unicode range: \u0900-\u097F
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(devanagari_pattern.search(text))
    
    def _detect_language_ratio(self, text: str) -> Dict[str, float]:
        """Calculate the ratio of Hindi to English in text."""
        # Split into words
        words = text.split()
        if not words:
            return {"hindi": 0.0, "english": 0.0, "other": 1.0}
        
        hindi_count = 0
        english_count = 0
        
        for word in words:
            # Check for Devanagari
            if re.search(r'[\u0900-\u097F]', word):
                hindi_count += 1
            # Check for Latin alphabet
            elif re.search(r'[a-zA-Z]', word):
                english_count += 1
        
        total = len(words)
        return {
            "hindi": hindi_count / total if total > 0 else 0.0,
            "english": english_count / total if total > 0 else 0.0,
            "other": (total - hindi_count - english_count) / total if total > 0 else 0.0
        }
    
    def _detect_dialect(self, text: str, lang_ratio: Dict[str, float]) -> HindiDialect:
        """Determine the dialect/variation of Hindi used."""
        has_hindi = lang_ratio.get("hindi", 0) > 0.1
        has_english = lang_ratio.get("english", 0) > 0.1
        
        if has_hindi and has_english:
            return HindiDialect.HINGLISH
        elif has_hindi:
            # Check for colloquial patterns
            colloquial_markers = ["यार", "बॉस", "भाई", "क्या बात", "कमाल"]
            if any(marker in text for marker in colloquial_markers):
                return HindiDialect.COLLOQUIAL
            return HindiDialect.STANDARD
        else:
            return HindiDialect.HINGLISH  # Romanized Hindi
    
    def _detect_slang(self, text: str) -> List[str]:
        """Detect slang terms in the content."""
        found_slang = []
        text_lower = text.lower()
        
        for slang, info in self.SLANG_DICTIONARY.items():
            if slang.lower() in text_lower or slang in text:
                found_slang.append(slang)
        
        return found_slang
    
    def _analyze_sentiment(self, text: str) -> tuple:
        """Analyze sentiment of the content."""
        text_lower = text.lower()
        
        negative_score = sum(1 for ind in self.NEGATIVE_INDICATORS if ind.lower() in text_lower)
        positive_score = sum(1 for ind in self.POSITIVE_INDICATORS if ind.lower() in text_lower)
        
        # Check for offensive slang
        for slang, info in self.SLANG_DICTIONARY.items():
            if slang in text or slang.lower() in text_lower:
                severity = info.get("severity", "none")
                if severity in ["high", "critical"]:
                    negative_score += 3
                elif severity == "medium":
                    negative_score += 2
                elif severity == "low":
                    negative_score += 1
        
        # Calculate sentiment
        total = positive_score + negative_score
        if total == 0:
            return SentimentType.NEUTRAL, 0.0
        
        score = (positive_score - negative_score) / max(total, 1)
        
        if negative_score > positive_score * 2:
            if any(threat in text_lower for threat in ["मार", "जान से", "kill", "destroy"]):
                return SentimentType.THREATENING, score
            return SentimentType.HOSTILE, score
        elif negative_score > positive_score:
            return SentimentType.NEGATIVE, score
        elif positive_score > negative_score:
            return SentimentType.POSITIVE, score
        else:
            return SentimentType.NEUTRAL, score
    
    def _detect_cultural_contexts(self, text: str) -> List[CulturalContext]:
        """Detect cultural sensitivities in content."""
        contexts = []
        text_lower = text.lower()
        
        for context_name, context_info in self.CULTURAL_SENSITIVITIES.items():
            keywords = context_info["keywords"]
            matched = [kw for kw in keywords if kw.lower() in text_lower or kw in text]
            
            if matched:
                related_policies = []
                if context_name in ["caste", "religion", "regional"]:
                    related_policies.append("hate_speech")
                if context_name == "cow":
                    related_policies.extend(["hate_speech", "graphic_violence"])
                
                contexts.append(CulturalContext(
                    context_type=context_name,
                    description=context_info["description"],
                    sensitivity_level=context_info["sensitivity"],
                    related_policies=related_policies
                ))
        
        return contexts
    
    def _check_casteism(self, text: str) -> bool:
        """Check for caste-based discrimination."""
        for slang, info in self.SLANG_DICTIONARY.items():
            if info.get("is_casteist"):
                if slang in text or slang.lower() in text.lower():
                    return True
        return False
    
    def analyze(self, content: str, content_id: str = "unknown") -> HindiAnalysisResult:
        """
        Perform comprehensive Hindi cultural analysis.
        
        Args:
            content: The text content to analyze
            content_id: Identifier for the content
            
        Returns:
            HindiAnalysisResult with all cultural analysis
        """
        # Basic language detection
        is_hindi = self._detect_hindi(content)
        lang_ratio = self._detect_language_ratio(content)
        is_mixed = lang_ratio["hindi"] > 0.1 and lang_ratio["english"] > 0.1
        
        # Dialect detection
        dialect = self._detect_dialect(content, lang_ratio)
        
        # Slang detection
        slang_found = self._detect_slang(content)
        
        # Sentiment analysis
        sentiment, sentiment_score = self._analyze_sentiment(content)
        
        # Cultural context detection
        cultural_contexts = self._detect_cultural_contexts(content)
        
        # Check for special attention requirements
        requires_attention = (
            self._check_casteism(content) or
            sentiment in [SentimentType.HOSTILE, SentimentType.THREATENING] or
            any(c.sensitivity_level == "high" for c in cultural_contexts) or
            any(self.SLANG_DICTIONARY.get(s, {}).get("severity") in ["high", "critical"] for s in slang_found)
        )
        
        # Generate transliteration notes
        transliteration_notes = ""
        if is_mixed or dialect == HindiDialect.HINGLISH:
            transliteration_notes = (
                "Content uses code-mixing (Hinglish). "
                "Hindi and English are interspersed, common in Indian social media."
            )
        elif is_hindi:
            transliteration_notes = "Content is primarily in Hindi (Devanagari script)."
        else:
            # Check for romanized Hindi
            romanized_indicators = ["yaar", "bhai", "kya", "hai", "nahi", "aur", "tum", "mein"]
            if any(ind in content.lower() for ind in romanized_indicators):
                transliteration_notes = "Content appears to use romanized Hindi (Hindi written in Latin script)."
        
        # Generate summary
        summary_parts = []
        if is_hindi or is_mixed:
            summary_parts.append(f"Language: {dialect.value}")
        if slang_found:
            summary_parts.append(f"Slang detected: {len(slang_found)} terms")
        if cultural_contexts:
            summary_parts.append(f"Cultural sensitivities: {', '.join(c.context_type for c in cultural_contexts)}")
        summary_parts.append(f"Sentiment: {sentiment.value}")
        
        analysis_summary = ". ".join(summary_parts) if summary_parts else "No significant Hindi cultural elements detected."
        
        return HindiAnalysisResult(
            content_id=content_id,
            is_hindi=is_hindi,
            is_mixed_language=is_mixed,
            dialect=dialect,
            detected_language_ratio=lang_ratio,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            cultural_contexts=cultural_contexts,
            slang_detected=slang_found,
            transliteration_notes=transliteration_notes,
            analysis_summary=analysis_summary,
            requires_special_attention=requires_attention
        )
    
    def get_slang_meaning(self, slang: str) -> Optional[Dict]:
        """Get the meaning and severity of a slang term."""
        return self.SLANG_DICTIONARY.get(slang)
    
    def is_casteist_content(self, content: str) -> bool:
        """Check if content contains caste-based discrimination."""
        return self._check_casteism(content)
