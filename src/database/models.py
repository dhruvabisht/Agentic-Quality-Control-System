"""
Database Models for Sentinel-AI

SQLAlchemy ORM models for:
- ContentAudit: Main audit records
- PolicyViolation: Detected violations
- HumanReview: HITL overrides and feedback
- KPIMetrics: Performance tracking
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, 
    DateTime, ForeignKey, Enum, JSON
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


class VerdictType(str, enum.Enum):
    """Possible audit verdicts"""
    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"


class ViolationCategory(str, enum.Enum):
    """Policy violation categories"""
    HATE_SPEECH = "hate_speech"
    GRAPHIC_VIOLENCE = "graphic_violence"
    HARASSMENT = "harassment"
    MISINFORMATION = "misinformation"
    ADULT_CONTENT = "adult_content"
    SPAM = "spam"
    ADVERTISER_VIOLATION = "advertiser_violation"
    OTHER = "other"


class ContentLanguage(str, enum.Enum):
    """Supported content languages"""
    ENGLISH = "english"
    HINDI = "hindi"
    MIXED = "mixed"  # Code-mixing


class ContentAudit(Base):
    """
    Main audit record for content analysis.
    Stores the original content, agent analysis, and final verdict.
    """
    __tablename__ = "content_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Content Information
    content_id = Column(String(100), unique=True, nullable=False, index=True)
    content_text = Column(Text, nullable=False)
    content_type = Column(String(50), default="post")  # post, comment, ad, story
    language = Column(Enum(ContentLanguage), default=ContentLanguage.ENGLISH)
    
    # Audit Results
    verdict = Column(Enum(VerdictType), nullable=False)
    confidence_score = Column(Float, nullable=False)  # 0-100
    
    # Agent Analysis (stored as JSON for flexibility)
    policy_agent_output = Column(JSON)
    hindi_cultural_agent_output = Column(JSON)
    auditor_agent_output = Column(JSON)
    chain_of_thought = Column(Text)  # Full reasoning chain
    
    # Sensitivity
    is_sensitive = Column(Boolean, default=False)
    sensitivity_category = Column(String(100))
    
    # Status
    requires_human_review = Column(Boolean, default=False)
    is_reviewed = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    violations = relationship("PolicyViolation", back_populates="audit", cascade="all, delete-orphan")
    human_review = relationship("HumanReview", back_populates="audit", uselist=False)
    
    def __repr__(self):
        return f"<ContentAudit(id={self.id}, verdict={self.verdict}, confidence={self.confidence_score})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "content_id": self.content_id,
            "content_text": self.content_text,
            "content_type": self.content_type,
            "language": self.language.value if self.language else None,
            "verdict": self.verdict.value if self.verdict else None,
            "confidence_score": self.confidence_score,
            "chain_of_thought": self.chain_of_thought,
            "is_sensitive": self.is_sensitive,
            "requires_human_review": self.requires_human_review,
            "is_reviewed": self.is_reviewed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "violations": [v.to_dict() for v in self.violations] if self.violations else []
        }


class PolicyViolation(Base):
    """
    Detailed record of policy violations found in content.
    Linked to ContentAudit for tracking specific rule breaches.
    """
    __tablename__ = "policy_violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(Integer, ForeignKey("content_audits.id"), nullable=False)
    
    # Violation Details
    category = Column(Enum(ViolationCategory), nullable=False)
    policy_name = Column(String(200), nullable=False)
    policy_rule_id = Column(String(50))
    
    # Evidence
    severity = Column(String(20))  # low, medium, high, critical
    matched_keywords = Column(JSON)  # List of matched keywords/phrases
    violation_snippet = Column(Text)  # Relevant part of content
    explanation = Column(Text)  # Why this is a violation
    
    # Hindi-specific context
    is_hindi_specific = Column(Boolean, default=False)
    cultural_context = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    audit = relationship("ContentAudit", back_populates="violations")
    
    def __repr__(self):
        return f"<PolicyViolation(id={self.id}, category={self.category}, severity={self.severity})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "category": self.category.value if self.category else None,
            "policy_name": self.policy_name,
            "severity": self.severity,
            "explanation": self.explanation,
            "is_hindi_specific": self.is_hindi_specific,
            "cultural_context": self.cultural_context
        }


class HumanReview(Base):
    """
    Human-in-the-Loop review records.
    Tracks Quality Measurement Specialist decisions and feedback.
    """
    __tablename__ = "human_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(Integer, ForeignKey("content_audits.id"), unique=True, nullable=False)
    
    # Reviewer Information
    reviewer_id = Column(String(100), nullable=False)
    reviewer_name = Column(String(200))
    
    # Review Decision
    original_verdict = Column(Enum(VerdictType), nullable=False)
    final_verdict = Column(Enum(VerdictType), nullable=False)
    was_overridden = Column(Boolean, default=False)
    
    # Feedback
    feedback_notes = Column(Text)
    disagreement_reason = Column(Text)
    suggested_improvements = Column(Text)
    
    # For model retraining
    is_flagged_for_retraining = Column(Boolean, default=False)
    retraining_priority = Column(Integer, default=0)  # 1-5 priority
    
    # Timestamps
    reviewed_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    audit = relationship("ContentAudit", back_populates="human_review")
    
    def __repr__(self):
        return f"<HumanReview(id={self.id}, overridden={self.was_overridden})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "audit_id": self.audit_id,
            "reviewer_name": self.reviewer_name,
            "original_verdict": self.original_verdict.value if self.original_verdict else None,
            "final_verdict": self.final_verdict.value if self.final_verdict else None,
            "was_overridden": self.was_overridden,
            "feedback_notes": self.feedback_notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None
        }


class KPIMetrics(Base):
    """
    Key Performance Indicator tracking.
    Aggregated metrics for dashboard analytics.
    """
    __tablename__ = "kpi_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time Period
    metric_date = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), default="daily")  # hourly, daily, weekly, monthly
    
    # Audit Metrics
    total_audits = Column(Integer, default=0)
    pass_count = Column(Integer, default=0)
    fail_count = Column(Integer, default=0)
    escalate_count = Column(Integer, default=0)
    
    # Accuracy Metrics
    accuracy_rate = Column(Float)  # % of correct decisions after human review
    false_positive_rate = Column(Float)
    false_negative_rate = Column(Float)
    
    # Performance Metrics
    avg_confidence_score = Column(Float)
    avg_processing_time_ms = Column(Float)
    
    # Human Review Metrics
    human_review_count = Column(Integer, default=0)
    override_count = Column(Integer, default=0)
    override_rate = Column(Float)
    
    # Language Breakdown
    english_count = Column(Integer, default=0)
    hindi_count = Column(Integer, default=0)
    mixed_count = Column(Integer, default=0)
    
    # Violation Category Breakdown (JSON for flexibility)
    violation_breakdown = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<KPIMetrics(date={self.metric_date}, total={self.total_audits})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "metric_date": self.metric_date.isoformat() if self.metric_date else None,
            "period_type": self.period_type,
            "total_audits": self.total_audits,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "escalate_count": self.escalate_count,
            "accuracy_rate": self.accuracy_rate,
            "avg_confidence_score": self.avg_confidence_score,
            "override_rate": self.override_rate,
            "violation_breakdown": self.violation_breakdown
        }
