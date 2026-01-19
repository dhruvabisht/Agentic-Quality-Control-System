"""
Market Insights Report Generator for Sentinel-AI

Generates comprehensive reports on policy violation trends,
with special focus on Hindi-language content patterns.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..database.models import ContentAudit, PolicyViolation, KPIMetrics, VerdictType, ContentLanguage
from ..database.connection import get_session


@dataclass
class TrendData:
    """Data point for trend analysis"""
    date: str
    count: int
    category: str
    percentage: float = 0.0


@dataclass
class MarketInsight:
    """Single market insight finding"""
    title: str
    description: str
    severity: str  # info, warning, critical
    data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MarketInsightsReport:
    """Complete market insights report"""
    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    
    # Summary metrics
    total_audits: int
    hindi_content_count: int
    hindi_percentage: float
    violation_rate: float
    
    # Trend data
    violation_trends: List[TrendData]
    category_distribution: Dict[str, int]
    hindi_specific_violations: List[Dict[str, Any]]
    
    # Insights
    key_insights: List[MarketInsight]
    
    # Recommendations
    action_items: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "period": {
                "start": self.period_start,
                "end": self.period_end
            },
            "summary": {
                "total_audits": self.total_audits,
                "hindi_content_count": self.hindi_content_count,
                "hindi_percentage": self.hindi_percentage,
                "violation_rate": self.violation_rate
            },
            "trends": {
                "violations": [
                    {"date": t.date, "count": t.count, "category": t.category}
                    for t in self.violation_trends
                ],
                "category_distribution": self.category_distribution,
                "hindi_specific": self.hindi_specific_violations
            },
            "insights": [
                {
                    "title": i.title,
                    "description": i.description,
                    "severity": i.severity,
                    "data": i.data,
                    "recommendations": i.recommendations
                }
                for i in self.key_insights
            ],
            "action_items": self.action_items
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Sentinel-AI Market Insights Report",
            f"",
            f"**Report ID:** {self.report_id}",
            f"**Generated:** {self.generated_at}",
            f"**Period:** {self.period_start} to {self.period_end}",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Audits | {self.total_audits:,} |",
            f"| Hindi Content | {self.hindi_content_count:,} ({self.hindi_percentage:.1f}%) |",
            f"| Violation Rate | {self.violation_rate:.1f}% |",
            f"",
            f"---",
            f"",
            f"## Violation Category Distribution",
            f"",
        ]
        
        if self.category_distribution:
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            for category, count in sorted(
                self.category_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                lines.append(f"| {category.replace('_', ' ').title()} | {count} |")
        else:
            lines.append("*No violations recorded in this period.*")
        
        lines.extend([
            f"",
            f"---",
            f"",
            f"## Hindi-Language Content Analysis",
            f"",
        ])
        
        if self.hindi_specific_violations:
            lines.append("### Top Hindi-Specific Violation Patterns")
            lines.append("")
            for i, violation in enumerate(self.hindi_specific_violations[:5], 1):
                lines.append(f"{i}. **{violation.get('pattern', 'Unknown')}**")
                lines.append(f"   - Count: {violation.get('count', 0)}")
                lines.append(f"   - Category: {violation.get('category', 'N/A')}")
                lines.append("")
        else:
            lines.append("*No Hindi-specific violations identified.*")
        
        lines.extend([
            f"",
            f"---",
            f"",
            f"## Key Insights",
            f"",
        ])
        
        for insight in self.key_insights:
            severity_emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "critical": "🚨"
            }.get(insight.severity, "📌")
            
            lines.append(f"### {severity_emoji} {insight.title}")
            lines.append(f"")
            lines.append(insight.description)
            lines.append("")
            
            if insight.recommendations:
                lines.append("**Recommendations:**")
                for rec in insight.recommendations:
                    lines.append(f"- {rec}")
                lines.append("")
        
        lines.extend([
            f"---",
            f"",
            f"## Action Items",
            f"",
        ])
        
        for i, action in enumerate(self.action_items, 1):
            lines.append(f"{i}. {action}")
        
        lines.extend([
            f"",
            f"---",
            f"",
            f"*Report generated by Sentinel-AI v1.0.0*"
        ])
        
        return "\n".join(lines)


class MarketInsightsGenerator:
    """
    Generates Market Insight Reports for Hindi-language content analysis.
    
    Features:
    - Trend analysis over time
    - Category distribution analysis
    - Hindi-specific violation patterns
    - Automated insight generation
    - PDF report export
    """
    
    def __init__(self):
        """Initialize the Market Insights Generator."""
        pass
    
    def _get_audits_in_period(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[ContentAudit]:
        """Get all audits within the specified period."""
        with get_session() as session:
            audits = session.query(ContentAudit).filter(
                ContentAudit.created_at >= start_date,
                ContentAudit.created_at <= end_date
            ).all()
            # Detach from session
            return [self._audit_to_dict(a) for a in audits]
    
    def _audit_to_dict(self, audit: ContentAudit) -> Dict[str, Any]:
        """Convert audit to dictionary (detached from session)."""
        return {
            "id": audit.id,
            "content_id": audit.content_id,
            "language": audit.language.value if audit.language else "english",
            "verdict": audit.verdict.value if audit.verdict else "pass",
            "confidence_score": audit.confidence_score,
            "created_at": audit.created_at.isoformat() if audit.created_at else None,
            "is_sensitive": audit.is_sensitive,
            "violations": [
                {
                    "category": v.category.value if v.category else "other",
                    "policy_name": v.policy_name,
                    "severity": v.severity,
                    "is_hindi_specific": v.is_hindi_specific
                }
                for v in audit.violations
            ] if audit.violations else []
        }
    
    def _calculate_category_distribution(
        self, 
        audits: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate violation distribution by category."""
        distribution = defaultdict(int)
        
        for audit in audits:
            for violation in audit.get("violations", []):
                category = violation.get("category", "other")
                distribution[category] += 1
        
        return dict(distribution)
    
    def _identify_hindi_patterns(
        self, 
        audits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify patterns specific to Hindi content."""
        patterns = defaultdict(lambda: {"count": 0, "categories": set()})
        
        for audit in audits:
            if audit.get("language") in ["hindi", "mixed"]:
                for violation in audit.get("violations", []):
                    if violation.get("is_hindi_specific"):
                        policy = violation.get("policy_name", "Unknown")
                        patterns[policy]["count"] += 1
                        patterns[policy]["categories"].add(violation.get("category", "other"))
        
        # Convert to list
        result = []
        for pattern, data in patterns.items():
            result.append({
                "pattern": pattern,
                "count": data["count"],
                "category": ", ".join(data["categories"])
            })
        
        # Sort by count
        result.sort(key=lambda x: x["count"], reverse=True)
        
        return result
    
    def _generate_insights(
        self,
        audits: List[Dict[str, Any]],
        category_dist: Dict[str, int],
        hindi_patterns: List[Dict[str, Any]]
    ) -> List[MarketInsight]:
        """Generate key insights from the data."""
        insights = []
        
        total = len(audits)
        if total == 0:
            return [MarketInsight(
                title="Insufficient Data",
                description="Not enough audit data to generate insights.",
                severity="info",
                recommendations=["Run more content audits to generate meaningful insights."]
            )]
        
        # Insight 1: Hindi content proportion
        hindi_count = sum(1 for a in audits if a.get("language") in ["hindi", "mixed"])
        hindi_pct = (hindi_count / total) * 100 if total > 0 else 0
        
        if hindi_pct > 20:
            insights.append(MarketInsight(
                title="High Hindi Content Volume",
                description=f"Hindi content represents {hindi_pct:.1f}% of all audited content. "
                           f"This indicates strong engagement from the Hindi-speaking market.",
                severity="info",
                data={"hindi_percentage": hindi_pct, "hindi_count": hindi_count},
                recommendations=[
                    "Ensure adequate Hindi language expertise in moderation team.",
                    "Consider expanding Hindi-specific policy training.",
                    "Monitor for emerging Hindi slang and code-mixing patterns."
                ]
            ))
        
        # Insight 2: Dominant violation category
        if category_dist:
            top_category = max(category_dist.items(), key=lambda x: x[1])
            top_pct = (top_category[1] / sum(category_dist.values())) * 100
            
            severity = "critical" if top_pct > 50 else "warning" if top_pct > 30 else "info"
            
            insights.append(MarketInsight(
                title=f"Dominant Violation Category: {top_category[0].replace('_', ' ').title()}",
                description=f"{top_category[0].replace('_', ' ').title()} accounts for {top_pct:.1f}% "
                           f"of all violations ({top_category[1]} instances).",
                severity=severity,
                data={"category": top_category[0], "count": top_category[1], "percentage": top_pct},
                recommendations=[
                    f"Review {top_category[0].replace('_', ' ')} policy guidelines.",
                    "Consider targeted user education campaigns.",
                    "Analyze root causes for this violation type."
                ]
            ))
        
        # Insight 3: Hindi-specific patterns
        if hindi_patterns:
            top_pattern = hindi_patterns[0]
            insights.append(MarketInsight(
                title="Hindi-Specific Violation Pattern Detected",
                description=f"'{top_pattern['pattern']}' is the most common Hindi-specific violation "
                           f"with {top_pattern['count']} instances.",
                severity="warning",
                data=top_pattern,
                recommendations=[
                    "Update Hindi language models with new patterns.",
                    "Add regional slang to policy keyword database.",
                    "Consider cultural context training for moderators."
                ]
            ))
        
        # Insight 4: Escalation rate
        escalations = sum(1 for a in audits if a.get("verdict") == "escalate")
        escalation_rate = (escalations / total) * 100 if total > 0 else 0
        
        if escalation_rate > 20:
            insights.append(MarketInsight(
                title="High Escalation Rate",
                description=f"{escalation_rate:.1f}% of audits are being escalated for human review. "
                           f"This may indicate model uncertainty or edge cases.",
                severity="warning",
                data={"escalation_rate": escalation_rate, "escalation_count": escalations},
                recommendations=[
                    "Review escalated cases for patterns.",
                    "Consider adjusting confidence thresholds.",
                    "Use escalation data for model retraining."
                ]
            ))
        
        return insights
    
    def _generate_action_items(
        self,
        insights: List[MarketInsight],
        violation_rate: float
    ) -> List[str]:
        """Generate prioritized action items."""
        actions = []
        
        # Priority actions based on violation rate
        if violation_rate > 30:
            actions.append("🚨 URGENT: High violation rate detected. Implement stricter content filters.")
        
        # Actions from insights
        for insight in insights:
            if insight.severity == "critical":
                actions.append(f"🔴 Address critical issue: {insight.title}")
            elif insight.severity == "warning":
                actions.append(f"🟡 Review: {insight.title}")
        
        # Standard recommendations
        actions.extend([
            "📊 Share report with Trust & Safety leadership.",
            "📝 Update policy documentation based on findings.",
            "🔄 Schedule follow-up analysis in 7 days."
        ])
        
        return actions
    
    def generate_report(
        self,
        days: int = 7,
        end_date: Optional[datetime] = None
    ) -> MarketInsightsReport:
        """
        Generate a comprehensive market insights report.
        
        Args:
            days: Number of days to analyze (default 7)
            end_date: End date for analysis (default: now)
            
        Returns:
            MarketInsightsReport with full analysis
        """
        import uuid
        
        if end_date is None:
            end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get audit data
        audits = self._get_audits_in_period(start_date, end_date)
        
        # Calculate metrics
        total_audits = len(audits)
        hindi_count = sum(1 for a in audits if a.get("language") in ["hindi", "mixed"])
        hindi_pct = (hindi_count / total_audits) * 100 if total_audits > 0 else 0
        
        violations = sum(1 for a in audits if a.get("verdict") == "fail")
        violation_rate = (violations / total_audits) * 100 if total_audits > 0 else 0
        
        # Analyze patterns
        category_dist = self._calculate_category_distribution(audits)
        hindi_patterns = self._identify_hindi_patterns(audits)
        
        # Generate trends (simplified - by day)
        trends = []
        for i in range(days):
            day = start_date + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            day_audits = [a for a in audits if a.get("created_at", "").startswith(day_str)]
            day_violations = sum(1 for a in day_audits if a.get("verdict") == "fail")
            trends.append(TrendData(
                date=day_str,
                count=day_violations,
                category="violations",
                percentage=(day_violations / len(day_audits) * 100) if day_audits else 0
            ))
        
        # Generate insights
        insights = self._generate_insights(audits, category_dist, hindi_patterns)
        
        # Generate action items
        actions = self._generate_action_items(insights, violation_rate)
        
        return MarketInsightsReport(
            report_id=f"MIR-{uuid.uuid4().hex[:8].upper()}",
            generated_at=datetime.utcnow().isoformat(),
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_audits=total_audits,
            hindi_content_count=hindi_count,
            hindi_percentage=hindi_pct,
            violation_rate=violation_rate,
            violation_trends=trends,
            category_distribution=category_dist,
            hindi_specific_violations=hindi_patterns,
            key_insights=insights,
            action_items=actions
        )
    
    def export_to_pdf(self, report: MarketInsightsReport, output_path: str) -> str:
        """
        Export report to PDF format.
        
        Args:
            report: The report to export
            output_path: Path for the output PDF
            
        Returns:
            Path to the generated PDF
        """
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Sentinel-AI Market Insights Report", ln=True, align="C")
            
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, f"Generated: {report.generated_at}", ln=True, align="C")
            pdf.cell(0, 10, f"Period: {report.period_start} to {report.period_end}", ln=True, align="C")
            
            # Summary
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Executive Summary", ln=True)
            
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 8, f"Total Audits: {report.total_audits:,}", ln=True)
            pdf.cell(0, 8, f"Hindi Content: {report.hindi_content_count:,} ({report.hindi_percentage:.1f}%)", ln=True)
            pdf.cell(0, 8, f"Violation Rate: {report.violation_rate:.1f}%", ln=True)
            
            # Category Distribution
            if report.category_distribution:
                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Violation Categories", ln=True)
                
                pdf.set_font("Arial", "", 10)
                for category, count in report.category_distribution.items():
                    pdf.cell(0, 8, f"  {category.replace('_', ' ').title()}: {count}", ln=True)
            
            # Insights
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Key Insights", ln=True)
            
            pdf.set_font("Arial", "", 10)
            for insight in report.key_insights:
                pdf.multi_cell(0, 8, f"• {insight.title}: {insight.description[:100]}...")
            
            # Action Items
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Action Items", ln=True)
            
            pdf.set_font("Arial", "", 10)
            for action in report.action_items:
                pdf.multi_cell(0, 8, f"• {action}")
            
            pdf.output(output_path)
            return output_path
            
        except ImportError:
            # Fallback to markdown file if fpdf not available
            md_path = output_path.replace(".pdf", ".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            return md_path
