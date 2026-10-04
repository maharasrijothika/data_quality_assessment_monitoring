from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetContext(Base):
    """Dataset business context (dataset-level)."""

    __tablename__ = "dataset_contexts"

    context_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
    )

    table_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    rule_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    rule_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    metric: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="system",
    )

    risk: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="low",
    )

    validation_status: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    validation_issues_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="recommended",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class RuleApproval(Base):
    """Human approval/rejection/edit decision for a rule."""

    __tablename__ = "rule_approvals"

    approval_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    rule_id: Mapped[int] = mapped_column(
        ForeignKey("rules.rule_id"),
        nullable=False,
        index=True,
    )

    decision: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    edited_rule_json: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class RuleExecution(Base):
    """Result of one rule execution run over a dataset version."""

    __tablename__ = "rule_executions"

    execution_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    rule_id: Mapped[int] = mapped_column(
        ForeignKey("rules.rule_id"),
        nullable=False,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Populated only when status == "error" (execution failure evidence).
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    total_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    applicable_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    passed_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    failed_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    not_applicable_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    pass_rate: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )

    violation_rate: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )

    evidence_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class MetricResult(Base):
    """Aggregated metric score for a version, from executions."""

    __tablename__ = "metric_results"

    metric_result_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    metric: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    score: Mapped[float] = mapped_column(
        nullable=False,
    )

    applicable_records: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    failed_records: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    rule_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="available",
    )

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class DQScore(Base):
    """Overall DQ score snapshot for a version."""

    __tablename__ = "dq_scores"

    score_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    overall_score: Mapped[float] = mapped_column(
        nullable=False,
    )

    weighted: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    metric_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    excluded_metrics: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    execution_errors: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    details_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class RCAFinding(Base):
    """Evidence-based root cause analysis finding."""

    __tablename__ = "rca_findings"

    finding_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    rule_id: Mapped[int] = mapped_column(
        ForeignKey("rules.rule_id"),
        nullable=False,
        index=True,
    )

    failure_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    analysis_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class RemediationRecord(Base):
    """Remediation proposal and its outcome."""

    __tablename__ = "remediation_records"

    remediation_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
    )

    source_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    remediation_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="proposed",
    )

    proposal_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    affected_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    correction_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    resulting_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=True,
    )

    before_score: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    after_score: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class RemediationApproval(Base):
    """Human decision on a remediation proposal."""

    __tablename__ = "remediation_approvals"

    approval_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    remediation_id: Mapped[int] = mapped_column(
        ForeignKey("remediation_records.remediation_id"),
        nullable=False,
        index=True,
    )

    decision: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class DriftResult(Base):
    """Monitoring drift result comparing a version against a baseline."""

    __tablename__ = "drift_results"

    drift_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
    )

    baseline_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
    )

    comparison_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
    )

    method: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    column_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    table_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    statistic: Mapped[float] = mapped_column(
        nullable=False,
    )

    threshold: Mapped[float] = mapped_column(
        nullable=False,
    )

    drift_detected: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    details_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class SemanticFeedback(Base):
    """Human feedback on a semantic prediction (approved/rejected/edited)."""

    __tablename__ = "semantic_feedback"

    feedback_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("semantic_predictions.prediction_id"),
        nullable=True,
        index=True,
    )

    column_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    table_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    original_concept_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    corrected_concept_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    decision: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    features_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    model_name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class ModelVersion(Base):
    """Versioned offline-trained model in the registry."""

    __tablename__ = "model_versions"

    model_version_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="candidate",
    )

    metrics_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
