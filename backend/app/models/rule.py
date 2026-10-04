from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Rule(Base):
    """A DQ rule: Observation -> Recommendation -> (human approval) -> Business Rule.

    status:
        recommended  - system-suggested, awaiting validation/approval
        approved     - human-approved and eligible for execution
        rejected     - human-rejected
        retired      - previously approved, later removed
    """

    __tablename__ = "rules"

    rule_id: Mapped[int] = mapped_column(
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

    # Sequential code within the version (R001, R002, ...). Assigned when a
    # rule is first approved; recommendations carry code=None until then.
    rule_code: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )


class RuleVersion(Base):
    """Immutable snapshot of one rule version (R001 v1, v2, ...).

    Historical rule definitions are never overwritten: editing an approved
    rule creates a new RuleVersion snapshot and bumps rules.version_number,
    so every execution result references the exact definition that ran.
    """

    __tablename__ = "rule_versions"

    rule_version_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    rule_id: Mapped[int] = mapped_column(
        ForeignKey("rules.rule_id"),
        nullable=False,
        index=True,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    rule_code: Mapped[str | None] = mapped_column(Text, nullable=True)

    rule_name: Mapped[str] = mapped_column(Text, nullable=False)

    rule_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    metric: Mapped[str] = mapped_column(Text, nullable=False)

    # How this version came to be: initial_approval | edited
    change_kind: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="initial_approval",
    )

    approved_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
    )


class RuleFeedback(Base):
    """Human decision on a rule/metric candidate - feedback-store only.

    Captures the candidate, its features/evidence and the human decision so
    a future offline ranker (e.g. XGBRanker) can learn from real decisions.
    Feedback NEVER modifies the current run: no retraining here, no behavior
    change until an offline pipeline produces a human-promoted model.
    """

    __tablename__ = "rule_feedback"

    feedback_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    rule_id: Mapped[int | None] = mapped_column(
        ForeignKey("rules.rule_id"),
        nullable=True,
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

    # Stable identity of the candidate recommendation (template + target),
    # so feedback stays comparable across re-runs.
    candidate_key: Mapped[str] = mapped_column(Text, nullable=False)

    metric: Mapped[str] = mapped_column(Text, nullable=False)

    rule_template: Mapped[str] = mapped_column(Text, nullable=False)

    target_table: Mapped[str] = mapped_column(Text, nullable=False)

    target_columns: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # accepted | rejected | edited
    decision: Mapped[str] = mapped_column(Text, nullable=False)

    # Evidence snapshot used by the recommendation (feature source).
    evidence_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Ranking features captured at decision time (future XGBRanker input).
    features_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    recommendation_score: Mapped[float | None] = mapped_column(nullable=True)

    confidence: Mapped[str | None] = mapped_column(Text, nullable=True)

    recommendation_source: Mapped[str | None] = mapped_column(Text, nullable=True)

    edited_rule_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
    )


class ColumnMetricContext(Base):
    """Business context for one column/metric pair (user-owned).

    Example: completeness for middle_name is measurable but NOT business-
    required. required=False suppresses requiredness recommendations while
    the metric stays measurable. required=None means undecided (default
    treated as required for recommendation purposes only).
    """

    __tablename__ = "column_metric_contexts"

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

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    table_name: Mapped[str] = mapped_column(Text, nullable=False)

    column_name: Mapped[str] = mapped_column(Text, nullable=False)

    metric: Mapped[str] = mapped_column(Text, nullable=False)

    # None = undecided (recommendations treat as required); False explicitly
    # marks the column/metric as NOT business-required.
    required: Mapped[bool | None] = mapped_column(nullable=True, default=None)

    # Free-text business note ("Middle name is optional.").
    business_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )
