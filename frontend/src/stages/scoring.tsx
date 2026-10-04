import { useCallback, useEffect, useState } from 'react'
import { apiErrorMessage, apiFetch, jsonInit } from '../api'
import {
  CardHeader,
  Notice,
  PageHeader,
  StatusBadge,
  SummaryGrid,
  scoreTone,
} from '../ui'

type MetricEntry = {
  status: string
  score: number | null
  applicable?: number
  failed?: number
  rules?: Array<{
    rule_id: number
    rule_code: string | null
    rule_name: string
    status: string
    applicable: number | null
    failed: number | null
    pass_rate: number | null
    error_message?: string
  }>
}

type ColumnBreakdown = {
  table_name: string
  column_name: string
  metric_results: Record<string, MetricEntry>
  column_score: number | null
  evaluated_metrics: number
  applicable_rows_total: number
  failed_rows_total: number
}

type TableBreakdown = {
  table_name: string
  column_count: number
  metric_results: Record<string, MetricEntry>
  table_score: number | null
  evaluated_metrics: number
}

type ScoreData = {
  overall_score: number
  weighted: boolean
  metric_count: number
  excluded_metrics: number
  details: {
    metrics: Array<{
      metric: string
      score: number
      status: string
      applicable_records: number
      failed_records: number
      rule_count: number
    }>
    not_applicable_metrics: string[]
    critical_failures: Array<{
      metric: string
      score: number
      failed_records: number
      applicable_records: number
    }>
    column_breakdown?: {
      columns: ColumnBreakdown[]
      tables: TableBreakdown[]
      dataset: {
        metric_results: Record<string, MetricEntry>
        overall_score: number | null
        evaluated_metrics: number
      }
    }
  }
  computed_at: string
}

type RCAResponse = {
  findings: Array<{
    finding_id: number | null
    rule_id: number
    rule_name: string | null
    failure_count: number
    analysis: {
      status: string
      failed_rows?: number
      failure_percentage?: number
      patterns?: Array<{
        column: string
        distribution: Array<{ value: string; count: number; percentage: number }>
      }>
      concentration?: Array<{
        column: string
        distribution: Array<{ value: string; count: number; percentage: number }>
      }>
      language_note?: string
      message?: string
    }
  }>
  message?: string
}

type RemediationSummary = {
  remediation_id: number
  source_version_id: number
  resulting_version_id: number | null
  status: string
  affected_rows: number
  correction_count: number
  before_score: number | null
  after_score: number | null
}

type VersionItem = {
  version_id: number
  version_number: number
  parent_version_id: number | null
  overall_score: number | null
}

const formatNumber = (value: number | null | undefined): string =>
  typeof value === 'number' ? value.toLocaleString() : '—'

const METRIC_ORDER = [
  'completeness',
  'uniqueness',
  'validity',
  'accuracy',
  'consistency',
  'referential_integrity',
  'timeliness',
] as const

const METRIC_LABELS: Record<string, string> = {
  completeness: 'Completeness',
  uniqueness: 'Uniqueness',
  validity: 'Validity',
  accuracy: 'Accuracy',
  consistency: 'Consistency',
  referential_integrity: 'Referential Integrity',
  timeliness: 'Timeliness',
}

function MetricEntryBadge({ entry }: { entry: MetricEntry | undefined }) {
  if (!entry || entry.status === 'N/A') {
    return <StatusBadge tone="neutral">N/A</StatusBadge>
  }
  if (entry.status === 'EXECUTION_ERROR') {
    return <StatusBadge tone="danger">Error</StatusBadge>
  }
  if (entry.score === null || entry.score === undefined) {
    return <StatusBadge tone="neutral">N/A</StatusBadge>
  }
  return (
    <StatusBadge tone={scoreTone(entry.score)}>{entry.score.toFixed(2)}%</StatusBadge>
  )
}

function ColumnBreakdownView({ breakdown }: {
  breakdown: NonNullable<ScoreData['details']['column_breakdown']>
}) {
  const datasetMetrics = breakdown.dataset.metric_results
  const notApplicableCount = Object.values(datasetMetrics).filter(
    (e) => e.status === 'N/A',
  ).length
  const evaluatedCount = breakdown.dataset.evaluated_metrics

  return (
    <div className="section">
      <div className="section-head">
        <h3 className="section-title">Score breakdown (column → table → dataset)</h3>
        <p className="section-hint">
          N/A metrics are excluded from every denominator — they are never
          zero. Dataset score = mean of evaluated metric scores (equal
          weights; no silent business weighting).
        </p>
      </div>

      <SummaryGrid
        items={[
          [
            'Dataset score',
            breakdown.dataset.overall_score !== null ? (
              <StatusBadge
                key="ds"
                tone={scoreTone(breakdown.dataset.overall_score)}
              >
                {breakdown.dataset.overall_score.toFixed(2)}%
              </StatusBadge>
            ) : (
              <StatusBadge key="ds" tone="neutral">
                N/A
              </StatusBadge>
            ),
          ],
          ['Evaluated dimensions', `${evaluatedCount} / ${METRIC_ORDER.length}`],
          ['N/A dimensions', notApplicableCount],
          ['Tables', breakdown.tables.length],
          ['Columns evaluated', breakdown.columns.length],
        ]}
      />

      {/* Dataset-level metric matrix */}
      <div className="table table--metrics" style={{ marginTop: 12 }}>
        <div className="table-head">
          <span>Dimension (dataset)</span>
          <span className="num">Score</span>
          <span className="num">Applicable</span>
          <span className="num">Failed</span>
        </div>

        {METRIC_ORDER.map((metric) => {
          const entry = datasetMetrics[metric]
          if (!entry) return null
          return (
            <div className="table-row" key={metric}>
              <span className="cell-strong">{METRIC_LABELS[metric]}</span>
              <span className="num">
                <MetricEntryBadge entry={entry} />
              </span>
              <span className="num">{formatNumber(entry.applicable ?? null)}</span>
              <span className="num">{formatNumber(entry.failed ?? null)}</span>
            </div>
          )
        })}
      </div>

      {/* Table-level scores */}
      {breakdown.tables.length > 1 && (
        <>
          <p className="semantic-heading" style={{ marginTop: 16 }}>
            Table scores
          </p>
          <div className="table table--evidence">
            <div className="table-head">
              <span>Table</span>
              <span className="num">Columns</span>
              <span className="num">Evaluated dims</span>
              <span className="num">Table score</span>
            </div>

            {breakdown.tables.map((table) => (
              <div className="table-row" key={table.table_name}>
                <span className="cell-strong mono">{table.table_name}</span>
                <span className="num">{table.column_count}</span>
                <span className="num">{table.evaluated_metrics} / {METRIC_ORDER.length}</span>
                <span className="num">
                  <MetricEntryBadge
                    entry={{
                      status: table.table_score === null ? 'N/A' : 'ok',
                      score: table.table_score,
                    }}
                  />
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Column-level detail (per column: 7-dimension matrix) */}
      <p className="semantic-heading" style={{ marginTop: 16 }}>
        Column scores
      </p>
      <div className="table table--evidence">
        <div className="table-head">
          <span>Column</span>
          <span className="num">Score</span>
          <span className="num">Evaluated</span>
          <span className="num">Applicable rows</span>
        </div>

        {breakdown.columns.map((column) => (
          <details key={`${column.table_name}.${column.column_name}`}>
            <summary className="table-row" style={{ cursor: 'pointer' }}>
              <span className="cell-strong mono">{column.column_name}</span>
              <span className="num">
                <MetricEntryBadge
                  entry={{
                    status: column.column_score === null ? 'N/A' : 'ok',
                    score: column.column_score,
                  }}
                />
              </span>
              <span className="num">
                {column.evaluated_metrics} / {METRIC_ORDER.length}
              </span>
              <span className="num">{formatNumber(column.applicable_rows_total)}</span>
            </summary>

            <div style={{ padding: '8px 12px 16px' }}>
              {METRIC_ORDER.map((metric) => {
                const entry = column.metric_results[metric]
                if (!entry) return null
                return (
                  <div key={metric} className="rca-group">
                    <p className="semantic-heading" style={{ margin: '4px 0' }}>
                      {METRIC_LABELS[metric]} <MetricEntryBadge entry={entry} />
                    </p>
                    {entry.rules && entry.rules.length > 0 && (
                      <ul className="rca-distribution">
                        {entry.rules.map((rule) => (
                          <li key={rule.rule_id}>
                            <span>
                              {rule.rule_code ? `${rule.rule_code} ` : ''}
                              {rule.rule_name}
                              {rule.status === 'EXECUTION_ERROR' && ' — ERROR'}
                              {rule.status === 'NOT_EXECUTED' && ' — not executed'}
                            </span>
                            <span className="evidence-value">
                              {rule.status === 'ok' && rule.pass_rate !== null
                                ? `${rule.pass_rate.toFixed(2)}%`
                                : rule.status === 'ok'
                                  ? 'N/A'
                                  : rule.status}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )
              })}
            </div>
          </details>
        ))}
      </div>
    </div>
  )
}

export function ScoringStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [score, setScore] = useState<ScoreData | null>(null)
  const [rca, setRca] = useState<RCAResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const loadScore = useCallback(async () => {
    setLoading(true)

    try {
      const data = await apiFetch<ScoreData>(`/datasets/${datasetId}/scores`)
      setScore(data)
    } catch (err) {
      setScore(null)
      setError(apiErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  const loadRca = useCallback(async () => {
    try {
      const data = await apiFetch<RCAResponse>(`/datasets/${datasetId}/rca`)
      setRca(data)
    } catch (err) {
      setRca({ findings: [], message: apiErrorMessage(err) })
    }
  }, [datasetId])

  useEffect(() => {
    void loadScore()
    void loadRca()
  }, [loadScore, loadRca])

  const runRca = async () => {
    setBusy(true)
    setError('')

    try {
      const data = await apiFetch<RCAResponse>(
        `/datasets/${datasetId}/rca`,
        jsonInit('POST', {}),
      )
      setRca(data)
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const recalculateScore = async () => {
    setBusy(true)
    setError('')

    try {
      const data = await apiFetch<ScoreData>(
        `/datasets/${datasetId}/scores`,
        jsonInit('POST', {}),
      )
      setScore(data)
      setMessage('DQ score recalculated from the latest executions.')
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="container page">
      <PageHeader
        title="DQ scoring & root cause analysis"
        description="The overall score summarizes metric results; it never hides failing rules. RCA reports observed concentrations using correlation language only."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="DQ score"
          description="Mean of available metric scores. N/A metrics and critical failures are listed explicitly."
          action={
            <div className="rule-toolbar">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={busy}
                onClick={() => void recalculateScore()}
              >
                Recalculate score
              </button>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={busy}
                onClick={() => void runRca()}
              >
                Run RCA
              </button>
            </div>
          }
        />

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="warning" title={error} />}
        </div>

        {loading && <p className="empty-state">Loading score…</p>}

        {!loading && score && (
          <>
            <SummaryGrid
              items={[
                [
                  'Overall DQ score',
                  <StatusBadge key="score" tone={scoreTone(score.overall_score)}>
                    {score.overall_score.toFixed(2)}%
                  </StatusBadge>,
                ],
                ['Metrics scored', score.metric_count],
                ['N/A metrics', score.excluded_metrics],
              ]}
            />

            <div className="section">
              <div className="section-head">
                <h3 className="section-title">Metric breakdown</h3>
              </div>

              <div className="table table--metrics">
                <div className="table-head">
                  <span>Metric</span>
                  <span className="num">Score</span>
                  <span className="num">Applicable</span>
                  <span className="num">Failed</span>
                  <span className="num">Rules</span>
                </div>

                {score.details.metrics.map((metric) => (
                  <div className="table-row" key={metric.metric}>
                    <span className="cell-strong">{metric.metric}</span>
                    <span className="num">
                      <StatusBadge tone={scoreTone(metric.score)}>
                        {metric.score.toFixed(2)}%
                      </StatusBadge>
                    </span>
                    <span className="num">{formatNumber(metric.applicable_records)}</span>
                    <span className="num">{formatNumber(metric.failed_records)}</span>
                    <span className="num">{metric.rule_count}</span>
                  </div>
                ))}
              </div>

              {score.details.not_applicable_metrics.length > 0 && (
                <p className="cell-sub" style={{ marginTop: 12 }}>
                  N/A (no applicable records):{' '}
                  {score.details.not_applicable_metrics.join(', ')}
                </p>
              )}
            </div>

            {score.details.column_breakdown &&
              !('error' in score.details.column_breakdown) && (
                <ColumnBreakdownView breakdown={score.details.column_breakdown} />
              )}

            {score.details.critical_failures.length > 0 && (
              <div className="card-messages">
                <Notice tone="danger" title="Critical rule failures visible below the overall score">
                  <ul className="notice-list">
                    {score.details.critical_failures.map((failure) => (
                      <li className="notice-item" key={failure.metric}>
                        <span className="notice-item-name">
                          {failure.metric}: {failure.score.toFixed(2)}%
                        </span>
                        <span className="notice-item-meta">
                          {formatNumber(failure.failed_records)} failed of{' '}
                          {formatNumber(failure.applicable_records)} applicable records
                        </span>
                      </li>
                    ))}
                  </ul>
                </Notice>
              </div>
            )}
          </>
        )}

        {rca && rca.findings.length > 0 && (
          <div className="section">
            <div className="section-head">
              <h3 className="section-title">Root cause analysis findings</h3>

              <p className="section-hint">
                {rca.findings[0]?.analysis.language_note ??
                  'Concentrations are observational, not causal.'}
              </p>
            </div>

            {rca.findings.map((finding, index) => (
              <details className="rca-finding" key={finding.rule_id ?? index}>
                <summary className="rca-summary">
                  <span className="semantic-title">
                    <strong>{finding.rule_name ?? `Rule ${finding.rule_id}`}</strong>
                    <span>
                      {formatNumber(finding.failure_count)} failures
                      {finding.analysis.failure_percentage !== undefined
                        ? ` · ${finding.analysis.failure_percentage.toFixed(1)}% violation rate`
                        : ''}
                    </span>
                  </span>
                </summary>

                <div className="rca-body">
                  {finding.analysis.patterns?.map((pattern) => (
                    <div key={pattern.column} className="rca-group">
                      <p className="semantic-heading">
                        Value patterns — {pattern.column}
                      </p>

                      <ul className="rca-distribution">
                        {pattern.distribution.map((item) => (
                          <li key={item.value}>
                            <span>{item.value.replace(/_/g, ' ')}</span>
                            <span className="evidence-value">
                              {formatNumber(item.count)} ({item.percentage}%)
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}

                  {finding.analysis.concentration?.map((concentration) => (
                    <div key={concentration.column} className="rca-group">
                      <p className="semantic-heading">
                        Concentrated in — {concentration.column}
                      </p>

                      <ul className="rca-distribution">
                        {concentration.distribution.map((item) => (
                          <li key={item.value}>
                            <span>{item.value}</span>
                            <span className="evidence-value">
                              {formatNumber(item.count)} ({item.percentage}%)
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}

                  {finding.analysis.status !== 'analyzed' && (
                    <p className="cell-sub">
                      {finding.analysis.message ?? 'Analysis unavailable.'}
                    </p>
                  )}
                </div>
              </details>
            ))}
          </div>
        )}

        <div className="card-footer">
          <button type="button" className="btn btn-primary" onClick={onStageComplete}>
            Continue to remediation
          </button>
        </div>
      </section>
    </main>
  )
}

export function RemediationStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [remediations, setRemediations] = useState<RemediationSummary[]>([])
  const [proposal, setProposal] = useState<{
    remediation_id: number
    proposal: {
      total_corrections: number
      affected_rows: number
      risk: string
      safety_note: string
      corrections: Array<{
        table: string
        column: string
        affected_rows: number
        samples: Array<{
          row_index: number
          before: string
          after: string
        }>
      }>
    }
  } | null>(null)
  const [versions, setVersions] = useState<VersionItem[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const load = useCallback(async () => {
    try {
      const data = await apiFetch<{ remediations: RemediationSummary[] }>(
        `/datasets/${datasetId}/remediation`,
      )
      setRemediations(data.remediations)

      const versionData = await apiFetch<{ versions: VersionItem[] }>(
        `/datasets/${datasetId}/versions`,
      )
      setVersions(versionData.versions)
    } catch (err) {
      setError(apiErrorMessage(err))
    }
  }, [datasetId])

  useEffect(() => {
    void load()
  }, [load])

  const propose = async () => {
    setBusy(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch<{
        remediation_id: number
        proposal: {
          total_corrections: number
          affected_rows: number
          risk: string
          safety_note: string
          corrections: Array<{
            table: string
            column: string
            affected_rows: number
            samples: Array<{ row_index: number; before: string; after: string }>
          }>
        }
      }>(`/datasets/${datasetId}/remediation/propose`, jsonInit('POST', {}))

      setProposal(data)

      if (data.proposal.total_corrections === 0) {
        setMessage(
          'No safe corrections found. Whitespace normalization found nothing to fix; other correction types require source-system fixes.',
        )
      }

      await load()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const decide = async (remediationId: number, approve: boolean) => {
    setBusy(true)
    setError('')

    try {
      const data = await apiFetch<{
        message: string
        new_version_number: number
        correction_count: number
        before_score: number | null
        after_score: number | null
        score_delta: number | null
      }>(
        `/datasets/${datasetId}/remediation/${remediationId}/decision`,
        jsonInit('POST', { approve }),
      )

      setProposal(null)
      setMessage(
        approve
          ? `${data.message} ${data.correction_count} corrections applied. V${data.new_version_number} created. Score ${data.before_score ?? '—'} → ${data.after_score ?? '—'} (${data.score_delta !== null ? (data.score_delta >= 0 ? '+' : '') + data.score_delta : 'n/a'}).`
          : 'Remediation rejected.',
      )

      await load()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="container page">
      <PageHeader
        title="Remediation & reassessment"
        description="Only safe, deterministic corrections (whitespace normalization) can be applied. Every remediation creates a new immutable version — original data is never overwritten."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="Remediation proposals"
          description="Preview affected records before approving. Approval applies corrections and reassesses the new version automatically."
          action={
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={busy}
              onClick={() => void propose()}
            >
              {busy ? 'Scanning…' : 'Propose remediation'}
            </button>
          }
        />

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="danger" title={error} />}
        </div>

        {proposal && proposal.proposal.total_corrections > 0 && (
          <div className="section">
            <SummaryGrid
              items={[
                ['Corrections', proposal.proposal.total_corrections],
                ['Affected rows', proposal.proposal.affected_rows],
                ['Risk', proposal.proposal.risk],
              ]}
            />

            <p className="section-hint" style={{ margin: '12px 0' }}>
              {proposal.proposal.safety_note}
            </p>

            <div className="table table--evidence">
              <div className="table-head">
                <span>Table</span>
                <span>Column</span>
                <span className="num">Rows</span>
                <span>Preview (before → after)</span>
              </div>

              {proposal.proposal.corrections.map((correction) => (
                <div
                  className="table-row"
                  key={`${correction.table}-${correction.column}`}
                >
                  <span className="cell-strong">{correction.table}</span>
                  <span className="mono cell-sub">{correction.column}</span>
                  <span className="num">{correction.affected_rows}</span>
                  <span className="cell-sub">
                    {correction.samples
                      .slice(0, 2)
                      .map(
                        (sample) =>
                          `"${sample.before}" → "${sample.after}"`,
                      )
                      .join('; ') || '—'}
                  </span>
                </div>
              ))}
            </div>

            <div className="card-messages" style={{ marginTop: 16 }}>
              <Notice tone="warning" title="Approval required">
                <div className="notice-actions">
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    disabled={busy}
                    onClick={() => void decide(proposal.remediation_id, true)}
                  >
                    Approve and apply
                  </button>

                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={busy}
                    onClick={() => void decide(proposal.remediation_id, false)}
                  >
                    Reject
                  </button>
                </div>
              </Notice>
            </div>
          </div>
        )}

        {remediations.length > 0 && (
          <div className="section">
            <div className="section-head">
              <h3 className="section-title">Remediation history</h3>
            </div>

            <div className="table table--evidence">
              <div className="table-head">
                <span>ID</span>
                <span>Status</span>
                <span className="num">Corrections</span>
                <span className="num">Score</span>
              </div>

              {remediations.map((remediation) => (
                <div
                  className="table-row"
                  key={remediation.remediation_id}
                >
                  <span className="mono cell-sub">
                    RM-{String(remediation.remediation_id).padStart(3, '0')}
                  </span>

                  <StatusBadge
                    tone={
                      remediation.status === 'applied'
                        ? 'success'
                        : remediation.status === 'rejected'
                          ? 'danger'
                          : 'neutral'
                    }
                  >
                    {remediation.status}
                  </StatusBadge>

                  <span className="num">
                    {formatNumber(remediation.correction_count)}
                  </span>

                  <span className="num">
                    {remediation.before_score !== null
                      ? `${remediation.before_score.toFixed(1)}% → ${
                          remediation.after_score !== null
                            ? `${remediation.after_score.toFixed(1)}%`
                            : '—'
                        }`
                      : '—'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {versions.length > 0 && (
          <div className="section">
            <div className="section-head">
              <h3 className="section-title">Version lineage</h3>

              <p className="section-hint">
                Every version keeps its parent link and its own score.
              </p>
            </div>

            <div className="table table--evidence">
              <div className="table-head">
                <span>Version</span>
                <span>Parent</span>
                <span className="num">DQ score</span>
              </div>

              {versions.map((version) => (
                <div className="table-row" key={version.version_id}>
                  <span className="cell-strong">V{version.version_number}</span>
                  <span className="cell-sub">
                    {version.parent_version_id
                      ? `V${versions.find((v) => v.version_id === version.parent_version_id)?.version_number ?? version.parent_version_id}`
                      : 'None (original)'}
                  </span>
                  <span className="num">
                    {version.overall_score !== null
                      ? `${version.overall_score.toFixed(2)}%`
                      : 'Not scored'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="card-footer">
          <button type="button" className="btn btn-primary" onClick={onStageComplete}>
            Continue to monitoring
          </button>
        </div>
      </section>
    </main>
  )
}
