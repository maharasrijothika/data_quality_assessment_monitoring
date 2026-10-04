import { useCallback, useEffect, useState } from 'react'
import { apiErrorMessage, apiFetch, jsonInit } from '../api'
import {
  CardHeader,
  Notice,
  PageHeader,
  StatusBadge,
  SummaryGrid,
  ruleStatusTone,
} from '../ui'

type ApplicabilityMetric = {
  status: 'APPLICABLE' | 'NOT_APPLICABLE' | 'NEEDS_REVIEW'
  evidence: string[]
}

type ApplicabilityColumn = {
  table_name: string
  column_name: string
  semantic_type: string | null
  semantic_status: string | null
  metrics: Record<string, ApplicabilityMetric>
}

type RuleItem = {
  rule_id: number
  table_name: string
  rule_name: string
  rule_json: Record<string, any>
  metric: string
  rule_template: string | null
  parameters: Record<string, any>
  source: string
  risk: string
  rule_code: string | null
  version_number: number
  validation_status: string | null
  validation_issues: Array<{ severity: string; message: string }>
  status: string
}

type ExecutionItem = {
  execution_id: number
  rule_id: number
  rule_code: string | null
  rule_name: string | null
  metric: string | null
  rule_version_number?: number | null
  status: string
  error_message?: string | null
  total_rows: number
  applicable_rows: number
  passed_rows: number
  failed_rows: number
  pass_rate: number | null
  violation_rate: number | null
  failure_examples: Array<Record<string, unknown>>
}

const METRIC_LABELS: Record<string, string> = {
  completeness: 'Completeness',
  uniqueness: 'Uniqueness',
  validity: 'Validity',
  accuracy: 'Accuracy',
  consistency: 'Consistency',
  referential_integrity: 'Referential Integrity',
  timeliness: 'Timeliness',
}

const METRIC_ORDER = Object.keys(METRIC_LABELS)

const formatNumber = (value: number | null): string =>
  typeof value === 'number' ? value.toLocaleString() : '—'

const formatPercent = (value: number | null): string =>
  typeof value === 'number' ? `${value.toFixed(2)}%` : 'N/A'

function MetricTone({ status }: { status: string }) {
  if (status === 'APPLICABLE') return <StatusBadge tone="success">Applicable</StatusBadge>
  if (status === 'NEEDS_REVIEW') return <StatusBadge tone="warning">Needs review</StatusBadge>
  return <StatusBadge tone="neutral">N/A</StatusBadge>
}

function ApplicabilityMatrix({ columns }: { columns: ApplicabilityColumn[] }) {
  return (
    <div className="table table--evidence">
      <div className="table-head">
        <span>Column</span>
        <span>Semantic type</span>
        <span>Metrics (Applicable / Review / N/A)</span>
      </div>

      {columns.map((column) => {
        const applicable = METRIC_ORDER.filter(
          (m) => column.metrics[m]?.status === 'APPLICABLE',
        )
        const review = METRIC_ORDER.filter(
          (m) => column.metrics[m]?.status === 'NEEDS_REVIEW',
        )

        return (
          <details key={`${column.table_name}.${column.column_name}`}>
            <summary className="table-row" style={{ cursor: 'pointer' }}>
              <span className="cell-strong mono">{column.column_name}</span>
              <span className="cell-sub">{column.semantic_type ?? '—'}</span>
              <span className="cell-sub">
                <span className="evidence-value">
                  {applicable.length}/{METRIC_ORDER.length}
                </span>{' '}
                applicable
                {review.length > 0 && ` · ${review.length} need review`}
              </span>
            </summary>

            <div style={{ padding: '8px 12px 16px' }}>
              {METRIC_ORDER.map((metric) => {
                const entry = column.metrics[metric]
                if (!entry) return null
                return (
                  <div key={metric} className="rca-group">
                    <p className="semantic-heading" style={{ margin: '6px 0 2px' }}>
                      {METRIC_LABELS[metric]}{' '}
                      <MetricTone status={entry.status} />
                    </p>
                    <ul className="rca-distribution">
                      {entry.evidence.map((line, index) => (
                        <li key={index}>
                          <span>{line}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )
              })}
            </div>
          </details>
        )
      })}
    </div>
  )
}

function RuleReviewCard({
  rule,
  busy,
  onDecide,
}: {
  rule: RuleItem
  busy: boolean
  onDecide: (
    ruleId: number,
    decision: 'approved' | 'rejected',
    editedJson?: Record<string, any>,
    editedName?: string,
  ) => void
}) {
  const [editing, setEditing] = useState(false)

  // Editable parameter form (flat: one input per registry parameter).
  const params = rule.parameters ?? {}
  const [minValue, setMinValue] = useState(params.min != null ? String(params.min) : '')
  const [maxValue, setMaxValue] = useState(params.max != null ? String(params.max) : '')
  const [allowedText, setAllowedText] = useState(
    Array.isArray(params.allowed_values) ? params.allowed_values.join(', ') : '',
  )
  const [maxAge, setMaxAge] = useState(params.max_age?.amount != null ? String(params.max_age.amount) : '')
  const [maxAgeUnit, setMaxAgeUnit] = useState(params.max_age?.unit ?? 'hours')
  const [nameDraft, setNameDraft] = useState(rule.rule_name)

  const buildEditedJson = (): Record<string, any> => {
    const next: Record<string, any> = { ...rule.rule_json, parameters: { ...(rule.parameters ?? {}) } }
    if (rule.rule_template === 'NUMERIC_RANGE') {
      next.parameters.min = minValue === '' ? null : Number(minValue)
      next.parameters.max = maxValue === '' ? null : Number(maxValue)
    }
    if (rule.rule_template === 'ALLOWED_VALUES') {
      next.parameters.allowed_values = allowedText
        .split(',')
        .map((v) => v.trim())
        .filter(Boolean)
    }
    if (rule.rule_template === 'FRESHNESS_THRESHOLD') {
      next.parameters.max_age = { amount: Number(maxAge), unit: maxAgeUnit }
    }
    return next
  }

  return (
    <div className="rule-card">
      <div className="rule-head">
        <div className="rule-title">
          <strong>{rule.rule_name}</strong>
          <span className="cell-sub">
            {rule.table_name}
            {rule.rule_code ? ` · ${rule.rule_code} v${rule.version_number}` : ''}
            {rule.source === 'user' ? ' · business rule' : ''}
          </span>
        </div>

        <div className="rule-badges">
          <span className="badge">{METRIC_LABELS[rule.metric] ?? rule.metric}</span>
          <span className="badge">{rule.rule_template ?? '?'}</span>
          <StatusBadge tone={ruleStatusTone(rule.status)}>{rule.status}</StatusBadge>
          {rule.validation_status && (
            <StatusBadge tone={ruleStatusTone(rule.validation_status)}>
              {rule.validation_status === 'VALID'
                ? 'VALID'
                : rule.validation_status === 'NEEDS_REVIEW'
                  ? 'NEEDS REVIEW'
                  : 'INVALID'}
            </StatusBadge>
          )}
        </div>
      </div>

      {!editing && (
        <div className="rule-actions">
          {rule.status === 'recommended' && (
            <>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={busy || rule.validation_status === 'INVALID'}
                onClick={() => onDecide(rule.rule_id, 'approved')}
              >
                Accept
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={busy}
                onClick={() => onDecide(rule.rule_id, 'rejected')}
              >
                Reject
              </button>
            </>
          )}

          {(rule.status === 'recommended' || rule.status === 'approved') && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={busy}
              onClick={() => {
                setEditing((current) => !current)
                setNameDraft(rule.rule_name)
              }}
            >
              {editing ? 'Cancel edit' : 'Edit'}
            </button>
          )}
        </div>
      )}

      {editing && (
        <div className="rule-detail">
          <div className="manual-entry-field">
            <label>Rule name</label>
            <input
              className="manual-entry-input"
              value={nameDraft}
              onChange={(event) => setNameDraft(event.target.value)}
            />
          </div>

          {rule.rule_template === 'NUMERIC_RANGE' && (
            <>
              <div className="manual-entry-field">
                <label>Min (blank = none)</label>
                <input
                  className="manual-entry-input"
                  value={minValue}
                  onChange={(event) => setMinValue(event.target.value)}
                />
              </div>
              <div className="manual-entry-field">
                <label>Max (blank = none)</label>
                <input
                  className="manual-entry-input"
                  value={maxValue}
                  onChange={(event) => setMaxValue(event.target.value)}
                />
              </div>
            </>
          )}

          {rule.rule_template === 'ALLOWED_VALUES' && (
            <div className="manual-entry-field">
              <label>Allowed values (comma separated)</label>
              <input
                className="manual-entry-input"
                value={allowedText}
                onChange={(event) => setAllowedText(event.target.value)}
              />
            </div>
          )}

          {rule.rule_template === 'FRESHNESS_THRESHOLD' && (
            <>
              <div className="manual-entry-field">
                <label>Max age (SLA)</label>
                <input
                  className="manual-entry-input"
                  value={maxAge}
                  onChange={(event) => setMaxAge(event.target.value)}
                />
              </div>
              <div className="manual-entry-field">
                <label>Unit</label>
                <select
                  className="manual-entry-input"
                  value={maxAgeUnit}
                  onChange={(event) => setMaxAgeUnit(event.target.value)}
                >
                  <option value="minutes">minutes</option>
                  <option value="hours">hours</option>
                  <option value="days">days</option>
                  <option value="weeks">weeks</option>
                </select>
              </div>
            </>
          )}

          {rule.rule_template !== 'NUMERIC_RANGE' &&
            rule.rule_template !== 'ALLOWED_VALUES' &&
            rule.rule_template !== 'FRESHNESS_THRESHOLD' && (
              <p className="cell-sub">This template has no editable parameters.</p>
            )}

          <div className="notice-actions" style={{ marginTop: 10 }}>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={busy}
              onClick={() => {
                onDecide(rule.rule_id, 'approved', buildEditedJson(), nameDraft || undefined)
                setEditing(false)
              }}
            >
              {rule.status === 'approved' ? 'Save as new version' : 'Validate & approve'}
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setEditing(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {!editing && (
        <details>
          <summary className="cell-sub" style={{ cursor: 'pointer', marginTop: 6 }}>
            Show evidence & definition
          </summary>
          <div className="rule-detail">
            <code className="code">{JSON.stringify(rule.rule_json, null, 2)}</code>
            {rule.validation_issues.length > 0 && (
              <ul className="validation-issues">
                {rule.validation_issues.map((issue, index) => (
                  <li key={index} className={`validation-issue severity-${issue.severity}`}>
                    {issue.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </details>
      )}
    </div>
  )
}

function BusinessRuleForm({
  datasetId,
  tables,
  onCreated,
}: {
  datasetId: number
  tables: string[]
  onCreated: () => void
}) {
  const [text, setText] = useState('')
  const [table, setTable] = useState(tables[0] ?? '')
  const [column, setColumn] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  type InterpretResult = {
    status: 'RESOLVED' | 'UNRESOLVED'
    metric?: string | null
    rule_template?: string | null
    parameters?: Record<string, any>
    target?: { table: string | null; columns: string[] }
    notes?: string[]
    metric_options?: string[]
    validation?: { status: string; issues: Array<{ severity: string; message: string }> } | null
    candidate_json?: Record<string, any>
  }

  const [interpretation, setInterpretation] = useState<InterpretResult | null>(null)

  const interpret = async () => {
    setBusy(true)
    setError('')
    setInterpretation(null)

    try {
      const data = await apiFetch<InterpretResult>(
        `/datasets/${datasetId}/rules/interpret`,
        jsonInit('POST', { text, table_name: table || null, column_name: column || null }),
      )
      setInterpretation(data)
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const createFromInterpretation = async (metric: string, template: string, parameters: Record<string, any>, columns: string[]) => {
    setBusy(true)
    setError('')

    try {
      await apiFetch(
        `/datasets/${datasetId}/rules/manual`,
        jsonInit('POST', {
          metric,
          rule_template: template,
          table_name: table,
          column_name: columns[0] ?? column ?? null,
          columns,
          parameters,
          rule_name: text,
          original_text: text,
        }),
      )
      setText('')
      setInterpretation(null)
      onCreated()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="manual-entry">
      <div className="key-entry-fields">
        <div className="manual-entry-field">
          <label>Business rule text</label>
          <input
            className="manual-entry-input"
            placeholder='e.g. "Age must be between 18 and 65."'
            value={text}
            onChange={(event) => setText(event.target.value)}
          />
        </div>
        <div className="manual-entry-field">
          <label>Table</label>
          <select
            className="manual-entry-input"
            value={table}
            onChange={(event) => setTable(event.target.value)}
          >
            {tables.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <div className="manual-entry-field">
          <label>Column (optional)</label>
          <input
            className="manual-entry-input"
            value={column}
            onChange={(event) => setColumn(event.target.value)}
          />
        </div>
      </div>

      <button
        type="button"
        className="btn btn-secondary btn-sm"
        disabled={busy || text.trim() === ''}
        onClick={() => void interpret()}
      >
        {busy ? 'Interpreting…' : 'Interpret rule text'}
      </button>

      {error && (
        <div className="card-messages">
          <Notice tone="danger" title={error} />
        </div>
      )}

      {interpretation && interpretation.status === 'RESOLVED' && (
        <div className="rule-detail" style={{ marginTop: 10 }}>
          <p className="semantic-heading">
            Detected: {METRIC_LABELS[interpretation.metric ?? ''] ?? interpretation.metric} ·{' '}
            {interpretation.rule_template}
          </p>
          <p className="cell-sub mono">
            {JSON.stringify(interpretation.parameters)}
          </p>
          {interpretation.notes?.map((note, index) => (
            <p key={index} className="cell-sub">
              {note}
            </p>
          ))}
          {interpretation.validation && (
            <StatusBadge tone={ruleStatusTone(interpretation.validation.status)}>
              {interpretation.validation.status}
            </StatusBadge>
          )}
          <div className="notice-actions" style={{ marginTop: 8 }}>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={busy || !interpretation.candidate_json}
              onClick={() =>
                void createFromInterpretation(
                  interpretation.metric!,
                  interpretation.rule_template!,
                  interpretation.parameters ?? {},
                  interpretation.target?.columns ?? (column ? [column] : []),
                )
              }
            >
              Add as candidate for review
            </button>
          </div>
        </div>
      )}

      {interpretation && interpretation.status === 'UNRESOLVED' && (
        <div className="card-messages" style={{ marginTop: 10 }}>
          <Notice tone="warning" title="Unable to determine the metric safely — nothing was guessed.">
            <p className="notice-text">
              Choose a metric and add the rule with explicit parameters via the
              template form below. Possible metrics:{' '}
              {(interpretation.metric_options ?? [])
                .map((m) => METRIC_LABELS[m] ?? m)
                .join(', ')}
            </p>
          </Notice>
        </div>
      )}
    </div>
  )
}

function ManualTemplateForm({
  datasetId,
  templates,
  tables,
  onCreated,
}: {
  datasetId: number
  templates: Array<{
    name: string
    metric: string
    target: string
    parameters: Array<{ name: string; kind: string; required: boolean }>
    requires: string[]
    description: string
  }>
  tables: string[]
  onCreated: () => void
}) {
  const [metric, setMetric] = useState('validity')
  const [templateName, setTemplateName] = useState('NUMERIC_RANGE')
  const [table, setTable] = useState(tables[0] ?? '')
  const [column, setColumn] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const metricTemplates = templates.filter((t) => t.metric === metric)
  const template = metricTemplates.find((t) => t.name === templateName) ?? metricTemplates[0]

  const paramValues: Record<string, string> = {}
  const [paramState, setParamState] = useState<Record<string, string>>({})

  const submit = async () => {
    if (!template) return
    setBusy(true)
    setError('')
    setMessage('')

    const parameters: Record<string, any> = {}
    for (const spec of template.parameters) {
      const raw = paramState[spec.name] ?? ''
      if (raw === '') continue
      if (spec.kind === 'number' || spec.kind === 'number_or_none') {
        parameters[spec.name] = Number(raw)
      } else if (spec.kind === 'string_list') {
        parameters[spec.name] = raw.split(',').map((v) => v.trim()).filter(Boolean)
      } else if (spec.kind === 'boolean') {
        parameters[spec.name] = raw.toLowerCase() === 'true'
      } else if (spec.kind === 'interval') {
        // Input format: "24 hours" -> {amount, unit}
        const [amount, unit] = raw.split(/\s+/)
        parameters[spec.name] = { amount: Number(amount), unit: unit ?? 'hours' }
      } else {
        parameters[spec.name] = raw
      }
    }

    try {
      const columns = column ? [column] : []
      const data = await apiFetch<{ message: string }>(
        `/datasets/${datasetId}/rules/manual`,
        jsonInit('POST', {
          metric: template.metric,
          rule_template: template.name,
          table_name: table,
          column_name: columns[0] ?? null,
          columns,
          parameters,
        }),
      )
      setMessage(data.message)
      onCreated()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="manual-entry">
      <div className="key-entry-fields">
        <div className="manual-entry-field">
          <label>Metric</label>
          <select
            className="manual-entry-input"
            value={metric}
            onChange={(event) => {
              setMetric(event.target.value)
              const first = templates.find((t) => t.metric === event.target.value)
              setTemplateName(first?.name ?? '')
            }}
          >
            {Object.entries(METRIC_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
        <div className="manual-entry-field">
          <label>Template</label>
          <select
            className="manual-entry-input"
            value={template?.name ?? ''}
            onChange={(event) => setTemplateName(event.target.value)}
          >
            {metricTemplates.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name}
              </option>
            ))}
          </select>
        </div>
        <div className="manual-entry-field">
          <label>Table</label>
          <select
            className="manual-entry-input"
            value={table}
            onChange={(event) => setTable(event.target.value)}
          >
            {tables.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <div className="manual-entry-field">
          <label>Column (if applicable)</label>
          <input
            className="manual-entry-input"
            value={column}
            onChange={(event) => setColumn(event.target.value)}
          />
        </div>
      </div>

      {template && template.parameters.length > 0 && (
        <div className="key-entry-fields">
          {template.parameters.map((spec) => (
            <div className="manual-entry-field" key={spec.name}>
              <label>
                {spec.name}
                {spec.required ? ' *' : ''} ({spec.kind})
              </label>
              <input
                className="manual-entry-input"
                value={paramState[spec.name] ?? ''}
                onChange={(event) =>
                  setParamState((current) => ({ ...current, [spec.name]: event.target.value }))
                }
              />
            </div>
          ))}
        </div>
      )}

      {template && <p className="section-hint">{template.description}</p>}
      {template && template.requires.length > 0 && (
        <p className="section-hint">
          Requires: {template.requires.join(', ')} (validated before approval).
        </p>
      )}

      <div className="notice-actions" style={{ marginTop: 8 }}>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={busy || !template}
          onClick={() => void submit()}
        >
          {busy ? 'Creating…' : 'Create rule candidate'}
        </button>
      </div>

      {message && (
        <div className="card-messages">
          <Notice tone="success" title={message} />
        </div>
      )}
      {error && (
        <div className="card-messages">
          <Notice tone="danger" title={error} />
        </div>
      )}
      {paramValues && null}
    </div>
  )
}

export function RulesStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [rules, setRules] = useState<RuleItem[]>([])
  const [applicability, setApplicability] = useState<ApplicabilityColumn[] | null>(null)
  const [templates, setTemplates] = useState<
    Array<{
      name: string
      metric: string
      target: string
      parameters: Array<{ name: string; kind: string; required: boolean }>
      requires: string[]
      description: string
    }>
  >([])
  const [tables, setTables] = useState<string[]>([])
  const [showApplicability, setShowApplicability] = useState(false)
  const [showBusinessForm, setShowBusinessForm] = useState(false)
  const [showTemplateForm, setShowTemplateForm] = useState(false)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const loadRules = useCallback(async () => {
    setLoading(true)

    try {
      const data = await apiFetch<{ rules: RuleItem[] }>(
        `/datasets/${datasetId}/rules`,
      )

      setRules(data.rules)

      const tableSet = Array.from(new Set(data.rules.map((r) => r.table_name)))
      if (tableSet.length > 0) {
        setTables((current) => (current.length > 0 ? current : tableSet))
      }
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  const loadContext = useCallback(async () => {
    try {
      const appData = await apiFetch<{ columns: ApplicabilityColumn[] }>(
        `/datasets/${datasetId}/applicability`,
      )
      setApplicability(appData.columns)

      const tableSet = Array.from(new Set(appData.columns.map((c) => c.table_name)))
      setTables(tableSet)

      const tmplData = await apiFetch<{ templates: typeof templates }>(
        `/datasets/${datasetId}/rule-templates`,
      )
      setTemplates(tmplData.templates)
    } catch (err) {
      // Applicability is auxiliary; rules stage remains usable without it.
      setApplicability(null)
    }
  }, [datasetId])

  useEffect(() => {
    void loadRules()
    void loadContext()
  }, [loadRules, loadContext])

  const generateRecommendations = async () => {
    setBusy(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch<{ recommendation_count: number }>(
        `/datasets/${datasetId}/recommendations`,
        jsonInit('POST', {}),
      )

      setMessage(
        data.recommendation_count === 0
          ? 'No rule candidates generated.'
          : `${data.recommendation_count} rule candidate(s) generated from applicability evidence and profile statistics.`,
      )

      await loadRules()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const validateAll = async () => {
    setBusy(true)
    setError('')

    try {
      const data = await apiFetch<{ validated: number }>(
        `/datasets/${datasetId}/rules/validate-all`,
        jsonInit('POST', {}),
      )

      setMessage(`${data.validated} rule(s) validated.`)
      await loadRules()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const decideRule = async (
    ruleId: number,
    decision: 'approved' | 'rejected',
    editedJson?: Record<string, any>,
    editedName?: string,
  ) => {
    setError('')

    try {
      await apiFetch(
        `/rules/${ruleId}/approve`,
        jsonInit('POST', {
          decision,
          edited_rule_json: editedJson ?? undefined,
          edited_rule_name: editedName ?? undefined,
        }),
      )

      await loadRules()
    } catch (err) {
      setError(apiErrorMessage(err))
    }
  }

  const recommendedCount = rules.filter((r) => r.status === 'recommended').length
  const approvedCount = rules.filter((r) => r.status === 'approved').length
  const rejectedCount = rules.filter((r) => r.status === 'rejected').length

  return (
    <main className="container page">
      <PageHeader
        title="Metric applicability & rule recommendations"
        description="Applicability is evidence-based per column and metric. Rules are generated from controlled templates, validated before approval, and never execute without an explicit human decision. Business ranges from observed data are candidates only."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="Rule candidates"
          description="Every candidate shows its metric, template, validation status and evidence. Accept, Reject or Edit each one independently."
          action={
            <div className="rule-toolbar">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={busy}
                onClick={() => void generateRecommendations()}
              >
                {busy ? 'Working…' : 'Generate recommendations'}
              </button>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={busy || recommendedCount === 0}
                onClick={() => void validateAll()}
              >
                Validate pending
              </button>
            </div>
          }
        />

        {rules.length > 0 && (
          <SummaryGrid
            items={[
              ['Recommended', recommendedCount],
              ['Approved', approvedCount],
              ['Rejected', rejectedCount],
              [
                'Invalid',
                rules.filter((r) => r.validation_status === 'INVALID').length,
              ],
            ]}
          />
        )}

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="danger" title={error} />}
        </div>

        {loading && <p className="empty-state">Loading rules…</p>}

        {!loading && rules.length === 0 && !error && (
          <p className="empty-state">
            No rule candidates yet. Generate recommendations after profiling,
            semantic and relationship stages.
          </p>
        )}

        {rules.length > 0 && (
          <div className="rule-list">
            {rules.map((rule) => (
              <RuleReviewCard
                key={rule.rule_id}
                rule={rule}
                busy={busy}
                onDecide={(ruleId, decision, editedJson, editedName) =>
                  void decideRule(ruleId, decision, editedJson, editedName)
                }
              />
            ))}
          </div>
        )}

        <div className="card-footer">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onStageComplete}
            disabled={approvedCount === 0}
          >
            Continue to execution
          </button>
        </div>
      </section>

      <section className="card">
        <CardHeader
          title="Business rules & templates"
          description="Add rules per column: type a business rule in plain language (interpreted — never executed as text), or build one from a controlled template."
          action={
            <div className="rule-toolbar">
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => setShowBusinessForm((v) => !v)}
              >
                {showBusinessForm ? 'Hide' : '+ Add business rule'}
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => setShowTemplateForm((v) => !v)}
              >
                {showTemplateForm ? 'Hide' : '+ Build from template'}
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => setShowApplicability((v) => !v)}
              >
                {showApplicability ? 'Hide applicability' : 'Show applicability'}
              </button>
            </div>
          }
        />

        {showBusinessForm && tables.length > 0 && (
          <BusinessRuleForm
            datasetId={datasetId}
            tables={tables}
            onCreated={() => void loadRules()}
          />
        )}

        {showTemplateForm && tables.length > 0 && templates.length > 0 && (
          <ManualTemplateForm
            datasetId={datasetId}
            templates={templates}
            tables={tables}
            onCreated={() => void loadRules()}
          />
        )}

        {showApplicability && applicability && (
          <ApplicabilityMatrix columns={applicability} />
        )}

        {showApplicability && !applicability && (
          <p className="empty-state">
            Applicability requires profiling and semantic stages to be complete.
          </p>
        )}
      </section>
    </main>
  )
}

export function ExecutionStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [rules, setRules] = useState<RuleItem[]>([])
  const [executions, setExecutions] = useState<ExecutionItem[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const loadRules = useCallback(async () => {
    try {
      const data = await apiFetch<{ rules: RuleItem[] }>(
        `/datasets/${datasetId}/rules`,
      )

      setRules(data.rules)
    } catch (err) {
      setError(apiErrorMessage(err))
    }
  }, [datasetId])

  const loadExecutions = useCallback(async () => {
    setLoading(true)

    try {
      const data = await apiFetch<{ executions: ExecutionItem[] }>(
        `/datasets/${datasetId}/executions`,
      )

      setExecutions(data.executions)
    } catch {
      setExecutions([])
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  useEffect(() => {
    void loadRules()
    void loadExecutions()
  }, [loadRules, loadExecutions])

  const executeApproved = async () => {
    setBusy(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch<{
        executed: number
        errors: number
      }>(`/datasets/${datasetId}/execute`, jsonInit('POST', { rule_ids: [] }))

      setMessage(
        data.errors === 0
          ? `${data.executed} rule(s) executed successfully.`
          : `${data.executed} rule(s) executed with ${data.errors} error(s). Errored rules are reported honestly and excluded from pass rates.`,
      )

      await loadExecutions()
      await loadRules()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const approvedCount = rules.filter((r) => r.status === 'approved').length
  const errorCount = executions.filter((e) => e.status === 'error').length

  return (
    <main className="container page">
      <PageHeader
        title="Authoritative execution"
        description="Only human-approved rules execute here, through one constrained handler per template — never eval, never arbitrary code. Pass rates are N/A (never 100%) when no rows are applicable. Execution errors are persisted and surfaced."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="Execution"
          description="Results are bounded: row counts, pass rates and failure evidence — never full dataset copies."
          action={
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={busy || approvedCount === 0}
              onClick={() => void executeApproved()}
            >
              {busy
                ? 'Executing…'
                : `Execute approved rules (${approvedCount})`}
            </button>
          }
        />

        {rules.length > 0 && (
          <SummaryGrid
            items={[
              ['Approved rules', approvedCount],
              ['Stored executions', executions.length],
              ['Execution errors', errorCount],
            ]}
          />
        )}

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="danger" title={error} />}
        </div>

        {loading && <p className="empty-state">Loading executions…</p>}

        {!loading && rules.length === 0 && !error && (
          <p className="empty-state">
            No rules exist yet. Generate and approve rules in the previous
            stage first.
          </p>
        )}

        {!loading && rules.length > 0 && approvedCount === 0 && (
          <div className="card-messages">
            <Notice
              tone="warning"
              title="Approval gate: nothing will execute"
            >
              <p className="notice-text">
                No rules are approved. With 0 approved rules, 0 executions run
                and no DQ score is produced — by design.
              </p>
            </Notice>
          </div>
        )}

        {executions.length > 0 && (
          <div className="execution-list">
            {executions.map((execution) => (
              <details className="execution-item" key={execution.execution_id}>
                <summary className="execution-summary">
                  <span className="semantic-title">
                    <strong>{execution.rule_name ?? `Rule ${execution.rule_id}`}</strong>
                    <span>
                      {formatNumber(execution.failed_rows)} failed of{' '}
                      {formatNumber(execution.applicable_rows)} applicable
                    </span>
                  </span>

                  <span className="semantic-badges">
                    <StatusBadge
                      tone={
                        execution.status === 'error'
                          ? 'danger'
                          : execution.failed_rows === 0
                            ? 'success'
                            : 'warning'
                      }
                    >
                      {execution.status === 'error'
                        ? 'EXECUTION ERROR'
                        : execution.applicable_rows === 0
                          ? 'N/A (0 applicable)'
                          : execution.failed_rows === 0
                            ? 'All passed'
                            : `${formatPercent(execution.violation_rate)} violations`}
                    </StatusBadge>
                  </span>
                </summary>

                <div className="execution-body">
                  {execution.status === 'error' && (
                    <div className="card-messages">
                      <Notice
                        tone="danger"
                        title="This rule failed to execute"
                      >
                        <p className="notice-text">
                          {execution.error_message ??
                            'The error is persisted and excluded from metric scores — it can never count as a passing result.'}
                        </p>
                      </Notice>
                    </div>
                  )}

                  <SummaryGrid
                    items={[
                      ['Total rows', formatNumber(execution.total_rows)],
                      ['Applicable (A)', formatNumber(execution.applicable_rows)],
                      ['Passed (P)', formatNumber(execution.passed_rows)],
                      ['Failed (F)', formatNumber(execution.failed_rows)],
                      ['Pass rate', formatPercent(execution.pass_rate)],
                      ['Coverage', `${formatNumber(execution.applicable_rows)} / ${formatNumber(execution.total_rows)}`],
                    ]}
                  />

                  {execution.failure_examples.length > 0 && (
                    <div className="table table--evidence">
                      <div className="table-head">
                        <span>Row</span>
                        <span>Failing values</span>
                      </div>

                      {execution.failure_examples.map((example, index) => (
                        <div className="table-row" key={index}>
                          <span className="mono cell-sub">
                            #{String(example.row_index ?? '')}
                          </span>

                          <code className="code code-truncate">
                            {JSON.stringify(example)}
                          </code>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </details>
            ))}
          </div>
        )}

        <div className="card-footer">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onStageComplete}
            disabled={executions.length === 0}
          >
            Continue to scoring &amp; RCA
          </button>
        </div>
      </section>
    </main>
  )
}
