import { useCallback, useEffect, useState } from 'react'
import { apiErrorMessage, apiFetch, jsonInit } from '../api'
import {
  CardHeader,
  Notice,
  PageHeader,
  StatusBadge,
  SummaryGrid,
} from '../ui'

type DriftResponse = {
  baseline_version_number: number
  comparison_version_number: number
  schema_changes: Array<{ type: string }>
  column_drift: Array<{
    column: string
    method: string
    statistic: number
    threshold: number
    drift_detected: boolean
  }>
  insufficient_sample: Array<{
    column: string
    baseline_count: number
    comparison_count: number
    minimum_required: number
  }>
  methods: Record<string, string>
  error?: string
  summary?: {
    columns_compared: number
    drift_columns: number
    schema_changed: boolean
    insufficient_sample_count: number
  }
}

type ModelItem = {
  model_version_id: number
  model_name: string
  version: number
  status: string
  metrics: Record<string, unknown>
  trained_at: string | null
  promoted_at: string | null
}

type FeedbackSummary = {
  total: number
  decisions: Record<string, number>
}

export function MonitoringStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [result, setResult] = useState<DriftResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const runMonitoring = async () => {
    setBusy(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch<DriftResponse>(
        `/datasets/${datasetId}/monitoring`,
        jsonInit('POST', {}),
      )

      setResult(data)
      setMessage(
        data.summary
          ? `Compared ${data.summary.columns_compared} columns against V${data.baseline_version_number}. ${data.summary.drift_columns} column(s) drifted, schema ${data.summary.schema_changed ? 'changed' : 'unchanged'}.`
          : (data.error ?? 'Monitoring complete.'),
      )
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="container page">
      <PageHeader
        title="Monitoring & drift detection"
        description="Compares the latest version against an explicit baseline version. Small samples are reported as insufficient instead of producing misleading alerts."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="Drift detection"
          description="Numeric columns use the Kolmogorov–Smirnov statistic; categorical columns use PSI; schema changes are detected via fingerprints."
          action={
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={busy}
              onClick={() => void runMonitoring()}
            >
              {busy ? 'Comparing…' : 'Run monitoring'}
            </button>
          }
        />

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="danger" title={error} />}
        </div>

        {!result && !error && (
          <p className="empty-state">
            Run monitoring to compare the current version with the previous
            version as baseline. A second version is required.
          </p>
        )}

        {result && (
          <>
            <SummaryGrid
              items={[
                ['Baseline', `V${result.baseline_version_number}`],
                ['Comparison', `V${result.comparison_version_number}`],
                [
                  'Schema',
                  <StatusBadge
                    key="schema"
                    tone={result.schema_changes.length > 0 ? 'warning' : 'success'}
                  >
                    {result.schema_changes.length > 0 ? 'Changed' : 'Unchanged'}
                  </StatusBadge>,
                ],
                ['Drifted columns', result.summary?.drift_columns ?? 0],
              ]}
            />

            {result.column_drift.length > 0 && (
              <div className="section">
                <div className="section-head">
                  <h3 className="section-title">Column drift</h3>
                </div>

                <div className="table table--evidence">
                  <div className="table-head">
                    <span>Column</span>
                    <span>Method</span>
                    <span className="num">Statistic</span>
                    <span className="num">Threshold</span>
                    <span>Result</span>
                  </div>

                  {result.column_drift.map((item) => (
                    <div className="table-row" key={item.column}>
                      <span className="cell-strong">{item.column}</span>
                      <span className="badge">{item.method.toUpperCase()}</span>
                      <span className="num">{item.statistic.toFixed(4)}</span>
                      <span className="num">{item.threshold}</span>
                      <StatusBadge
                        tone={item.drift_detected ? 'warning' : 'success'}
                      >
                        {item.drift_detected ? 'Drift detected' : 'Stable'}
                      </StatusBadge>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.insufficient_sample.length > 0 && (
              <div className="card-messages">
                <Notice
                  tone="warning"
                  title={`Insufficient sample for ${result.insufficient_sample.length} column(s)`}
                >
                  <ul className="notice-list">
                    {result.insufficient_sample.map((item) => (
                      <li className="notice-item" key={item.column}>
                        <span className="notice-item-name">{item.column}</span>
                        <span className="notice-item-meta">
                          {item.baseline_count} vs {item.comparison_count} rows —
                          minimum {item.minimum_required} required
                        </span>
                      </li>
                    ))}
                  </ul>
                </Notice>
              </div>
            )}
          </>
        )}

        <div className="card-footer">
          <button type="button" className="btn btn-primary" onClick={onStageComplete}>
            Continue to feedback
          </button>
        </div>
      </section>
    </main>
  )
}

export function FeedbackStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [feedback, setFeedback] = useState<FeedbackSummary | null>(null)
  const [models, setModels] = useState<ModelItem[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const load = useCallback(async () => {
    try {
      const summary = await apiFetch<FeedbackSummary>(
        `/datasets/${datasetId}/semantic/feedback`,
      )
      setFeedback(summary)

      const modelData = await apiFetch<{ models: ModelItem[] }>('/models')
      setModels(modelData.models)
    } catch (err) {
      setError(apiErrorMessage(err))
    }
  }, [datasetId])

  useEffect(() => {
    void load()
  }, [load])

  const train = async () => {
    setBusy(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch<{
        status: string
        message?: string
        version?: number
        status_note?: string
      }>('/models/train', jsonInit('POST', {}))

      if (data.status === 'trained') {
        setMessage(
          `Candidate model v${data.version} trained. ${data.status_note ?? ''}`,
        )
      } else {
        setMessage(data.message ?? 'Training not performed.')
      }

      await load()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const promote = async (modelVersionId: number) => {
    setBusy(true)
    setError('')

    try {
      await apiFetch(
        '/models/promote',
        jsonInit('POST', { model_version_id: modelVersionId }),
      )

      setMessage('Model promoted. It is now the active reranker definition.')

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
        title="Feedback & offline learning"
        description="Human decisions from every stage are stored as versioned feedback. Models are trained offline, evaluated on held-out data, and only activated through explicit promotion."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="Semantic feedback"
          description="Approved, edited and rejected mappings recorded on this dataset."
        />

        {feedback && (
          <SummaryGrid
            items={[
              ['Total feedback', feedback.total],
              ['Approved', feedback.decisions.approved ?? 0],
              ['Edited', feedback.decisions.edited ?? 0],
              ['Rejected', feedback.decisions.rejected ?? 0],
            ]}
          />
        )}

        {feedback && feedback.total === 0 && (
          <p className="empty-state">
            No feedback recorded yet. Review semantic predictions in stage 05.
          </p>
        )}
      </section>

      <section className="card" style={{ marginTop: 16 }}>
        <CardHeader
          title="Model registry"
          description="Offline-trained reranker candidates. Training runs only with sufficient labeled feedback and never uses model predictions as labels."
          action={
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={busy}
              onClick={() => void train()}
            >
              {busy ? 'Training…' : 'Train candidate model'}
            </button>
          }
        />

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="danger" title={error} />}
        </div>

        {models.length === 0 && (
          <p className="empty-state">
            No trained models yet. Training requires at least 20 labeled human
            decisions covering both positive and negative examples.
          </p>
        )}

        {models.length > 0 && (
          <div className="section">
            <div className="table table--evidence">
              <div className="table-head">
                <span>Model</span>
                <span>Version</span>
                <span>Status</span>
                <span className="num">Held-out accuracy</span>
                <span>Action</span>
              </div>

              {models.map((model) => (
                <div className="table-row" key={model.model_version_id}>
                  <span className="cell-strong">{model.model_name}</span>
                  <span className="mono cell-sub">v{model.version}</span>

                  <StatusBadge
                    tone={
                      model.status === 'promoted'
                        ? 'success'
                        : model.status === 'candidate'
                          ? 'accent'
                          : 'neutral'
                    }
                  >
                    {model.status}
                  </StatusBadge>

                  <span className="num">
                    {typeof model.metrics.accuracy === 'number'
                      ? `${(model.metrics.accuracy * 100).toFixed(1)}%`
                      : '—'}
                  </span>

                  <span>
                    {model.status === 'candidate' && (
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        disabled={busy}
                        onClick={() => void promote(model.model_version_id)}
                      >
                        Promote
                      </button>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="card-footer">
          <button type="button" className="btn btn-primary" onClick={onStageComplete}>
            Back to dataset overview
          </button>
        </div>
      </section>
    </main>
  )
}
