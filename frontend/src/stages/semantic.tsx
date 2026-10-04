import { useCallback, useEffect, useMemo, useState } from 'react'
import { apiErrorMessage, apiFetch, jsonInit } from '../api'
import {
  CardHeader,
  Notice,
  PageHeader,
  StatusBadge,
  SummaryGrid,
  confidenceTone,
  IconCheck,
  IconChevron,
} from '../ui'

type EvidenceItem = { value: number; applicable: boolean } | number

/** Evidence items are {value, applicable}; legacy predictions may store
 * plain numbers. Legacy values are treated as applicable evidence. */
const evidenceValue = (item: EvidenceItem): { value: number; applicable: boolean } =>
  typeof item === 'number'
    ? { value: item, applicable: true }
    : {
        value: Number(item?.value ?? 0),
        applicable: Boolean(item?.applicable),
      }

type SemanticInput = {
  table_name?: string
  column_name?: string
  data_type?: string | null
  column_description?: string | null
  dataset_domain?: string | null
  table_context?: string | null
  sibling_names?: string[] | null
  embedding_text?: string | null
  embedding_model?: string | null
  embedding_representation_version?: number | null
}

type NormalizationEvidence = {
  original_name?: string
  normalized_name?: string
  expanded_name?: string
  abbreviation_evidence?: Array<{
    token: string
    expansion: string | null
    alternatives: string[] | null
    confidence: string
  }>
}

type RunComparison = {
  comparison_available: boolean
  reason?: string
  old_run?: { run_number: number; created_at: string; model_version: string; kb_fingerprint: string }
  new_run?: { run_number: number; created_at: string; model_version: string; kb_fingerprint: string }
  columns?: Array<{
    table_name: string
    column_name: string
    old_prediction: string | null
    new_prediction: string | null
    changed: boolean
  }>
}

type Proposal = {
  name: string | null
  family: string | null
  source?: string
  ask_user: boolean
  group_key: string | null
}

type CandidateRecord = {
  concept: string
  confidence_score: number
  decision_kind?: string
  is_family?: boolean
  family?: string | null
  gates?: Record<string, boolean>
}

type SemanticPrediction = {
  prediction_id: number
  table_name: string
  column_name: string
  predicted_concept: string | null
  concept_id: number | null
  confidence_score: number
  confidence_level: string
  evidence: Record<string, EvidenceItem>
  evidence_coverage: number
  source: string
  match_status: string
  normalization?: NormalizationEvidence | null
  alternatives: Array<{ concept: string; score: number }>
  status: string
  final_semantic_type: string | null
  user_defined_type: string | null
  decision_source: string | null
  decided_at: string | null
  // v2 additive fields (absent on legacy payloads)
  semantic_input?: SemanticInput | null
  decision_kind?: string | null
  semantic_family?: string | null
  proposal?: Proposal | null
  candidates?: CandidateRecord[] | null
  decision_rule?: string | null
  margin?: number | null
  exact_stage?: boolean | null
}


/** v2 decision kinds get a short human label + badge tone. Legacy rows
 * (no decision_kind) fall back to the matched/unknown status. */
const decisionKindLabel = (kind: string | null | undefined): string => {
  switch (kind) {
    case 'KB_MATCH':
      return 'KB match'
    case 'FAMILY_MATCH':
      return 'Family proposal'
    case 'UNKNOWN':
      return 'Open discovery'
    default:
      return kind ?? ''
  }
}

const decisionKindTone = (
  kind: string | null | undefined,
): 'neutral' | 'success' | 'warning' | 'danger' | 'accent' => {
  switch (kind) {
    case 'KB_MATCH':
      return 'success'
    case 'FAMILY_MATCH':
      return 'accent'
    default:
      return 'warning'
  }
}

const evidenceLabels: Record<string, string> = {
  embedding_similarity: 'Embedding similarity',
  name_similarity: 'Name similarity',
  description_similarity: 'Description similarity',
  datatype_compatibility: 'Datatype compatibility',
  profile_compatibility: 'Profile compatibility',
  context_similarity: 'Context similarity',
  value_evidence: 'Value vocabulary evidence',
  relationship_evidence: 'Approved relationship support',
}

/** Evidence that BELONGS to a later stage or to a narrower signal: hidden
 * from the semantic evidence list unless it actually fired (applicable).
 * relationship_evidence only exists when approved Stage-05 relationships
 * feed hints into semantic scoring — showing it as "no evidence available"
 * for every unreviewed column was misleading. */
const stageHiddenEvidence = new Set(['relationship_evidence', 'value_evidence'])

/** Human-readable display form for internally derived concept names:
 * open-discovery proposals come from expanded tokens joined with "_"
 * (e.g. "customer_type"); the UI presents them as "Customer Type". The
 * stored value is untouched — this is presentation only. */
const readableConceptName = (name: string | null | undefined): string =>
  (name ?? '')
    .replace(/[_\-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')

/** The lifecycle distinguishes the SYSTEM RECOMMENDATION from the USER
 * DECISION. Status labels follow the persisted decision lifecycle. */
const statusLabel = (status: string): string => {
  switch (status) {
    case 'approved':
      return 'Approved'
    case 'edited':
      return 'Edited'
    case 'rejected':
      return 'Rejected'
    default:
      return 'Recommended'
  }
}

const statusTone = (
  status: string,
): 'neutral' | 'success' | 'warning' | 'danger' | 'accent' => {
  switch (status) {
    case 'approved':
    case 'edited':
      return 'success'
    case 'rejected':
      return 'danger'
    default:
      return 'accent'
  }
}

/** Primary label shown for a column. For decided columns the FINAL semantic
 * type (user decision) is authoritative; the system recommendation moves to
 * supporting evidence. Rejected columns display "Not approved". */
const primaryType = (prediction: SemanticPrediction): string | null => {
  if (prediction.status === 'rejected') return null

  if (prediction.status === 'edited' && prediction.user_defined_type) {
    return prediction.user_defined_type
  }

  if (
    prediction.status === 'approved' &&
    (prediction.final_semantic_type ?? prediction.predicted_concept)
  ) {
    return prediction.final_semantic_type ?? prediction.predicted_concept
  }

  return prediction.predicted_concept
}

export function SemanticStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [columns, setColumns] = useState<SemanticPrediction[]>([])
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [decisionError, setDecisionError] = useState('')
  const [reviewIndex, setReviewIndex] = useState(0)
  const [showReview, setShowReview] = useState(false)
  // Edit state is per-prediction so navigating between columns cancels the
  // draft rather than applying it to the wrong column.
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editValue, setEditValue] = useState('')
  // v2: an Ask-User column starts the edit draft from the engine's PROPOSAL
  // (abbreviation-expanded, non-role name) instead of a blank input.
  const startEditWith = (prediction: SemanticPrediction, value: string) => {
    setEditingId(prediction.prediction_id)
    setEditValue(value)
  }

  // Clickable alternatives: inspecting an alternative shows ITS evidence and
  // offers Approve / Edit / Reject for that concept.
  const [alternativeView, setAlternativeView] = useState<string | null>(null)
  // S1-vs-S2 comparison: loaded from the append-only run history.
  const [runComparison, setRunComparison] = useState<RunComparison | null>(null)

  const loadPredictions = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const data = await apiFetch<{
        columns: SemanticPrediction[]
        column_count: number
      }>(`/datasets/${datasetId}/semantic`)

      setColumns(data.columns)

      // Keep the review cursor on the first column still pending.
      setReviewIndex((current) => {
        const pending = data.columns.findIndex((c) => c.status === 'pending')

        if (current >= data.columns.length) {
          return pending === -1 ? 0 : pending
        }

        return current
      })
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  useEffect(() => {
    void loadPredictions()
  }, [loadPredictions])

  useEffect(() => {
    let cancelled = false

    apiFetch<RunComparison>(
      `/datasets/${datasetId}/semantic/runs/compare`,
    )
      .then((data) => {
        if (!cancelled) setRunComparison(data)
      })
      .catch(() => {
        /* comparison is optional UI; absence is not an error */
      })

    return () => {
      cancelled = true
    }
  }, [datasetId, columns.length])

  const runAnalysis = async () => {
    setAnalyzing(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch<{ column_count: number }>(
        `/datasets/${datasetId}/semantic/analyze`,
        jsonInit('POST', {}),
      )

      setMessage(
        `Semantic analysis completed for ${data.column_count} columns. Approved and edited mappings were kept.`,
      )

      setReviewIndex(0)
      setShowReview(false)
      setEditingId(null)
      await loadPredictions()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setAnalyzing(false)
    }
  }

  const decide = async (
    predictionId: number,
    decision: 'approved' | 'rejected' | 'edited',
    conceptId: number | null,
    userDefinedType?: string,
  ) => {
    setDecisionError('')

    try {
      await apiFetch(
        `/datasets/${datasetId}/semantic/${predictionId}/decision`,
        jsonInit('POST', {
          decision,
          concept_id: conceptId,
          user_defined_type: userDefinedType ?? null,
        }),
      )

      await loadPredictions()
    } catch (err) {
      setDecisionError(apiErrorMessage(err))
    }
  }

  const startEdit = (prediction: SemanticPrediction) => {
    setEditingId(prediction.prediction_id)
    // Prefill with the current recommendation (or final type when already
    // decided) so the user edits rather than re-types.
    setEditValue(
      primaryType(prediction) ?? prediction.predicted_concept ?? '',
    )
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditValue('')
  }

  const saveEdit = async (prediction: SemanticPrediction) => {
    const value = editValue.trim()

    if (!value) {
      setDecisionError('Enter a semantic type to save.')
      return
    }

    const recommendation = prediction.predicted_concept ?? ''
    // Saving the unchanged recommendation is an approval, not an edit.
    const decision: 'approved' | 'edited' =
      recommendation && value === recommendation ? 'approved' : 'edited'

    setEditingId(null)
    setEditValue('')
    await decide(prediction.prediction_id, decision, null, value)
  }

  const pending = useMemo(
    () => columns.filter((c) => c.status === 'pending'),
    [columns],
  )
  const decidedCount = columns.filter(
    (c) => c.status === 'approved' || c.status === 'edited',
  ).length
  const rejectedCount = columns.filter((c) => c.status === 'rejected').length
  const pendingCount = pending.length

  const current = columns[reviewIndex]

  const advance = () => {
    setShowReview(false)
    setAlternativeView(null)

    setReviewIndex((current) => {
      const nextPending = columns.findIndex(
        (c, index) => index > current && c.status === 'pending',
      )

      if (nextPending !== -1) return nextPending

      const firstPending = columns.findIndex((c) => c.status === 'pending')

      return firstPending === -1 ? Math.min(current + 1, columns.length - 1) : firstPending
    })
  }

  /** Evidence details block shared by the review card and the list rows:
   * evidence scores, alternatives, and name normalization (original /
   * normalized / expanded names with abbreviation evidence). */
  const evidenceBlock = (prediction: SemanticPrediction) => (
    <>
      <p className="semantic-heading">Evidence</p>

      <ul className="evidence-list">
        {Object.entries(prediction.evidence)
          // Later-stage / narrower signals are hidden unless they actually
          // fired; they are never shown as failed semantic evidence.
          .filter(([key, rawItem]) => {
            const item = evidenceValue(rawItem)
            return !stageHiddenEvidence.has(key) || item.applicable
          })
          .map(([key, rawItem]) => {
            const item = evidenceValue(rawItem)
            const label = evidenceLabels[key] ?? key

            return (
              <li key={key} className="evidence-item">
                <span
                  className={`evidence-marker${item.applicable && item.value > 0 ? ' is-active' : ''}`}
                >
                  {item.applicable && item.value > 0 ? <IconCheck /> : '·'}
                </span>
                <span>
                  {label}
                  {!item.applicable &&
                    (key === 'description_similarity' ? (
                      <em className="cell-sub">
                        {' '}
                        (no column description was supplied to the engine)
                      </em>
                    ) : (
                      <em className="cell-sub"> (not applicable)</em>
                    ))}
                </span>
                <span className="evidence-value">
                  {item.applicable ? item.value.toFixed(2) : '—'}
                </span>
              </li>
            )
          })}
        {Object.keys(prediction.evidence).length === 0 && (
          <li className="evidence-item">
            <span>Insufficient evidence — open discovery</span>
          </li>
        )}
      </ul>

      {prediction.semantic_input && (
        <details className="semantic-audit-details" style={{ marginTop: 8 }}>
          <summary className="cell-sub">Semantic input (audit: exact embedding text)</summary>
          <div className="semantic-normalization" style={{ marginTop: 8 }}>
            <div className="normalization-grid">
              <span className="normalization-label">Description</span>
              <span className="normalization-value">
                {prediction.semantic_input.column_description ?? '— not supplied —'}
              </span>
              <span className="normalization-label">Domain</span>
              <span className="normalization-value">
                {prediction.semantic_input.dataset_domain ?? '—'}
              </span>
              <span className="normalization-label">Table context</span>
              <span className="normalization-value">
                {prediction.semantic_input.table_context ?? '—'}
              </span>
              <span className="normalization-label">Model</span>
              <span className="normalization-value">
                {prediction.semantic_input.embedding_model} / repr v
                {prediction.semantic_input.embedding_representation_version}
              </span>
            </div>
            <p className="semantic-heading" style={{ marginTop: 8 }}>
              Embedding text
            </p>
            <pre className="cell-sub" style={{ whiteSpace: 'pre-wrap' }}>
              {prediction.semantic_input.embedding_text ?? '—'}
            </pre>
          </div>
        </details>
      )}

      {prediction.normalization && (
        <details className="semantic-audit-details" style={{ marginTop: 8 }}>
          <summary className="cell-sub">Name normalization details (audit)</summary>
          <div className="semantic-normalization" style={{ marginTop: 8 }}>
          <p className="semantic-heading">Name normalization</p>
          <div className="normalization-grid">
            <span className="normalization-label">Original</span>
            <span className="normalization-value">
              {prediction.normalization.original_name ?? prediction.column_name}
            </span>

            <span className="normalization-label">Normalized</span>
            <span className="normalization-value">
              {prediction.normalization.normalized_name ?? '—'}
            </span>

            <span className="normalization-label">Expanded</span>
            <span className="normalization-value">
              {prediction.normalization.expanded_name ?? '—'}
            </span>
          </div>

          {(prediction.normalization.abbreviation_evidence?.length ?? 0) > 0 && (
            <ul className="evidence-list" style={{ marginTop: 8 }}>
              {prediction.normalization.abbreviation_evidence!.map((entry) => (
                <li key={entry.token} className="evidence-item">
                  <span
                    className={`evidence-marker${entry.expansion ? ' is-active' : ''}`}
                  >
                    {entry.expansion ? <IconCheck /> : '·'}
                  </span>
                  <span>
                    <code>{entry.token}</code>
                    {entry.expansion ? ` → ${entry.expansion}` : ' — no expansion'}
                    {entry.confidence === 'uncertain' && (
                      <em className="cell-sub"> (uncertain)</em>
                    )}
                    {(entry.alternatives?.length ?? 0) > 0 && (
                      <em className="cell-sub">
                        {' '}
                        (also: {entry.alternatives!.join(', ')})
                      </em>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
          </div>
        </details>
      )}

      {prediction.decision_kind === 'KB_MATCH' && prediction.exact_stage === true && (
        <p className="cell-sub">Decided by exact name match (highest precision path).</p>
      )}

      <p className="semantic-heading">Alternatives</p>

      {prediction.alternatives.length > 0 ? (
        <ul className="alternatives-list">
          {prediction.alternatives.map((alternative) => (
            <li
              key={alternative.concept}
              style={{ cursor: 'pointer' }}
              onClick={() =>
                setAlternativeView(
                  alternativeView === alternative.concept ? null : alternative.concept,
                )
              }
            >
              <span>
                {alternative.concept}
                <em className="cell-sub"> (click to inspect)</em>
              </span>
              <span className="evidence-value">
                {alternative.score.toFixed(2)}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="cell-sub">No alternative candidates above threshold.</p>
      )}

      {alternativeView && (
        <div className="semantic-proposal" style={{ marginTop: 8 }}>
          <p className="cell-sub">
            Alternative <strong>{alternativeView}</strong> — its evidence from the
            ranked audit trail:
          </p>
          {(() => {
            const candidate = prediction.candidates?.find(
              (entry) => entry.concept === alternativeView,
            )
            if (!candidate) {
              return <p className="cell-sub">No stored evidence for this alternative.</p>
            }
            return (
              <>
                <ul className="alternatives-list">
                  <li>
                    <span>Score</span>
                    <span className="evidence-value">
                      {candidate.confidence_score.toFixed(2)}
                    </span>
                  </li>
                  {candidate.gates?.entity_conflict && <li><span>Entity conflict</span></li>}
                  {candidate.gates?.head_conflict && <li><span>Head conflict</span></li>}
                  {candidate.gates?.profile_veto && <li><span>Profile veto</span></li>}
                </ul>
                <div className="notice-actions" style={{ marginTop: 6 }}>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => {
                      setAlternativeView(null)
                      void decide(prediction.prediction_id, 'edited', null, alternativeView)
                    }}
                  >
                    Use this concept
                  </button>
                </div>
              </>
            )
          })()}
        </div>
      )}
    </>
  )

  return (
    <main className="container page">
      <PageHeader
        title="Semantic understanding"
        description="Each column is mapped to a canonical concept — or openly discovered as unknown — from name, description, embedding, datatype and profile evidence. Evidence is shown, never hidden behind a single number. Approve, reject or edit each recommendation; your decision becomes the final semantic type."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="Semantic predictions"
          description="Review each column sequentially: approve, edit or reject. Decided columns keep your final semantic type when the analysis is re-run."
          action={
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={analyzing}
              onClick={() => void runAnalysis()}
            >
              {analyzing
                ? 'Analyzing…'
                : columns.length === 0
                  ? 'Run semantic analysis'
                  : 'Re-run analysis'}
            </button>
          }
        />

        {columns.length > 0 && (
          <SummaryGrid
            items={[
              ['Columns analyzed', columns.length],
              ['Pending review', pendingCount],
              ['Approved / edited', decidedCount],
              ['Rejected', rejectedCount],
            ]}
          />
        )}

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="danger" title={error} />}
          {decisionError && <Notice tone="danger" title={decisionError} />}
        </div>

        {runComparison?.comparison_available && (
          <details style={{ marginBottom: 12 }}>
            <summary className="cell-sub">
              Compare runs S{runComparison.old_run?.run_number} → S
              {runComparison.new_run?.run_number}{' '}
              (
              {
                runComparison.columns?.filter((entry) => entry.changed).length
              }{' '}
              changed predictions)
            </summary>
            <table style={{ width: '100%', marginTop: 8, borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th className="cell-sub" style={{ textAlign: 'left' }}>Column</th>
                  <th className="cell-sub" style={{ textAlign: 'left' }}>
                    S{runComparison.old_run?.run_number} prediction
                  </th>
                  <th className="cell-sub" style={{ textAlign: 'left' }}>
                    S{runComparison.new_run?.run_number} prediction
                  </th>
                </tr>
              </thead>
              <tbody>
                {runComparison.columns?.map((entry) => (
                  <tr key={`${entry.table_name}.${entry.column_name}`}>
                    <td>
                      {entry.column_name}
                      <span className="cell-sub"> · {entry.table_name}</span>
                    </td>
                    <td className={entry.changed ? '' : 'cell-sub'}>
                      {entry.old_prediction ?? '—'}
                    </td>
                    <td className={entry.changed ? '' : 'cell-sub'}>
                      {entry.new_prediction ?? '—'}
                      {entry.changed && (
                        <em className="cell-sub"> (changed)</em>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )}

        {loading && <p className="empty-state">Loading semantic predictions…</p>}

        {!loading && columns.length === 0 && !error && (
          <p className="empty-state">
            No semantic predictions yet. Run the analysis to map columns to
            concepts.
          </p>
        )}

        {!loading && columns.length > 0 && current && (
          <>
            {/* Sequential review navigation */}
            <div className="review-nav">
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                disabled={reviewIndex === 0}
                onClick={() => {
                  setShowReview(false)
                  cancelEdit()
                  setAlternativeView(null)
                  setReviewIndex((index) => Math.max(0, index - 1))
                }}
              >
                Previous
              </button>

              <span className="review-position cell-sub">
                Column {reviewIndex + 1} of {columns.length}
                {pendingCount > 0 ? ` · ${pendingCount} pending` : ' · all decided'}
              </span>

              <button
                type="button"
                className="btn btn-ghost btn-sm"
                disabled={reviewIndex === columns.length - 1}
                onClick={() => {
                  setShowReview(false)
                  cancelEdit()
                  setAlternativeView(null)
                  setReviewIndex((index) => Math.min(columns.length - 1, index + 1))
                }}
              >
                Next
              </button>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setShowReview((value) => !value)}
              >
                {showReview ? 'Hide all columns' : 'Review all columns'}
              </button>
            </div>

            {/* Current column under review */}
            <div className="semantic-item is-open">
              <div className="semantic-body">
                <div className="semantic-columns">
                  <div>
                    <p className="semantic-heading">
                      <strong>{current.column_name}</strong>
                      <span className="cell-sub"> · {current.table_name}</span>
                    </p>

                    <div className="semantic-badges" style={{ marginBottom: 12 }}>
                      <StatusBadge tone={statusTone(current.status)}>
                        {statusLabel(current.status)}
                      </StatusBadge>

                      {current.status === 'pending' && (
                        <>
                          {current.decision_kind && (
                            <StatusBadge tone={decisionKindTone(current.decision_kind)}>
                              {decisionKindLabel(current.decision_kind)}
                            </StatusBadge>
                          )}
                          <StatusBadge tone={confidenceTone(current.confidence_level)}>
                            {current.confidence_level} (
                            {current.confidence_score.toFixed(2)})
                          </StatusBadge>
                          <span className="badge">Evidence coverage {(current.evidence_coverage * 100).toFixed(0)}%</span>
                        </>
                      )}
                    </div>

                    {/* PRIMARY display: the final semantic type once decided;
                        the system recommendation only until then. */}
                    <div className="semantic-final" style={{ marginBottom: 16 }}>
                      <p className="semantic-heading">
                        Semantic type
                      </p>

                      {current.status === 'rejected' ? (
                        <p className="semantic-final-value semantic-final-rejected">
                          Not approved
                        </p>
                      ) : (
                        <p className="semantic-final-value">
                          {primaryType(current) ?? 'Unknown / Ambiguous'}
                        </p>
                      )}

                      {current.status === 'pending' && current.predicted_concept && (
                        <p className="cell-sub">
                          System recommendation · confidence{' '}
                          {current.confidence_score.toFixed(2)}
                        </p>
                      )}

                      {/* v2: semantic family chip — the GENERIC layer answer
                          is shown even when the specific concept is unknown. */}
                      {current.status === 'pending' && current.semantic_family && (
                        <p className="cell-sub">
                          Semantic family:{' '}
                          <span className="badge">{current.semantic_family.replace(/_/g, ' ')}</span>
                        </p>
                      )}

                      {/* v2: proposal review — FAMILY_MATCH / UNKNOWN columns
                          carry a new-concept proposal the user accepts (as a
                          user-defined type), edits, or rejects. */}
                      {current.status === 'pending' &&
                        current.proposal &&
                        (current.proposal.name || current.proposal.ask_user) && (
                          <div className="semantic-proposal" style={{ marginTop: 8 }}>
                            <p className="cell-sub">
                              {current.proposal.ask_user ? (
                                <>
                                  This column does not match any concept. Define a
                                  new semantic type (family:{' '}
                                  {current.proposal.family?.replace(/_/g, ' ') ?? 'unspecified'}).
                                </>
                              ) : (
                                <>
                                  Proposed new concept{' '}
                                  <strong>{readableConceptName(current.proposal.name)}</strong>
                                  {current.proposal.family
                                    ? ` (family: ${current.proposal.family.replace(/_/g, ' ')})`
                                    : ''}
                                </>
                              )}
                            </p>

                            <div className="notice-actions" style={{ marginTop: 6 }}>
                              {!current.proposal.ask_user && (
                                <button
                                  type="button"
                                  className="btn btn-primary btn-sm"
                                  onClick={() => {
                                    cancelEdit()
                                    void decide(
                                      current.prediction_id,
                                      'edited',
                                      null,
                                      readableConceptName(current.proposal!.name) || undefined,
                                    )
                                    advance()
                                  }}
                                >
                                  Accept proposal
                                </button>
                              )}

                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() =>
                                  startEditWith(
                                    current,
                                    readableConceptName(current.proposal!.name),
                                  )
                                }
                              >
                                {current.proposal.ask_user ? 'Define type' : 'Edit proposal'}
                              </button>

                              {/* Group action: every pending column sharing the
                                  proposal's group_key gets the same user-defined
                                  type (sequential API calls, no backend change). */}
                              {!current.proposal.ask_user && current.proposal.group_key && (
                                <button
                                  type="button"
                                  className="btn btn-ghost btn-sm"
                                  onClick={() => {
                                    const key = current.proposal!.group_key!
                                    const name = current.proposal!.name ?? ''
                                    const group = columns.filter(
                                      (c) =>
                                        c.status === 'pending' &&
                                        c.proposal?.group_key === key,
                                    )
                                    void (async () => {
                                      for (const member of group) {
                                        await decide(member.prediction_id, 'edited', null, name)
                                      }
                                    })()
                                    advance()
                                  }}
                                >
                                  Accept for all {columns.filter((c) => c.status === 'pending' && c.proposal?.group_key === current.proposal!.group_key).length} matching columns
                                </button>
                              )}
                            </div>
                          </div>
                        )}

                      {current.decision_source && (
                        <p className="cell-sub">
                          Decision source:{' '}
                          {current.decision_source === 'user_defined'
                            ? 'user (edited semantic type)'
                            : current.decision_source === 'kb_approval'
                              ? 'user (approved recommendation)'
                              : 'user (rejected)'}
                          {current.decided_at
                            ? ` · ${new Date(current.decided_at).toLocaleString()}`
                            : ''}
                        </p>
                      )}
                    </div>

                    {/* PENDING: Approve / Reject / Edit */}
                    {current.status === 'pending' && editingId !== current.prediction_id && (
                      <div className="semantic-actions" style={{ marginBottom: 16 }}>
                        <div className="notice-actions">
                          {/* Approval requires a KB concept: the backend
                              records kb_approval via concept_id. Open
                              discovery columns use Edit instead. */}
                          <button
                            type="button"
                            className="btn btn-primary btn-sm"
                            onClick={() =>
                              void decide(
                                current.prediction_id,
                                'approved',
                                current.concept_id,
                              )
                            }
                            disabled={current.concept_id === null}
                            title={
                              current.concept_id === null
                                ? 'No knowledge-base concept matched — use Edit to define the semantic type.'
                                : undefined
                            }
                          >
                            Approve
                          </button>

                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                              cancelEdit()
                              void decide(current.prediction_id, 'rejected', null)
                              advance()
                            }}
                          >
                            Reject
                          </button>

                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => startEdit(current)}
                          >
                            Edit
                          </button>

                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={advance}
                          >
                            Decide later
                          </button>
                        </div>
                      </div>
                    )}

                    {/* EDIT: inline text input prefilled with the current
                        recommendation; save may be approval (unchanged) or a
                        user-defined semantic type. */}
                    {editingId === current.prediction_id && (
                      <div className="semantic-edit" style={{ marginBottom: 16 }}>
                        <label className="field">
                          <span className="field-label">Edit semantic type</span>

                          <input
                            className="control control-sm"
                            value={editValue}
                            autoFocus
                            placeholder="Enter a semantic type (e.g. Customer Identifier)"
                            onChange={(event) => setEditValue(event.target.value)}
                            onKeyDown={(event) => {
                              if (event.key === 'Enter') {
                                event.preventDefault()
                                void saveEdit(current)
                              } else if (event.key === 'Escape') {
                                event.preventDefault()
                                cancelEdit()
                              }
                            }}
                          />
                        </label>

                        <p className="cell-sub" style={{ marginTop: 4 }}>
                          Enter any semantic type — new values are kept as
                          user-defined types and are not silently added to the
                          knowledge base.
                        </p>

                        <div className="notice-actions" style={{ marginTop: 8 }}>
                          <button
                            type="button"
                            className="btn btn-primary btn-sm"
                            disabled={!editValue.trim()}
                            onClick={() => void saveEdit(current)}
                          >
                            Save
                          </button>

                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={cancelEdit}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}

                    {/* DECIDED + not rejected: Edit remains available; a KB
                        concept picker supports re-deciding onto an existing
                        canonical concept. */}
                    {(current.status === 'approved' || current.status === 'edited') &&
                      editingId !== current.prediction_id && (
                        <div className="semantic-actions" style={{ marginBottom: 16 }}>
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => startEdit(current)}
                          >
                            Edit
                          </button>

                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                              cancelEdit()
                              void decide(current.prediction_id, 'rejected', null)
                              advance()
                            }}
                          >
                            Reject
                          </button>
                        </div>
                      )}

                    {evidenceBlock(current)}
                  </div>
                </div>
              </div>
            </div>

            {/* Full list review mode */}
            {showReview && (
              <div className="semantic-list">
                {columns.map((prediction, index) => (
                  <details
                    className="semantic-item"
                    key={prediction.prediction_id}
                    open={index === reviewIndex}
                  >
                    <summary
                      className="semantic-summary"
                      onClick={(event) => {
                        event.preventDefault()
                        setShowReview(false)
                        cancelEdit()
                        setReviewIndex(index)
                      }}
                    >
                      <span className="semantic-title">
                        <strong>{prediction.column_name}</strong>
                        <span>
                          {prediction.table_name} ·{' '}
                          {prediction.status === 'rejected'
                            ? 'Not approved'
                            : (primaryType(prediction) ?? 'Unknown')}
                        </span>
                      </span>

                      <span className="semantic-badges">
                        {prediction.status === 'pending' &&
                          prediction.decision_kind === 'FAMILY_MATCH' && (
                            <span className="badge badge-warning">
                              Family: {(prediction.semantic_family ?? '').replace(/_/g, ' ')}
                            </span>
                          )}

                        {prediction.status === 'pending' &&
                          prediction.predicted_concept === null && (
                            <span className="badge badge-warning">
                              Open discovery
                            </span>
                          )}

                        <StatusBadge tone={statusTone(prediction.status)}>
                          {statusLabel(prediction.status)}
                        </StatusBadge>

                        <IconChevron />
                      </span>
                    </summary>
                  </details>
                ))}
              </div>
            )}
          </>
        )}

        <div className="card-footer">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onStageComplete}
            disabled={columns.length === 0}
          >
            Continue to relationship discovery
          </button>
        </div>
      </section>
    </main>
  )
}
