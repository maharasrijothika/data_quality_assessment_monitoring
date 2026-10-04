import { useCallback, useEffect, useMemo, useState } from 'react'
import { apiErrorMessage, apiFetch, jsonInit } from '../api'
import {
  CardHeader,
  Notice,
  PageHeader,
  StatusBadge,
  SummaryGrid,
  IconCheck,
} from '../ui'

type RelationshipCandidate = {
  relationship_id: number
  candidate_kind: string | null
  parent_table: string
  parent_column: string
  child_table: string
  child_column: string
  parent_role: string
  score: number
  status: string
  human_note: string | null
  evidence: Record<string, unknown>
  containment: number | null
  orphan_distinct_count: number | null
  orphan_distinct_rate: number | null
  stats: Record<string, unknown>
}

type EvidenceItem = { value: number; applicable: boolean }

type OriginalRecommendation = {
  parent_table: string
  parent_column: string
  child_table: string
  child_column: string
}

const evidenceLabels: Record<string, string> = {
  name_similarity: 'Name similarity',
  semantic_compatibility: 'Semantic compatibility (final semantic types)',
  datatype_compatibility: 'Datatype compatibility',
  parent_uniqueness: 'Parent uniqueness (PK evidence)',
  value_containment: 'Value containment (child in parent)',
  value_overlap: 'Value overlap',
  cardinality_compatibility: 'Cardinality compatibility',
  child_completeness: 'Child completeness',
  structural_evidence: 'Structural evidence',
  composite_uniqueness: 'Composite uniqueness',
}

const statusLabel = (status: string): string => {
  switch (status) {
    case 'approved':
      return 'Accepted'
    case 'rejected':
      return 'Rejected'
    case 'edited':
      return 'Edited'
    case 'manual':
      return 'Manually defined'
    case 'missed':
      return 'Missed'
    default:
      return 'Candidate'
  }
}

const statusTone = (
  status: string,
): 'neutral' | 'success' | 'warning' | 'danger' | 'accent' => {
  switch (status) {
    case 'approved':
    case 'manual':
      return 'success'
    case 'edited':
      return 'accent'
    case 'rejected':
      return 'danger'
    case 'missed':
      return 'warning'
    default:
      return 'neutral'
  }
}

const isKeyCandidate = (candidate: RelationshipCandidate): boolean =>
  candidate.candidate_kind === 'pk' || candidate.candidate_kind === 'composite_pk'

const evidenceItem = (raw: unknown): EvidenceItem => {
  if (typeof raw === 'number') return { value: raw, applicable: true }

  const record = (raw ?? {}) as { value?: unknown; applicable?: unknown }

  return {
    value: Number(record.value ?? 0),
    applicable: Boolean(record.applicable),
  }
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' ? (value as Record<string, unknown>) : null

type RelationshipFields = {
  parent_table: string
  parent_column: string
  child_table: string
  child_column: string
}

const emptyFields = (): RelationshipFields => ({
  parent_table: '',
  parent_column: '',
  child_table: '',
  child_column: '',
})

export function RelationshipsStage({
  datasetId,
  onStageComplete,
}: {
  datasetId: number
  onStageComplete: () => void
}) {
  const [candidates, setCandidates] = useState<RelationshipCandidate[]>([])
  const [loading, setLoading] = useState(true)
  const [discovering, setDiscovering] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  // N manually-defined relationships (13.13): each entry is stored
  // separately; adding one never replaces previous ones.
  const [manualEntries, setManualEntries] = useState<RelationshipFields[]>([])
  const [showManual, setShowManual] = useState(false)

  // Manual PK entry (13.3): user-declared single or composite primary key.
  const [showKeyEntry, setShowKeyEntry] = useState(false)
  const [keyTable, setKeyTable] = useState('')
  const [keyColumns, setKeyColumns] = useState<string[]>([''])

  // Edit state (13.14): correction of a recommended relationship or of a
  // single-column PK candidate. editingKey marks which form is shown.
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editingKey, setEditingKey] = useState(false)
  const [editFields, setEditFields] = useState<RelationshipFields>(emptyFields())

  const loadCandidates = useCallback(async () => {
    setLoading(true)

    try {
      const data = await apiFetch<{ candidates: RelationshipCandidate[] }>(
        `/datasets/${datasetId}/relationships`,
      )

      setCandidates(data.candidates)
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  useEffect(() => {
    void loadCandidates()
  }, [loadCandidates])

  const runDiscovery = async () => {
    setDiscovering(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch<{
        candidate_count: number
        evaluated_pairs: number
      }>(
        `/datasets/${datasetId}/relationships/discover`,
        jsonInit('POST', {}),
      )

      setMessage(
        `Discovery evaluated ${data.evaluated_pairs} column pair(s); ${data.candidate_count} new candidate(s) found. Existing user decisions were kept.`,
      )

      await loadCandidates()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setDiscovering(false)
    }
  }

  const decide = async (
    relationshipId: number,
    decision: 'approved' | 'rejected' | 'missed',
  ) => {
    setBusy(true)
    setError('')

    try {
      await apiFetch(
        `/datasets/${datasetId}/relationships/${relationshipId}/decision`,
        jsonInit('POST', { decision }),
      )

      await loadCandidates()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditingKey(false)
    setEditFields(emptyFields())
  }

  const startEditRelationship = (candidate: RelationshipCandidate) => {
    setEditingId(candidate.relationship_id)
    setEditingKey(false)
    setEditFields({
      parent_table: candidate.parent_table,
      parent_column: candidate.parent_column,
      child_table: candidate.child_table,
      child_column: candidate.child_column,
    })
  }

  const startEditKey = (candidate: RelationshipCandidate) => {
    // Only single-column PK candidates are editable in place; composite
    // candidates should be rejected and re-declared via manual key entry.
    setEditingId(candidate.relationship_id)
    setEditingKey(true)
    setEditFields({
      parent_table: candidate.parent_table,
      parent_column: candidate.parent_column,
      child_table: '',
      child_column: '',
    })
  }

  const saveEdit = async () => {
    if (editingId === null) return

    setBusy(true)
    setError('')

    try {
      await apiFetch(
        `/datasets/${datasetId}/relationships/${editingId}/edit`,
        jsonInit('POST', {
          parent_table: editFields.parent_table,
          parent_column: editFields.parent_column,
          child_table: editingKey ? '' : editFields.child_table,
          child_column: editingKey ? '' : editFields.child_column,
        }),
      )

      setMessage(
        'Relationship corrected. The correction is now the user decision; the original recommendation stays in history.',
      )
      cancelEdit()
      await loadCandidates()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const openManualEntry = () => {
    setShowManual(true)

    if (manualEntries.length === 0) {
      setManualEntries([emptyFields()])
    }
  }

  const addManualRelationships = async () => {
    const valid = manualEntries.filter(
      (entry) =>
        entry.parent_table.trim() &&
        entry.parent_column.trim() &&
        entry.child_table.trim() &&
        entry.child_column.trim(),
    )

    if (valid.length === 0) {
      setError('Fill in all four fields for at least one relationship.')
      return
    }

    setBusy(true)
    setError('')

    try {
      for (const entry of valid) {
        await apiFetch(
          `/datasets/${datasetId}/relationships/manual`,
          jsonInit('POST', entry),
        )
      }

      setMessage(
        `${valid.length} manual relationship(s) added. Existing ones are never replaced.`,
      )
      setManualEntries([])
      setShowManual(false)
      await loadCandidates()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const addManualKey = async () => {
    const columns = keyColumns.map((column) => column.trim()).filter(Boolean)

    if (!keyTable.trim() || columns.length === 0) {
      setError('Enter a table name and at least one key column.')
      return
    }

    setBusy(true)
    setError('')

    try {
      await apiFetch(
        `/datasets/${datasetId}/relationships/manual-key`,
        jsonInit('POST', { table_name: keyTable, key_columns: columns }),
      )

      setMessage(
        columns.length === 1
          ? `Primary key candidate "${columns[0]}" recorded for ${keyTable}.`
          : `Composite key candidate ${columns.join(' + ')} recorded for ${keyTable}.`,
      )
      setKeyTable('')
      setKeyColumns([''])
      setShowKeyEntry(false)
      await loadCandidates()
    } catch (err) {
      setError(apiErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const relationships = useMemo(
    () => candidates.filter((candidate) => !isKeyCandidate(candidate)),
    [candidates],
  )
  const keyCandidates = useMemo(
    () => candidates.filter(isKeyCandidate),
    [candidates],
  )

  const pendingCount = candidates.filter((c) => c.status === 'pending').length
  const acceptedCount = candidates.filter(
    (c) =>
      c.status === 'approved' || c.status === 'manual' || c.status === 'edited',
  ).length

  const candidateTitle = (candidate: RelationshipCandidate) => {
    if (isKeyCandidate(candidate)) {
      return `${candidate.parent_table}.${candidate.parent_column}`
    }

    const child = candidate.child_table
      ? `${candidate.child_table}.${candidate.child_column}`
      : '—'

    return `${candidate.parent_table}.${candidate.parent_column} → ${child}`
  }

  /** One candidate card: evidence, edit form or decision buttons, and the
   * original recommendation when the user has corrected it. */
  const candidateCard = (candidate: RelationshipCandidate) => {
    const lowContainment =
      candidate.containment !== null &&
      candidate.containment < 0.9 &&
      (candidate.orphan_distinct_count ?? 0) > 0 &&
      candidate.status !== 'rejected'

    // After a user correction the original recommendation stays available
    // in the evidence payload (audit history).
    const originalRecommendation = asRecord(
      asRecord(candidate.evidence)?.original_recommendation,
    ) as OriginalRecommendation | null

    const isEditing = editingId === candidate.relationship_id

    return (
      <details
        className="semantic-item"
        key={candidate.relationship_id}
        open={candidate.status === 'pending'}
      >
        <summary className="semantic-summary">
          <span className="semantic-title">
            <strong>{candidateTitle(candidate)}</strong>
            <span>
              {isKeyCandidate(candidate)
                ? 'Key candidate'
                : `Evidence score ${candidate.score.toFixed(2)}${
                    candidate.containment !== null
                      ? ` · containment ${(candidate.containment * 100).toFixed(0)}%`
                      : ''
                  }`}
            </span>
          </span>

          <span className="semantic-badges">
            <span className="badge">
              {candidate.candidate_kind === 'pk'
                ? 'PK candidate'
                : candidate.candidate_kind === 'composite_pk'
                  ? 'Composite PK candidate'
                  : 'FK candidate'}
            </span>
            <StatusBadge tone={statusTone(candidate.status)}>
              {statusLabel(candidate.status)}
            </StatusBadge>
          </span>
        </summary>

        <div className="semantic-body">
          {lowContainment && (
            <div className="card-messages">
              <Notice
                tone="warning"
                title="Low containment with strong evidence — likely orphan records"
              >
                <p className="notice-text">
                  {candidate.orphan_distinct_count} distinct child value(s) (
                  {((candidate.orphan_distinct_rate ?? 0) * 100).toFixed(0)}
                  %) do not exist in the parent. Containment is evidence about
                  the data, not proof about whether the relationship exists —
                  it may be real with data-quality problems. Approval here
                  enables an RI check later; it never validates the data by
                  itself.
                </p>
              </Notice>
            </div>
          )}

          {originalRecommendation && (
            <div className="card-messages">
              <Notice
                tone="info"
                title="Corrected by user — original recommendation kept in history"
              >
                <p className="notice-text">
                  {originalRecommendation.parent_table}.
                  {originalRecommendation.parent_column}
                  {' → '}
                  {originalRecommendation.child_table || '—'}
                  {originalRecommendation.child_column
                    ? `.${originalRecommendation.child_column}`
                    : ''}
                </p>
              </Notice>
            </div>
          )}

          {isEditing ? (
            <div className="section">
              <p className="section-hint" style={{ marginBottom: 8 }}>
                Correct this {editingKey ? 'primary key candidate' : 'relationship'}.
                The corrected values become the user decision; the original
                recommendation remains in history.
              </p>

              <div className="manual-entry-fields">
                <label className="field">
                  <span className="field-label">Parent table</span>

                  <input
                    className="control control-sm"
                    value={editFields.parent_table}
                    onChange={(event) =>
                      setEditFields((current) => ({
                        ...current,
                        parent_table: event.target.value,
                      }))
                    }
                  />
                </label>

                <label className="field">
                  <span className="field-label">
                    {editingKey ? 'Key column' : 'Parent column (PK)'}
                  </span>

                  <input
                    className="control control-sm"
                    value={editFields.parent_column}
                    onChange={(event) =>
                      setEditFields((current) => ({
                        ...current,
                        parent_column: event.target.value,
                      }))
                    }
                  />
                </label>

                {!editingKey && (
                  <>
                    <label className="field">
                      <span className="field-label">Child table</span>

                      <input
                        className="control control-sm"
                        value={editFields.child_table}
                        onChange={(event) =>
                          setEditFields((current) => ({
                            ...current,
                            child_table: event.target.value,
                          }))
                        }
                      />
                    </label>

                    <label className="field">
                      <span className="field-label">Child column (FK)</span>

                      <input
                        className="control control-sm"
                        value={editFields.child_column}
                        onChange={(event) =>
                          setEditFields((current) => ({
                            ...current,
                            child_column: event.target.value,
                          }))
                        }
                      />
                    </label>
                  </>
                )}
              </div>

              <div className="notice-actions" style={{ marginTop: 12 }}>
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  disabled={
                    busy ||
                    !editFields.parent_table.trim() ||
                    !editFields.parent_column.trim() ||
                    (!editingKey &&
                      (!editFields.child_table.trim() ||
                        !editFields.child_column.trim()))
                  }
                  onClick={() => void saveEdit()}
                >
                  Save correction
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
          ) : (
            <div className="semantic-columns">
              <div>
                <p className="semantic-heading">Evidence</p>

                <ul className="evidence-list">
                  {Object.entries(candidate.evidence ?? {})
                    .filter(
                      ([key]) =>
                        key !== 'original_recommendation' &&
                        key !== 'source' &&
                        key !== 'note',
                    )
                    .map(([key, raw]) => {
                      const item = evidenceItem(raw)

                      return (
                        <li key={key} className="evidence-item">
                          <span
                            className={`evidence-marker${item.applicable && item.value > 0 ? ' is-active' : ''}`}
                          >
                            {item.applicable && item.value > 0 ? (
                              <IconCheck />
                            ) : (
                              '·'
                            )}
                          </span>
                          <span>
                            {evidenceLabels[key] ?? key}
                            {!item.applicable && (
                              <em className="cell-sub"> (not applicable)</em>
                            )}
                          </span>
                          <span className="evidence-value">
                            {item.applicable ? item.value.toFixed(2) : '—'}
                          </span>
                        </li>
                      )
                    })}

                  {candidate.human_note && (
                    <li className="evidence-item">
                      <span>Note: {candidate.human_note}</span>
                    </li>
                  )}
                </ul>
              </div>

              <div>
                <p className="semantic-heading">Decision</p>

                {candidate.status === 'pending' ? (
                  <div className="semantic-actions">
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      disabled={busy}
                      onClick={() =>
                        void decide(candidate.relationship_id, 'approved')
                      }
                    >
                      Accept
                    </button>

                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      disabled={busy}
                      onClick={() =>
                        void decide(candidate.relationship_id, 'rejected')
                      }
                    >
                      Reject
                    </button>

                    {/* Composite key candidates are corrected by rejecting
                        and re-declaring the right key via manual entry. */}
                    {candidate.candidate_kind === 'composite_pk' ? (
                      <span className="cell-sub">
                        To correct: reject, then use “Add key manually”.
                      </span>
                    ) : (
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        disabled={busy}
                        onClick={() =>
                          isKeyCandidate(candidate)
                            ? startEditKey(candidate)
                            : startEditRelationship(candidate)
                        }
                      >
                        Edit
                      </button>
                    )}
                  </div>
                ) : (
                  <p className="cell-sub">
                    Decision recorded:{' '}
                    <strong>{statusLabel(candidate.status)}</strong>
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </details>
    )
  }

  return (
    <main className="container page">
      <PageHeader
        title="Relationship discovery"
        description="Primary-key and foreign-key candidates are discovered from deterministic evidence — final semantic types, name, datatype, uniqueness, containment, overlap and cardinality. Candidates are ranked, never decided, by the system: accept, reject or correct each one."
        onBack={() => onStageComplete()}
      />

      <section className="card">
        <CardHeader
          title="Candidate relationships"
          description="Only human-accepted relationships may later become authoritative RI rules. Low containment does not eliminate a candidate: it usually means the child table contains orphan records."
          action={
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={discovering}
              onClick={() => void runDiscovery()}
            >
              {discovering
                ? 'Discovering…'
                : candidates.length === 0
                  ? 'Run relationship discovery'
                  : 'Re-run discovery'}
            </button>
          }
        />

        {candidates.length > 0 && (
          <SummaryGrid
            items={[
              ['Candidates', candidates.length],
              ['Pending review', pendingCount],
              ['Accepted', acceptedCount],
              [
                'Rejected / missed',
                candidates.filter(
                  (c) => c.status === 'rejected' || c.status === 'missed',
                ).length,
              ],
            ]}
          />
        )}

        <div className="card-messages">
          {message && <Notice tone="success" title={message} />}
          {error && <Notice tone="danger" title={error} />}
        </div>

        {/* ---------------------------------------------------------------
         * Key candidates (13.1-13.4): single- AND multi-table datasets.
         * --------------------------------------------------------------- */}
        <div className="section">
          <div
            className="section-head"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }}
          >
            <div>
              <p className="section-hint" style={{ margin: 0, fontWeight: 600 }}>
                Key candidates
              </p>
              <p className="section-hint" style={{ margin: 0 }}>
                Primary-key and composite-key candidates per table — evidence,
                not confirmed constraints. Manually declared keys are supported
                for single-file datasets.
              </p>
            </div>

            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setShowKeyEntry((value) => !value)}
            >
              {showKeyEntry ? 'Cancel key entry' : 'Add key manually'}
            </button>
          </div>

          {showKeyEntry && (
            <div className="semantic-edit" style={{ marginTop: 12 }}>
              <div className="key-entry-fields">
                <label className="field">
                  <span className="field-label">Table</span>

                  <input
                    className="control control-sm"
                    value={keyTable}
                    placeholder="e.g. customers"
                    onChange={(event) => setKeyTable(event.target.value)}
                  />
                </label>

                <div className="field">
                  <span className="field-label">
                    Key columns (add more for a composite key)
                  </span>

                  {keyColumns.map((column, index) => (
                    <div
                      key={index}
                      style={{
                        display: 'flex',
                        gap: 8,
                        marginTop: index === 0 ? 0 : 8,
                      }}
                    >
                      <input
                        className="control control-sm"
                        value={column}
                        placeholder={
                          index === 0 ? 'e.g. customer_id' : 'another column…'
                        }
                        onChange={(event) =>
                          setKeyColumns((current) =>
                            current.map((value, i) =>
                              i === index ? event.target.value : value,
                            ),
                          )
                        }
                      />

                      {keyColumns.length > 1 && (
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm"
                          onClick={() =>
                            setKeyColumns((current) =>
                              current.filter((_, i) => i !== index),
                            )
                          }
                        >
                          Remove
                        </button>
                      )}
                    </div>
                  ))}

                  <div style={{ marginTop: 8 }}>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => setKeyColumns((current) => [...current, ''])}
                    >
                      + Add key column
                    </button>
                  </div>
                </div>
              </div>

              <div className="notice-actions" style={{ marginTop: 12 }}>
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  disabled={busy || !keyTable.trim()}
                  onClick={() => void addManualKey()}
                >
                  Save key candidate
                </button>
              </div>
            </div>
          )}

          {!loading && keyCandidates.length > 0 && (
            <div className="semantic-list" style={{ marginTop: 12 }}>
              {keyCandidates.map((candidate) => candidateCard(candidate))}
            </div>
          )}

          {!loading && keyCandidates.length === 0 && !showKeyEntry && (
            <p className="empty-state">
              No key candidates yet. Run discovery — or declare a primary key
              manually above.
            </p>
          )}
        </div>

        {/* ---------------------------------------------------------------
         * Cross-table relationship candidates (13.5, 13.13-13.16).
         * --------------------------------------------------------------- */}
        <div className="section">
          <div
            className="section-head"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }}
          >
            <div>
              <p className="section-hint" style={{ margin: 0, fontWeight: 600 }}>
                Relationship candidates
              </p>
              <p className="section-hint" style={{ margin: 0 }}>
                PK → FK and other cross-table candidates. You can define any
                number of relationships manually.
              </p>
            </div>

            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() =>
                showManual ? setShowManual(false) : openManualEntry()
              }
            >
              {showManual ? 'Cancel manual add' : 'Add relationship'}
            </button>
          </div>

          {showManual && (
            <div className="semantic-edit" style={{ marginTop: 12 }}>
              <p className="section-hint" style={{ marginBottom: 8 }}>
                Each relationship is stored separately. Add as many as you
                need; nothing replaces existing entries.
              </p>

              <div className="manual-entries">
                {manualEntries.map((entry, index) => (
                  <div className="manual-entry" key={index}>
                    <div className="manual-entry-fields">
                      <label className="field">
                        <span className="field-label">Parent table</span>

                        <input
                          className="control control-sm"
                          value={entry.parent_table}
                          placeholder="e.g. customers"
                          onChange={(event) =>
                            setManualEntries((current) =>
                              current.map((value, i) =>
                                i === index
                                  ? {
                                      ...value,
                                      parent_table: event.target.value,
                                    }
                                  : value,
                              ),
                            )
                          }
                        />
                      </label>

                      <label className="field">
                        <span className="field-label">Parent column (PK)</span>

                        <input
                          className="control control-sm"
                          value={entry.parent_column}
                          placeholder="e.g. customer_id"
                          onChange={(event) =>
                            setManualEntries((current) =>
                              current.map((value, i) =>
                                i === index
                                  ? {
                                      ...value,
                                      parent_column: event.target.value,
                                    }
                                  : value,
                              ),
                            )
                          }
                        />
                      </label>

                      <label className="field">
                        <span className="field-label">Child table</span>

                        <input
                          className="control control-sm"
                          value={entry.child_table}
                          placeholder="e.g. orders"
                          onChange={(event) =>
                            setManualEntries((current) =>
                              current.map((value, i) =>
                                i === index
                                  ? { ...value, child_table: event.target.value }
                                  : value,
                              ),
                            )
                          }
                        />
                      </label>

                      <label className="field">
                        <span className="field-label">Child column (FK)</span>

                        <input
                          className="control control-sm"
                          value={entry.child_column}
                          placeholder="e.g. customer_id"
                          onChange={(event) =>
                            setManualEntries((current) =>
                              current.map((value, i) =>
                                i === index
                                  ? {
                                      ...value,
                                      child_column: event.target.value,
                                    }
                                  : value,
                              ),
                            )
                          }
                        />
                      </label>
                    </div>

                    <button
                      type="button"
                      className="btn btn-ghost btn-sm manual-entry-remove"
                      onClick={() =>
                        setManualEntries((current) =>
                          current.filter((_, i) => i !== index),
                        )
                      }
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>

              <div className="notice-actions" style={{ marginTop: 12 }}>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() =>
                    setManualEntries((current) => [...current, emptyFields()])
                  }
                >
                  + Add another relationship
                </button>

                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  disabled={busy || manualEntries.length === 0}
                  onClick={() => void addManualRelationships()}
                >
                  Save{' '}
                  {manualEntries.length > 0 ? `${manualEntries.length} ` : ''}
                  relationship
                  {manualEntries.length === 1 ? '' : 's'}
                </button>
              </div>
            </div>
          )}

          {!loading && relationships.length > 0 && (
            <div className="semantic-list" style={{ marginTop: 12 }}>
              {relationships.map((candidate) => candidateCard(candidate))}
            </div>
          )}

          {!loading && relationships.length === 0 && (
            <p className="empty-state">
              No relationship candidates yet. Cross-table relationships need at
              least two tables — single files only produce key candidates.
            </p>
          )}
        </div>

        <div className="card-footer">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onStageComplete}
          >
            Continue to metric &amp; rule recommendation
          </button>
        </div>
      </section>
    </main>
  )
}
