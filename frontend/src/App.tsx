import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'
import { apiErrorMessage, apiFetch } from './api'
import {
  CardHeader,
  IconCheck,
  IconChevron,
  IconFolder,
  IconUpload,
  Notice,
  PageHeader,
  SummaryGrid,
} from './ui'
import { SemanticStage } from './stages/semantic'
import { RelationshipsStage } from './stages/relationships'
import { RulesStage, ExecutionStage } from './stages/rules'
import { ScoringStage, RemediationStage } from './stages/scoring'
import { MonitoringStage, FeedbackStage } from './stages/monitoring'

const API_BASE_URL = 'http://127.0.0.1:8000'

const stages = [
  '01 Ingestion',
  '02 Context',
  '03 Version',
  '04 Profiling',
  '05 Semantic',
  '06 Relationships',
  '07 Rules',
  '08 Execution',
  '09 Scoring & RCA',
  '10 Remediation',
  '11 Monitoring',
  '12 Feedback',
]

// Stage navigation covers the full pipeline; availability depends on an
// opened dataset.

const SOURCE_SYSTEMS = [
  'SAP',
  'Salesforce',
  'Oracle',
  'Microsoft Dynamics',
  'ServiceNow',
  'Workday',
  'HubSpot',
  'Zoho CRM',
  'Custom / Internal System',
  'Other',
]

const UPDATE_CADENCES = [
  'Real-time',
  'Hourly',
  'Daily',
  'Weekly',
  'Monthly',
  'Quarterly',
  'Ad hoc',
]

type UploadResult = {
  message: string
  dataset_id: number
  dataset_name: string
  version_id: number
  version_number: number
  schema_fingerprint: string
  content_fingerprint: string
  files: Array<{
    filename: string
    stored_filename: string
    content_fingerprint: string
    tables: string[]
  }>
}

type UploadAnalysis = {
  status: 'all_new' | 'all_duplicates' | 'review_required'
  message: string
  total_files: number
  duplicate_count: number
  new_count: number
  duplicate_files: Array<{
    filename: string
    content_fingerprint: string
    existing_files: Array<{
      file_id: number
      original_filename: string
      stored_filename: string
      version_id: number
    }>
  }>
  new_files: Array<{
    filename: string
    content_fingerprint: string
  }>
}

type ContextData = {
  dataset_id: number
  dataset_name: string
  description: string | null
  domain: string | null
  source_system: string | null
  update_cadence: string | null
  current_version: {
    version_id: number
    version_number: number
    parent_version_id: number | null
    schema_fingerprint: string | null
    content_fingerprint: string | null
    files: Array<{
      file_id: number
      filename: string
      stored_filename: string
      content_fingerprint: string
    }>
  }
  tables: Array<{
    table_id: number
    table_name: string
    source_file: string
    sheet_name: string | null
    row_count: number | null
    description: string | null
    columns: Array<{
      column_id: number
      column_name: string
      data_type: string
      description: string | null
    }>
  }>
}

// Row shown in the "Registered datasets" list. Only id + name are required;
// everything else is optional so the list works with whatever GET /datasets returns.
type DatasetSummary = {
  dataset_id: number
  dataset_name: string
  version_number: number | null
  source_system: string | null
  domain: string | null
  file_count: number | null
  table_count: number | null
  first_incomplete_stage?: string | null
  completed_count?: number
  total_stages?: number
}

type StageProgress = {
  stages: Array<{
    stage_key: string
    label: string
    completed: boolean
  }>
  first_incomplete_stage: string | null
  completed_count: number
  total_stages: number
}

type BrowserFile = File & {
  webkitRelativePath?: string
}

const shortenFingerprint = (fingerprint: string | null): string => {
  if (!fingerprint) return 'Not available'
  if (fingerprint.length <= 24) return fingerprint

  return `${fingerprint.slice(0, 16)}...${fingerprint.slice(-8)}`
}

const toNumberOrNull = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null

const toTextOrNull = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value : null

const normalizeDatasets = (payload: unknown): DatasetSummary[] => {
  const container = payload as { datasets?: unknown } | null
  const list: unknown[] = Array.isArray(payload)
    ? payload
    : Array.isArray(container?.datasets)
      ? (container.datasets as unknown[])
      : []

  return list.flatMap((raw): DatasetSummary[] => {
    const item = (raw ?? {}) as Record<string, unknown>
    const version = (item.current_version ??
      item.latest_version ??
      {}) as Record<string, unknown>

    const id = toNumberOrNull(item.dataset_id ?? item.id)
    const name = toTextOrNull(item.dataset_name ?? item.name)

    if (id === null || name === null) return []

    return [
      {
        dataset_id: id,
        dataset_name: name,
        version_number: toNumberOrNull(
          version.version_number ??
            item.version_number ??
            item.current_version_number ??
            item.latest_version_number,
        ),
        source_system: toTextOrNull(item.source_system),
        domain: toTextOrNull(item.domain),
        file_count: toNumberOrNull(item.file_count ?? item.files_count),
        table_count: toNumberOrNull(item.table_count ?? item.tables_count),
        first_incomplete_stage:
          typeof item.first_incomplete_stage === 'string'
            ? item.first_incomplete_stage
            : null,
        completed_count:
          typeof item.completed_count === 'number' ? item.completed_count : 0,
        total_stages:
          typeof item.total_stages === 'number' ? item.total_stages : 0,
      },
    ]
  })
}

const describeDataset = (dataset: DatasetSummary): string =>
  [
    dataset.version_number !== null ? `V${dataset.version_number}` : null,
    dataset.file_count !== null
      ? `${dataset.file_count} file${dataset.file_count === 1 ? '' : 's'}`
      : null,
    dataset.source_system,
    dataset.domain,
  ]
    .filter(Boolean)
    .join(' · ') || `Dataset ${dataset.dataset_id}`

const formatNumber = (value: unknown): string =>
  typeof value === 'number' ? value.toLocaleString() : '—'

const formatPercent = (value: unknown): string =>
  typeof value === 'number' ? `${value.toFixed(2)}%` : '—'

const formatStat = (value: unknown): string => {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') {
    return Number.isInteger(value)
      ? String(value)
      : String(parseFloat(value.toFixed(4)))
  }

  return String(value)
}

// Renders a count that tolerates old profiles where the key is missing.
const formatCount = (value: unknown): string =>
  typeof value === 'number' ? value.toLocaleString() : String(value ?? '—')

// First value that is actually present (new-profile key wins, legacy key
// used as fallback); undefined only when neither profile generation has it.
const firstDefined = (...values: unknown[]): unknown =>
  values.find((value) => value !== undefined)

function App() {
  const [currentStage, setCurrentStage] = useState(0)
  const [profilingData, setProfilingData] = useState<any>(null)
  const [profilingDatasetId, setProfilingDatasetId] = useState<number | null>(
    null,
  )
  const [loadingProfiling, setLoadingProfiling] = useState(false)
  const [profilingError, setProfilingError] = useState('')
  const [expandedProfileTables, setExpandedProfileTables] = useState<
    Set<string>
  >(new Set())
  const [datasetName, setDatasetName] = useState('')
  const [sourceSystem, setSourceSystem] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])

  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const [uploadAnalysis, setUploadAnalysis] = useState<UploadAnalysis | null>(
    null,
  )
  const [error, setError] = useState('')

  const [contextData, setContextData] = useState<ContextData | null>(null)
  const [loadingContext, setLoadingContext] = useState(false)
  const [savingContext, setSavingContext] = useState(false)
  const [contextMessage, setContextMessage] = useState('')

  // Registered datasets + backend-persisted stage progress.
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [datasetsError, setDatasetsError] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')
  const [datasetToDelete, setDatasetToDelete] = useState<DatasetSummary | null>(null)
  const [deletingDataset, setDeletingDataset] = useState(false)
  const [stageProgress, setStageProgress] = useState<StageProgress | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  const getUploadFilename = (file: File): string => {
    const browserFile = file as BrowserFile

    return browserFile.webkitRelativePath || file.name
  }

  const handleFilesSelected = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])

    if (files.length === 0) {
      return
    }

    setSelectedFiles(files)
    setUploadResult(null)
    setUploadAnalysis(null)
    setError('')

    // Automatically suggest a dataset name from the uploaded file/folder.
    if (!datasetName.trim()) {
      const firstFile = files[0] as BrowserFile

      const relativePath = firstFile.webkitRelativePath ?? ''

      const sourceName = relativePath
        ? relativePath.split('/')[0]
        : firstFile.name

      const suggestedName = sourceName.replace(/\.[^/.]+$/, '')

      setDatasetName(suggestedName)
    }

    // Allow the same file/folder to be selected again.
    event.target.value = ''
  }

  const loadDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    setDatasetsError('')

    try {
      const data = await apiFetch<{
        datasets: Array<Record<string, unknown>>
      }>('/datasets')

      setDatasets(normalizeDatasets(data))
    } catch (err) {
      console.error('Dataset list request failed:', err)

      setDatasetsError(apiErrorMessage(err))
    } finally {
      setLoadingDatasets(false)
    }
  }, [])

  const requestDeleteDataset = (datasetId: number) => {
    const dataset = datasets.find(
      (item) => item.dataset_id === datasetId,
    )

    if (dataset) {
      setDatasetToDelete(dataset)
    }
  }

  const deleteDataset = async () => {
    if (!datasetToDelete) return

    const datasetId = datasetToDelete.dataset_id
    setDeletingDataset(true)
    setDatasetsError('')

    try {
      await apiFetch(`/datasets/${datasetId}`, {
        method: 'DELETE',
      })

      if (contextData?.dataset_id === datasetId) {
        setContextData(null)
      }

      setDatasetToDelete(null)
      await loadDatasets()
    } catch (err) {
      console.error('Dataset deletion failed:', err)
      setDatasetsError(apiErrorMessage(err))
    } finally {
      setDeletingDataset(false)
    }
  }


  const loadStageProgress = useCallback(async (datasetId: number) => {
    try {
      const data = await apiFetch<StageProgress>(
        `/datasets/${datasetId}/stages`,
      )

      setStageProgress(data)
    } catch {
      setStageProgress(null)
    }
  }, [])

  // Refresh the registered-dataset list whenever the Ingestion stage is shown.
  useEffect(() => {
    if (currentStage === 0) {
      void loadDatasets()
    }
  }, [currentStage, loadDatasets])

  const loadContext = async (datasetId: number) => {
    setLoadingContext(true)
    setError('')
    setContextMessage('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/datasets/${datasetId}/context`,
      )

      const data = await response.json()

      if (!response.ok) {
        setError(
          typeof data.detail === 'string'
            ? data.detail
            : 'Could not load dataset context.',
        )
        return
      }

      setContextData(data)
    } catch (err) {
      console.error('Context request failed:', err)

      setError(
        err instanceof Error ? err.message : 'Could not connect to the backend.',
      )
    } finally {
      setLoadingContext(false)
    }
  }

  const loadProfiling = async (datasetId: number) => {
    setLoadingProfiling(true)
    setProfilingError('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/datasets/${datasetId}/profiling`,
      )

      if (!response.ok) {
        const errorData = await response.json().catch(() => null)

        throw new Error(
          errorData?.detail ||
            `Profiling request failed with status ${response.status}`,
        )
      }

      const data = await response.json()

      setProfilingData(data)
      setProfilingDatasetId(datasetId)
      void loadStageProgress(datasetId)
    } catch (err) {
      console.error('Profiling load failed:', err)

      setProfilingError(
        err instanceof Error
          ? err.message
          : 'Unable to load profiling results.',
      )
    } finally {
      setLoadingProfiling(false)
    }
  }

  const openProfilingStage = async (datasetId: number) => {
    await loadProfiling(datasetId)
    setCurrentStage(3)
  }

  // Open a previously registered dataset at Stage 02. All later stages are
  // then reachable from the stage navigation.
  const openDataset = async (datasetId: number) => {
    setUploadResult(null)
    setUploadAnalysis(null)
    setError('')
    setContextMessage('')
    setContextData(null)
    setProfilingData(null)
    setProfilingDatasetId(null)
    setProfilingError('')
    setExpandedProfileTables(new Set())
    setCurrentStage(1)

    void loadStageProgress(datasetId)
    await loadContext(datasetId)
  }

  const goToStage = (index: number) => {
    setError('')
    setContextMessage('')

    if (index === 0) {
      setCurrentStage(0)
      return
    }

    if (!contextData) {
      return
    }

    if (
      index === 3 &&
      !(profilingData && profilingDatasetId === contextData.dataset_id)
    ) {
      void openProfilingStage(contextData.dataset_id)
      return
    }

    setCurrentStage(index)
  }

  const toggleProfileTable = (tableKey: string) => {
    setExpandedProfileTables((current) => {
      const next = new Set(current)

      if (next.has(tableKey)) {
        next.delete(tableKey)
      } else {
        next.add(tableKey)
      }

      return next
    })
  }

  const registerFiles = async (filesToRegister: File[]) => {
    if (!datasetName.trim()) {
      setError('Please enter a dataset name.')
      return
    }

    if (filesToRegister.length === 0) {
      setError('There are no new files available to register.')
      return
    }

    setUploading(true)
    setError('')
    setUploadResult(null)

    try {
      const formData = new FormData()

      filesToRegister.forEach((file) => {
        const filename = getUploadFilename(file)

        // Explicitly preserve browser folder-relative paths.
        formData.append('files', file, filename)
      })

      formData.append('dataset_name', datasetName.trim())

      if (sourceSystem.trim()) {
        formData.append('source_system', sourceSystem.trim())
      }

      const response = await fetch(`${API_BASE_URL}/datasets/upload`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        if (response.status === 409) {
          const duplicateFiles = data.detail?.duplicate_files

          if (Array.isArray(duplicateFiles) && duplicateFiles.length > 0) {
            const duplicateDetails = duplicateFiles
              .map((item: any) => {
                const existing = item.existing_files?.[0]

                if (!existing) {
                  return `${item.filename} already exists in the registry.`
                }

                return `${item.filename} already exists as ${existing.original_filename} (Dataset Version ${existing.version_id}).`
              })
              .join(' ')

            setError(
              `${data.detail?.message ?? 'Duplicate file detected.'} ${duplicateDetails}`,
            )
          } else {
            setError(
              data.detail?.message ??
                'One or more uploaded files already exist in the registered file history.',
            )
          }
        } else {
          setError(
            typeof data.detail === 'string'
              ? data.detail
              : 'Dataset upload failed.',
          )
        }

        return
      }

      setUploadAnalysis(null)
      setUploadResult(data)

      // Successful Stage 01 registration automatically opens Stage 02.
      setCurrentStage(1)

      // Load context for the newly registered dataset.
      await loadContext(data.dataset_id)
    } catch (err) {
      console.error('Upload request failed:', err)

      setError(
        err instanceof Error
          ? err.message
          : 'Could not connect to the backend. Make sure the FastAPI server is running on port 8000.',
      )
    } finally {
      setUploading(false)
    }
  }

  const analyzeUpload = async () => {
    if (!datasetName.trim()) {
      setError('Please enter a dataset name.')
      return
    }

    if (selectedFiles.length === 0) {
      setError('Please select a data file or folder.')
      return
    }

    setUploading(true)
    setError('')
    setUploadResult(null)
    setUploadAnalysis(null)

    try {
      const formData = new FormData()

      selectedFiles.forEach((file) => {
        const filename = getUploadFilename(file)

        // Preserve relative paths for folder uploads.
        formData.append('files', file, filename)
      })

      const response = await fetch(`${API_BASE_URL}/datasets/upload/analyze`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        setError(
          typeof data.detail === 'string'
            ? data.detail
            : 'Could not analyze the upload.',
        )
        return
      }

      const analysis = data as UploadAnalysis

      setUploadAnalysis(analysis)

      // If everything is new, no human review is needed.
      // Continue with the existing registration flow.
      if (analysis.status === 'all_new') {
        await registerFiles(selectedFiles)
      }
    } catch (err) {
      console.error('Upload analysis failed:', err)

      setError(
        err instanceof Error
          ? err.message
          : 'Could not connect to the backend. Make sure the FastAPI server is running on port 8000.',
      )
    } finally {
      setUploading(false)
    }
  }

  const registerNewFiles = async () => {
    if (!uploadAnalysis) {
      return
    }

    if (uploadAnalysis.new_files.length === 0) {
      setError('There are no new files available to register.')
      return
    }

    const newFilenames = new Set(
      uploadAnalysis.new_files.map((item) => item.filename),
    )

    const newFiles = selectedFiles.filter((file) =>
      newFilenames.has(getUploadFilename(file)),
    )

    if (newFiles.length === 0) {
      setError(
        'The analyzed new files could not be matched to the selected files. Please select the folder again and retry.',
      )
      return
    }

    await registerFiles(newFiles)
  }

  const cancelUploadReview = () => {
    setUploadAnalysis(null)
    setError('')
  }

  const uploadDataset = async () => {
    await analyzeUpload()
  }

  const saveContext = async () => {
    if (!contextData) {
      return
    }

    setSavingContext(true)
    setError('')
    setContextMessage('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/datasets/${contextData.dataset_id}/context`,
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            description: contextData.description,
            domain: contextData.domain,
            source_system: contextData.source_system,
            update_cadence: contextData.update_cadence,
            tables: contextData.tables.map((table) => ({
              table_id: table.table_id,
              description: table.description,
              columns: table.columns.map((column) => ({
                column_id: column.column_id,
                description: column.description,
              })),
            })),
          }),
        },
      )

      const data = await response.json()

      if (!response.ok) {
        setError(
          typeof data.detail === 'string'
            ? data.detail
            : 'Could not save dataset context.',
        )
        return
      }

      setContextMessage('Dataset context saved successfully.')
      setCurrentStage(2)
      void loadStageProgress(contextData.dataset_id)
    } catch (err) {
      console.error('Context save failed:', err)

      setError(
        err instanceof Error ? err.message : 'Could not connect to the backend.',
      )
    } finally {
      setSavingContext(false)
    }
  }

  const openFilePicker = () => {
    fileInputRef.current?.click()
  }

  const openFolderPicker = () => {
    folderInputRef.current?.click()
  }

  const updateContext = (
    field: 'description' | 'domain' | 'source_system' | 'update_cadence',
    value: string,
  ) => {
    if (!contextData) {
      return
    }

    setContextData({
      ...contextData,
      [field]: value,
    })
  }

  const updateTableDescription = (tableId: number, value: string) => {
    if (!contextData) {
      return
    }

    setContextData({
      ...contextData,
      tables: contextData.tables.map((table) =>
        table.table_id === tableId ? { ...table, description: value } : table,
      ),
    })
  }

  const updateColumnDescription = (
    tableId: number,
    columnId: number,
    value: string,
  ) => {
    if (!contextData) {
      return
    }

    setContextData({
      ...contextData,
      tables: contextData.tables.map((table) =>
        table.table_id === tableId
          ? {
              ...table,
              columns: table.columns.map((column) =>
                column.column_id === columnId
                  ? { ...column, description: value }
                  : column,
              ),
            }
          : table,
      ),
    })
  }

  /* ---------- Derived view state ---------- */

  // Stage completion comes from the backend (persisted), with local fallbacks
  // for stages 01-04 whose backends already tracked state implicitly.
  const isStageDone = (index: number): boolean => {
    if (contextData === null) return false

    const persistedKeys: Record<number, string> = {
      0: 'ingestion',
      1: 'context',
      2: 'version',
      3: 'profiling',
      4: 'semantic',
      5: 'relationships',
      6: 'recommendations',
      7: 'validation',
      8: 'execution',
      9: 'scoring',
      10: 'remediation',
      11: 'monitoring',
    }

    const stageKey = persistedKeys[index]

    if (stageKey && stageProgress) {
      return stageProgress.stages.some(
        (stage) => stage.stage_key === stageKey && stage.completed,
      )
    }

    return false
  }

  const datasetFilterText = datasetFilter.trim().toLowerCase()
  const visibleDatasets = datasets
    .filter(
      (dataset) =>
        !datasetFilterText ||
        dataset.dataset_name.toLowerCase().includes(datasetFilterText),
    )
    .sort((a, b) => b.dataset_id - a.dataset_id)

  const profilingReady =
    contextData !== null &&
    Boolean(profilingData) &&
    profilingDatasetId === contextData.dataset_id &&
    !loadingProfiling &&
    !profilingError

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="container topbar-inner">
          <div className="brand">
            <div className="brand-mark">DQ</div>

            <div>
              <div className="brand-title">Data Quality</div>
              <div className="brand-subtitle">
                Assessment &amp; Monitoring System
              </div>
            </div>
          </div>

          <div className="system-status">
            <span className="status-dot" />
            System ready
          </div>
        </div>
      </header>

      <nav className="stage-nav" aria-label="DQ pipeline stages">
        <div className="container">
          <div className="stage-list">
            {stages.map((stage, index) => {
              const isCurrent = index === currentStage
              const isAvailable = index === 0 || contextData !== null
              const isDone = isStageDone(index)

              const state = isCurrent
                ? 'is-current'
                : !isAvailable
                  ? 'is-locked'
                  : isDone
                    ? 'is-done'
                    : 'is-available'

              const label = stage.replace(/^\d+\s/, '')

              return (
                <button
                  key={stage}
                  type="button"
                  className={`stage-tab ${state}`}
                  disabled={!isAvailable}
                  aria-current={isCurrent ? 'step' : undefined}
                  title={
                    isAvailable
                      ? undefined
                      : 'Register or open a dataset first'
                  }
                  onClick={() => goToStage(index)}
                >
                  <span className="stage-number">
                    {isDone && !isCurrent ? <IconCheck /> : index + 1}
                  </span>

                  <span>
                    {index === 3 && loadingProfiling ? 'Profiling…' : label}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      </nav>

      {/* =========================================================
          STAGE 01 — DATA INGESTION
          ========================================================= */}
      {currentStage === 0 && (
        <main className="container page">
          <PageHeader
            title="Register a dataset"
            description="Upload a data file or a folder of related tables. Raw data is preserved as an immutable dataset version."
          />

          <div className="layout-split">
            <section className="card">
              <CardHeader
                title="Dataset details"
                description="The dataset name is suggested from your upload and can be edited."
              />

              <div className="section">
                <div className="form-grid">
                  <label className="field">
                    <span className="field-label">Dataset name</span>

                    <input
                      className="control"
                      value={datasetName}
                      onChange={(event) => setDatasetName(event.target.value)}
                      placeholder="Detected from uploaded file"
                    />

                    <span className="field-hint">
                      Suggested from the uploaded file or folder name.
                    </span>
                  </label>

                  <label className="field">
                    <span className="field-label">
                      Source system{' '}
                      <span className="field-optional">(optional)</span>
                    </span>

                    <select
                      className="control"
                      value={sourceSystem}
                      onChange={(event) => setSourceSystem(event.target.value)}
                    >
                      <option value="">Select source system</option>

                      {SOURCE_SYSTEMS.map((system) => (
                        <option key={system} value={system}>
                          {system}
                        </option>
                      ))}
                    </select>

                    <span className="field-hint">
                      The system that produced or owns this dataset.
                    </span>
                  </label>
                </div>
              </div>

              <div className="section">
                <div className="section-head">
                  <h3 className="section-title">Upload data</h3>

                  <p className="section-hint">
                    Use one file for a single dataset or a folder for multiple
                    related tables. Supported formats: CSV, XLSX and XLS. Each
                    registration creates a new version.
                  </p>
                </div>

                <div className="upload-options">
                  <button
                    type="button"
                    className="upload-option"
                    onClick={openFilePicker}
                  >
                    <span className="upload-option-icon">
                      <IconUpload />
                    </span>

                    <span>
                      <span className="upload-option-title">Upload file</span>
                      <span className="upload-option-text">
                        One CSV, XLSX or XLS file
                      </span>
                    </span>
                  </button>

                  <button
                    type="button"
                    className="upload-option"
                    onClick={openFolderPicker}
                  >
                    <span className="upload-option-icon">
                      <IconFolder />
                    </span>

                    <span>
                      <span className="upload-option-title">Upload folder</span>
                      <span className="upload-option-text">
                        Multiple related tables or files
                      </span>
                    </span>
                  </button>
                </div>

                <input
                  ref={fileInputRef}
                  className="hidden-input"
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFilesSelected}
                />

                <input
                  ref={folderInputRef}
                  className="hidden-input"
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  multiple
                  onChange={handleFilesSelected}
                  {...({
                    webkitdirectory: '',
                  } as Record<string, string>)}
                />

                {selectedFiles.length > 0 && (
                  <div className="table table--selected selected-files">
                    <div className="table-head">
                      <span>Type</span>
                      <span>Selected files ({selectedFiles.length})</span>
                      <span className="num">Size</span>
                    </div>

                    <div className="table-body-scroll">
                      {selectedFiles.map((file) => {
                        const filename = getUploadFilename(file)

                        return (
                          <div
                            className="table-row"
                            key={`${filename}-${file.size}`}
                          >
                            <span className="badge">
                              {file.name.split('.').pop()?.toUpperCase()}
                            </span>

                            <span className="cell-ellipsis" title={filename}>
                              {filename}
                            </span>

                            <span className="num cell-sub">
                              {(file.size / 1024).toFixed(1)} KB
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div className="card-messages">
                {/* Upload requires review */}
                {uploadAnalysis?.status === 'review_required' && (
                  <Notice tone="warning" title="Upload requires review">
                    <p className="notice-text">{uploadAnalysis.message}</p>

                    {uploadAnalysis.duplicate_files.length > 0 && (
                      <div>
                        <p className="notice-heading">
                          Duplicate files ({uploadAnalysis.duplicate_count})
                        </p>

                        <ul className="notice-list">
                          {uploadAnalysis.duplicate_files.map((item) => {
                            const existing = item.existing_files?.[0]

                            return (
                              <li className="notice-item" key={item.filename}>
                                <span className="notice-item-name">
                                  {item.filename}
                                </span>

                                <span className="notice-item-meta">
                                  {existing
                                    ? `Already registered as ${existing.original_filename} · Dataset Version ${existing.version_id}`
                                    : 'Already exists in the registered file history.'}
                                </span>
                              </li>
                            )
                          })}
                        </ul>
                      </div>
                    )}

                    {uploadAnalysis.new_files.length > 0 && (
                      <div>
                        <p className="notice-heading">
                          New files ({uploadAnalysis.new_count})
                        </p>

                        <ul className="notice-list">
                          {uploadAnalysis.new_files.map((item) => (
                            <li className="notice-item" key={item.filename}>
                              <span className="notice-item-name notice-item-name--icon">
                                <IconCheck />
                                {item.filename}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="notice-actions">
                      {uploadAnalysis.new_count > 0 && (
                        <button
                          type="button"
                          className="btn btn-primary"
                          disabled={uploading}
                          onClick={registerNewFiles}
                        >
                          {uploading
                            ? 'Registering...'
                            : `Register ${uploadAnalysis.new_count} new file${uploadAnalysis.new_count === 1 ? '' : 's'}`}
                        </button>
                      )}

                      <button
                        type="button"
                        className="btn btn-secondary"
                        disabled={uploading}
                        onClick={cancelUploadReview}
                      >
                        Cancel
                      </button>
                    </div>

                    {uploadAnalysis.new_count > 0 &&
                      uploadAnalysis.duplicate_count > 0 && (
                        <p className="notice-meta">
                          Duplicate files will not be registered. Only the new
                          files will be sent for registration.
                        </p>
                      )}
                  </Notice>
                )}

                {/* All uploaded files are duplicates */}
                {uploadAnalysis?.status === 'all_duplicates' && (
                  <Notice
                     tone="danger"
                        title={
                          uploadAnalysis.duplicate_files.length === 1
                            ? 'File already registered'
                            : 'All uploaded files already exist'
                      }
                    >
                    <p className="notice-text">
                      {uploadAnalysis.duplicate_files.length === 1
                        ? `The uploaded file "${uploadAnalysis.duplicate_files[0].filename}" has already been registered.`
                        : uploadAnalysis.message}
                    </p>
                    <ul className="notice-list">
                      {uploadAnalysis.duplicate_files.map((item) => {
                        const existing = item.existing_files?.[0]

                        return (
                          <li className="notice-item" key={item.filename}>
                            <span className="notice-item-name">
                              {item.filename}
                            </span>

                            <span className="notice-item-meta">
                              {existing
                                ? `Already registered as ${existing.original_filename} · Dataset Version ${existing.version_id}`
                                : 'Already exists in the registered file history.'}
                            </span>
                          </li>
                        )
                      })}
                    </ul>

                    <div className="notice-actions">
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={cancelUploadReview}
                      >
                        Cancel
                      </button>
                    </div>
                  </Notice>
                )}

                {error && (
                  <Notice tone="danger" title="Upload could not be completed">
                    <p className="notice-text">{error}</p>
                  </Notice>
                )}

                {uploadResult && (
                  <Notice tone="success" title="Dataset registered successfully">
                    <SummaryGrid
                      items={[
                        ['Dataset ID', uploadResult.dataset_id],
                        ['Version', `V${uploadResult.version_number}`],
                        ['Version ID', uploadResult.version_id],
                        ['Files', uploadResult.files.length],
                      ]}
                    />
                  </Notice>
                )}
              </div>

              <div className="card-footer">
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={uploading}
                  onClick={uploadDataset}
                >
                  {uploading ? 'Analyzing upload...' : 'Register dataset'}
                </button>
              </div>
            </section>

            {/* ---------- Registered datasets ---------- */}
            <aside className="card">
              <CardHeader
                title="Registered datasets"
                description="Open a dataset to review its completed stages or finish the remaining ones."
                action={
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    disabled={loadingDatasets}
                    onClick={() => void loadDatasets()}
                  >
                    {loadingDatasets ? 'Refreshing…' : 'Refresh'}
                  </button>
                }
              />

              {datasets.length > 6 && (
                <div className="list-toolbar">
                  <input
                    className="control control-sm"
                    type="search"
                    placeholder="Filter datasets"
                    aria-label="Filter datasets"
                    value={datasetFilter}
                    onChange={(event) => setDatasetFilter(event.target.value)}
                  />
                </div>
              )}

              {datasetsError && (
                <div className="card-messages">
                  <Notice
                    tone="warning"
                    title="Registered datasets could not be loaded"
                  >
                    <p className="notice-text">{datasetsError}</p>
                  </Notice>
                </div>
              )}

              {!datasetsError && datasets.length === 0 && (
                <p className="empty-state">
                  {loadingDatasets
                    ? 'Loading datasets…'
                    : 'No datasets registered yet. Upload a file or folder to register the first one.'}
                </p>
              )}

              {datasets.length > 0 && visibleDatasets.length === 0 && (
                <p className="empty-state">No datasets match this filter.</p>
              )}

              {visibleDatasets.length > 0 && (
                <ul className="dataset-list">
                  {visibleDatasets.map((dataset) => {
                    const isActive =
                      contextData?.dataset_id === dataset.dataset_id

                    return (
                      <li key={dataset.dataset_id}>
                        <div className="dataset-item-row">
                          <button
                            type="button"
                            className={`dataset-item${isActive ? ' is-active' : ''}`}
                            aria-current={isActive ? 'true' : undefined}
                            onClick={() => void openDataset(dataset.dataset_id)}
                          >
                            <span className="dataset-item-main">
                              <span className="dataset-item-name">
                                {dataset.dataset_name}
                              </span>

                              <span className="dataset-item-meta">
                                {describeDataset(dataset)}
                              </span>
                            </span>

                            <IconChevron />
                          </button>

                          <button
                            type="button"
                            className="dataset-delete-button"
                            aria-label={`Delete ${dataset.dataset_name}`}
                            title={`Delete ${dataset.dataset_name}`}
                            onClick={() => requestDeleteDataset(dataset.dataset_id)}
                          >
                            🗑
                          </button>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              )}
            </aside>
          </div>
        </main>
      )}

      {/* =========================================================
          STAGE 02 — DATASET CONTEXT
          ========================================================= */}
      {currentStage === 1 && (
        <main className="container page">
          <PageHeader
            title="Describe your dataset"
            description="Provide optional business context that helps later stages understand the meaning and intended use of the data."
            onBack={() => goToStage(0)}
          />

          {loadingContext && (
            <div className="card">
              <p className="empty-state">Loading dataset context...</p>
            </div>
          )}

          {!loadingContext && !contextData && (
            <div className="card">
              <div className="card-messages">
                <Notice tone="danger" title="Dataset context could not be loaded">
                  <p className="notice-text">
                    {error || 'The dataset could not be found.'}
                  </p>
                </Notice>
              </div>

              <div className="card-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => goToStage(0)}
                >
                  Back to datasets
                </button>
              </div>
            </div>
          )}

          {!loadingContext && contextData && (
            <section className="card">
              <CardHeader
                title="Dataset context"
                description="This information is optional. Provide what is known; the profiling and semantic stages will determine technical characteristics automatically."
              />

              <SummaryGrid
                items={[
                  ['Dataset', contextData.dataset_name],
                  ['Dataset ID', contextData.dataset_id],
                  [
                    'Current version',
                    `V${contextData.current_version.version_number}`,
                  ],
                  ['Source system', contextData.source_system || 'Not provided'],
                ]}
              />

              <div className="section">
                <div className="form-grid">
                  <label className="field">
                    <span className="field-label">Domain</span>

                    <input
                      className="control"
                      value={contextData.domain ?? ''}
                      onChange={(event) =>
                        updateContext('domain', event.target.value)
                      }
                      placeholder="e.g. Customer Management, Finance"
                    />
                  </label>

                  <label className="field">
                    <span className="field-label">Update cadence</span>

                    <select
                      className="control"
                      value={contextData.update_cadence ?? ''}
                      onChange={(event) =>
                        updateContext('update_cadence', event.target.value)
                      }
                    >
                      <option value="">Select cadence</option>

                      {UPDATE_CADENCES.map((cadence) => (
                        <option key={cadence} value={cadence}>
                          {cadence}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="field field-wide">
                    <span className="field-label">Dataset description</span>

                    <textarea
                      className="control"
                      value={contextData.description ?? ''}
                      onChange={(event) =>
                        updateContext('description', event.target.value)
                      }
                      placeholder="Describe what this dataset represents and how it is used."
                      rows={4}
                    />
                  </label>
                </div>
              </div>

              <div className="section section-lead">
                <div className="section-head">
                  <h3 className="section-title">Table context</h3>

                  <p className="section-hint">
                    Add descriptions only where they are useful. Column
                    descriptions can help the semantic understanding stage.
                  </p>
                </div>
              </div>

              {contextData.tables.map((table) => {
                const describedColumns = table.columns.filter((column) =>
                  Boolean(column.description?.trim()),
                ).length

                return (
                  <details
                    className="ctx-table"
                    key={table.table_id}
                    open={contextData.tables.length === 1}
                  >
                    <summary className="ctx-table-summary">
                      <span className="ctx-table-chevron">
                        <IconChevron />
                      </span>

                      <span className="ctx-table-title">
                        <strong>{table.table_name}</strong>

                        <span>
                          {(table.row_count ?? 0).toLocaleString()} rows,{' '}
                          {table.columns.length} columns
                        </span>
                      </span>

                      <span className="badge">
                        {describedColumns} of {table.columns.length} described
                      </span>
                    </summary>

                    <div className="ctx-table-body">
                      <label className="field">
                        <span className="field-label">Table description</span>

                        <textarea
                          className="control"
                          value={table.description ?? ''}
                          onChange={(event) =>
                            updateTableDescription(
                              table.table_id,
                              event.target.value,
                            )
                          }
                          placeholder={`Describe what ${table.table_name} contains.`}
                          rows={3}
                        />
                      </label>

                      <div className="table table--context">
                        <div className="table-head">
                          <span>Column</span>
                          <span>Data type</span>
                          <span>Description</span>
                        </div>

                        {table.columns.map((column) => (
                          <div className="table-row" key={column.column_id}>
                            <span className="cell-strong">
                              {column.column_name}
                            </span>

                            <span className="mono cell-sub">
                              {column.data_type}
                            </span>

                            <input
                              className="control control-sm"
                              aria-label={`Description for ${column.column_name}`}
                              value={column.description ?? ''}
                              onChange={(event) =>
                                updateColumnDescription(
                                  table.table_id,
                                  column.column_id,
                                  event.target.value,
                                )
                              }
                              placeholder="Optional column description"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  </details>
                )
              })}

              <div className="card-messages">
                {error && (
                  <Notice tone="danger" title="Context could not be saved">
                    <p className="notice-text">{error}</p>
                  </Notice>
                )}

                {contextMessage && (
                  <Notice tone="success" title={contextMessage} />
                )}
              </div>

              <div className="card-footer">
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={savingContext}
                  onClick={saveContext}
                >
                  {savingContext ? 'Saving context...' : 'Save and continue'}
                </button>
              </div>
            </section>
          )}
        </main>
      )}

      {/* =========================================================
          STAGE 03 — VERSION & FINGERPRINT
          ========================================================= */}
      {currentStage === 2 && contextData && (
        <main className="container page">
          <PageHeader
            title="Version & fingerprint"
            description="Review the immutable dataset version and the fingerprints recorded during ingestion."
            onBack={() => goToStage(0)}
          />

          <section className="card">
            <CardHeader
              title={`Version V${contextData.current_version.version_number}`}
              description="Original uploaded files remain immutable. This version keeps a registered snapshot of its files."
              action={<span className="badge badge-success">Recorded</span>}
            />

            <SummaryGrid
              items={[
                ['Dataset', contextData.dataset_name],
                ['Dataset ID', contextData.dataset_id],
                ['Version', `V${contextData.current_version.version_number}`],
                ['Version ID', contextData.current_version.version_id],
                [
                  'Parent version',
                  contextData.current_version.parent_version_id === null
                    ? 'None'
                    : `Version ${contextData.current_version.parent_version_id}`,
                ],
              ]}
            />

            <div className="section">
              <div className="section-head">
                <h3 className="section-title">Version integrity</h3>

                <p className="section-hint">
                  Schema and content fingerprints are associated with this
                  version.
                </p>
              </div>

              <div className="fingerprint-list">
                <div className="fingerprint-row">
                  <span className="fingerprint-label">Schema fingerprint</span>

                  <code className="code">
                    {contextData.current_version.schema_fingerprint ||
                      'Not available'}
                  </code>
                </div>

                <div className="fingerprint-row">
                  <span className="fingerprint-label">
                    Dataset content fingerprint
                  </span>

                  <code className="code">
                    {contextData.current_version.content_fingerprint ||
                      'Not available'}
                  </code>
                </div>
              </div>
            </div>

            <div className="section">
              <div className="section-head">
                <h3 className="section-title">
                  {contextData.current_version.files.length} file
                  {contextData.current_version.files.length === 1 ? '' : 's'} in
                  this version
                </h3>
              </div>

              <div className="table table--files">
                <div className="table-head">
                  <span>File</span>
                  <span>Fingerprint</span>
                </div>

                {contextData.current_version.files.map((file) => (
                  <div className="table-row" key={file.file_id}>
                    <div className="cell-main">
                      <span className="cell-strong cell-wrap">
                        {file.filename}
                      </span>

                      {file.filename !== file.stored_filename && (
                        <span className="cell-sub cell-wrap">
                          {file.stored_filename}
                        </span>
                      )}
                    </div>

                    <code
                      className="code code-truncate"
                      title={file.content_fingerprint}
                    >
                      {shortenFingerprint(file.content_fingerprint)}
                    </code>
                  </div>
                ))}
              </div>
            </div>

            <div className="card-footer">
              <button
                type="button"
                className="btn btn-primary"
                disabled={loadingProfiling}
                onClick={() => openProfilingStage(contextData.dataset_id)}
              >
                {loadingProfiling
                  ? 'Loading profiling...'
                  : 'Continue to profiling'}
              </button>
            </div>
          </section>
        </main>
      )}

      {/* =========================================================
          STAGE 04 — PROFILING
          ========================================================= */}
      {currentStage === 3 && contextData && (
        <main className="container page">
          <PageHeader
            title="Data profiling"
            description="Analyze the structure, completeness, uniqueness, patterns and statistical characteristics of the dataset."
            onBack={() => goToStage(0)}
          />

          <section className="card">
            <CardHeader
              title="Profile results"
              description="Completeness, uniqueness and value statistics for every table in this version."
              action={
                profilingReady ? (
                  <span className="badge badge-success">Generated</span>
                ) : undefined
              }
            />

            {loadingProfiling && (
              <p className="empty-state">Generating data profile...</p>
            )}

            {profilingError && !loadingProfiling && (
              <div className="card-messages">
                <Notice tone="danger" title="Profiling could not be loaded">
                  <p className="notice-text">{profilingError}</p>

                  <div className="notice-actions">
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => void loadProfiling(contextData.dataset_id)}
                    >
                      Try again
                    </button>
                  </div>
                </Notice>
              </div>
            )}

            {profilingReady && (
              <>
                <SummaryGrid
                  items={[
                    ['Dataset', profilingData.dataset_name],
                    ['Version', `V${profilingData.version_number}`],
                    ['Tables', formatNumber(profilingData.table_count)],
                    ['Total rows', formatNumber(profilingData.total_rows)],
                    ['Total columns', formatNumber(profilingData.total_columns)],
                  ]}
                />

                <div className="section">
                  <div className="section-head">
                    <h3 className="section-title">Table overview</h3>

                    <p className="section-hint">
                      Select a table to see column-level statistics.
                    </p>
                  </div>

                  <div className="table table--tables">
                    <div className="table-head">
                      <span>Table</span>
                      <span className="num">Rows</span>
                      <span className="num">Columns</span>
                      <span />
                    </div>

                    {profilingData.tables.map((table: any) => {
                      const tableKey = `${table.source_file}-${table.table_name}`
                      const isExpanded = expandedProfileTables.has(tableKey)

                      return (
                        <div className="table-group" key={tableKey}>
                          <div
                            className={`table-row is-clickable${isExpanded ? ' is-open' : ''}`}
                            role="button"
                            tabIndex={0}
                            aria-expanded={isExpanded}
                            onClick={() => toggleProfileTable(tableKey)}
                            onKeyDown={(event) => {
                              if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault()
                                toggleProfileTable(tableKey)
                              }
                            }}
                          >
                            <div className="cell-main">
                              <span className="cell-strong cell-wrap">
                                {table.table_name}
                              </span>

                              <span className="cell-sub cell-wrap">
                                {table.source_file}
                              </span>
                            </div>

                            <span className="num">
                              {formatNumber(table.row_count)}
                            </span>

                            <span className="num">
                              {formatNumber(table.column_count)}
                            </span>

                            <span className="row-toggle">
                              <span className="row-toggle-label">
                                {isExpanded ? 'Hide columns' : 'View columns'}
                              </span>

                              <IconChevron />
                            </span>
                          </div>

                          {isExpanded && (                              <div className="table-expand">
                              <div className="profile-evidence">
                                <span className="profile-stat">
                                  <span className="profile-evidence-strong">Complete duplicate rows</span>{' '}
                                  {formatNumber(table.complete_duplicate_rows ?? 0)}
                                  {table.complete_duplicate_excess_count > 0 && (
                                    <span> (excess {formatNumber(table.complete_duplicate_excess_count)})</span>
                                  )}
                                </span>
                                {(table.composite_uniqueness_candidates || []).map((composite: any) => (
                                  <span className="profile-stat" key={composite.columns.join('|')}>
                                    <span className="profile-evidence-strong">Composite uniqueness candidate</span>{' '}
                                    <span>{composite.columns.join(' + ')}</span>{' '}
                                    {Number(composite.composite_uniqueness_percentage ?? 0).toFixed(4)}% unique
                                    {composite.key_like_members?.length > 0 && (
                                      <span className="cell-sub"> (key-like: {composite.key_like_members.join(', ')})</span>
                                    )}
                                    {composite.contains_measure && (
                                      <span className="cell-sub"> (contains measure)</span>
                                    )}
                                    {Number(composite.composite_uniqueness_percentage ?? 0) >= 99.9 && (
                                      <span className="cell-sub"> (near-exact)</span>
                                    )}
                                    {composite.sampled && (
                                      <span className="cell-sub"> (sampled: {formatCount(composite.sample_rows)} rows)</span>
                                    )}
                                  </span>
                                ))}
                                {table.columns.filter((c: any) => c.identifier_signal || c.identifier_like).length > 0 && (
                                  table.columns
                                    .filter((c: any) => c.identifier_signal || c.identifier_like)
                                    .map((column: any) => (
                                      <span className="profile-evidence-stack" key={column.column_name}>
                                        <span className="profile-stat">
                                          <span className="profile-evidence-strong">Identifier candidate</span>{' '}
                                          <span>{column.column_name}</span>
                                        </span>
                                        <span className="profile-evidence-sub">
                                          <span>Name signal {column.identifier_name_signal ? 'Yes' : 'No'}</span>
                                          <span>Uniqueness {formatStat(column.identifier_uniqueness_percentage)}%</span>
                                          <span>Completeness {formatStat(column.identifier_completeness_percentage)}%</span>
                                          {column.identifier_like && <span>Identifier-like evidence: yes</span>}
                                          {column.identifier_repeats && <span>Repeats values (natural key)</span>}
                                        </span>
                                      </span>
                                    ))
                                )}
                              </div>
                              {table.row_completeness && (
                                <div className="profile-completeness">
                                  <span className="profile-stat">
                                    <span className="profile-evidence-strong">Fully complete rows</span>{' '}
                                    {formatStat(table.row_completeness.fully_complete_percentage ?? '—')}%
                                    {' '}({formatCount(table.row_completeness.fully_complete_rows)} of {formatCount(table.row_completeness.row_count)})
                                  </span>
                                  {(table.row_completeness.co_missing_patterns || []).slice(0, 5).map((pattern: any) => (
                                    <span className="profile-stat" key={pattern.columns.join('|')}>
                                      <span className="profile-evidence-strong">Co-missing</span>{' '}
                                      <span>{pattern.columns.join(' + ')}</span>{' '}
                                      {formatCount(pattern.rows)} rows
                                    </span>
                                  ))}
                                  {table.row_completeness.sampled && (
                                    <span className="cell-sub">(sampled: {formatCount(table.row_completeness.sample_rows)} rows)</span>
                                  )}
                                </div>
                              )}
                              {table.functional_dependencies && (table.functional_dependencies.dependencies || []).length > 0 && (
                                <div className="profile-deps">
                                  <span className="profile-evidence-strong">Determines (functional dependencies)</span>
                                  {(table.functional_dependencies.dependencies || []).slice(0, 8).map((dependency: any) => (
                                    <span className="profile-dep-line" key={`${dependency.determinant}->${dependency.dependent}`}>
                                      <span className="mono">{dependency.determinant} → {dependency.dependent}</span>{' '}
                                      {formatStat(dependency.coverage_percentage)}% of rows
                                      {dependency.bidirectional && (
                                        <span className="profile-dep-chips">1:1</span>
                                      )}
                                      {dependency.sampled && (
                                        <span className="profile-dep-chips">sampled</span>
                                      )}
                                    </span>
                                  ))}
                                </div>
                              )}
                              <div className="table-scroll">
                                <div className="table table--profile">
                                  <div className="table-head">
                                    <span>Column</span>
                                    <span className="num">Nulls</span>
                                    <span className="num">Distinct values</span>
                                    <span className="num">Empty strings</span>
                                    <span className="num">Whitespace only</span>
                                    <span>Identifier evidence</span>
                                    <span>Value statistics</span>
                                  </div>

                                  {table.columns.map((column: any) => (
                                    <div
                                      className="table-row"
                                      key={column.column_name}
                                    >
                                      <div className="cell-main">
                                        <span className="cell-strong cell-wrap">
                                          {column.column_name}
                                        </span>

                                        <span className="mono cell-sub">
                                          {column.data_type}
                                        </span>

                                        {column.identifier_signal && (
                                          <span className="badge badge-accent">
                                            Candidate identifier
                                          </span>
                                        )}
                                      </div>

                                      <span className="num">
                                        <span className="stat-main">
                                          {formatNumber(column.null_count)}
                                        </span>
                                        <span className="stat-sub">
                                          {formatPercent(column.null_percentage)}
                                        </span>
                                      </span>

                                      <span className="num">
                                        <span className="stat-main">
                                          {formatNumber(column.distinct_count)}
                                        </span>
                                        <span className="stat-sub">
                                          {formatPercent(
                                            column.distinct_percentage,
                                          )}
                                        </span>
                                      </span>

                                      <span className="num">
                                        <span className="stat-main">
                                          {formatNumber(
                                            column.empty_string_count,
                                          )}
                                        </span>
                                        <span className="stat-sub">
                                          {formatPercent(
                                            column.empty_string_percentage,
                                          )}
                                        </span>
                                      </span>

                                      <span className="num">
                                        <span className="stat-main">
                                          {formatNumber(
                                            column.whitespace_only_count,
                                          )}
                                        </span>
                                        <span className="stat-sub">
                                          {formatPercent(
                                            column.whitespace_only_percentage,
                                          )}
                                        </span>
                                      </span>

                                      <div className="profile-identifier-cell">
                                        {column.identifier_signal ? (
                                          <>
                                            <span className="badge badge-success">Candidate identifier</span>
                                            <span className="profile-identifier-line">
                                              <span className="profile-stat-label">Name signal</span>{' '}
                                              {column.identifier_name_signal ? 'Yes' : 'No'}
                                            </span>
                                            <span className="profile-identifier-line">
                                              <span className="profile-stat-label">Uniqueness</span>{' '}
                                              {formatStat(column.identifier_uniqueness_percentage)}%
                                            </span>
                                            <span className="profile-identifier-line">
                                              <span className="profile-stat-label">Completeness</span>{' '}
                                              {formatStat(column.identifier_completeness_percentage)}%
                                            </span>
                                            {column.identifier_like && (
                                              <span className="profile-identifier-line">
                                                <span className="profile-stat-label">Evidence</span>{' '}
                                                {(column.identifier_like_reasons || []).join('; ') || 'value-shape evidence'}
                                              </span>
                                            )}
                                          </>
                                        ) : (
                                          <span className="cell-sub">—</span>
                                        )}
                                      </div>

                                      <div className="detail-cell profile-detail-blocks">
                                        {column.numeric && (
                                          <div className="profile-detail-block">
                                            <strong>Numeric</strong>

                                            <div className="profile-stat-group">
                                              <span className="profile-stat"><span className="profile-stat-label">min</span> {formatStat(column.numeric.min)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">max</span> {formatStat(column.numeric.max)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">mean</span> {formatStat(column.numeric.mean)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">median</span> {formatStat(column.numeric.median)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">std</span> {formatStat(column.numeric.std)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">q25</span> {formatStat(column.numeric.q25)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">q50</span> {formatStat(column.numeric.q50)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">q75</span> {formatStat(column.numeric.q75)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">IQR</span> {formatStat(column.numeric.iqr)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">MAD</span> {formatStat(column.numeric.mad)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">count</span> {formatStat(column.numeric.count)}</span>
                                              {column.numeric.constant && (
                                                <span className="profile-stat"><span className="badge badge-warning">Constant (single value)</span></span>
                                              )}
                                              {!column.numeric.constant && column.numeric.near_constant && (
                                                <span className="profile-stat"><span className="badge badge-warning">Near-constant</span> <span className="cell-sub">top value {formatStat(column.numeric.top_value_share_percentage ?? 0)}% of rows</span></span>
                                              )}
                                              {column.numeric.meaningful_statistics === false && (
                                                <span className="profile-stat"><span className="badge badge-warning">Statistics not meaningful</span> <span className="cell-sub">values look like codes/labels</span></span>
                                              )}
                                              {firstDefined(column.numeric.zero_count, column.numeric.negative_count, column.numeric.positive_count) !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">zero / negative / positive</span> {formatCount(column.numeric.zero_count)} / {formatCount(column.numeric.negative_count)} / {formatCount(column.numeric.positive_count)}</span>
                                              )}
                                              {column.numeric.integer_valued !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">integer valued</span> {column.numeric.integer_valued ? 'yes' : 'no'}{column.numeric.max_decimal_places !== undefined && <> (max {formatStat(column.numeric.max_decimal_places)} decimals)</>}</span>
                                              )}
                                              {column.numeric.p01 !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">p01 / p05 / p95 / p99</span> {formatStat(column.numeric.p01)} / {formatStat(column.numeric.p05)} / {formatStat(column.numeric.p95)} / {formatStat(column.numeric.p99)}</span>
                                              )}
                                              {column.numeric.skewness !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">skewness</span> {formatStat(column.numeric.skewness)}</span>
                                              )}
                                              {column.numeric.suspicious_sentinel && (
                                                <span className="profile-stat"><span className="badge badge-warning">Suspicious sentinel values</span> <span className="cell-sub">e.g. {formatStat(column.numeric.suspicious_sentinel?.value)}</span></span>
                                              )}
                                              {column.numeric.code_like && (
                                                <span className="profile-stat"><span className="badge badge-accent">Code-like numeric</span> <span className="cell-sub">stored as {formatStat(column.stored_as ?? column.numeric.stored_as ?? '—')}, semantic type mismatch</span></span>
                                              )}
                                              {column.numeric.leading_zero_loss_suspected && (
                                                <span className="profile-stat"><span className="badge badge-warning">Leading zeros lost</span> <span className="cell-sub">{formatStat(column.numeric.leading_zero_loss_count ?? 0)} values ({formatStat(column.numeric.leading_zero_loss_percentage ?? 0)}%)</span></span>
                                              )}
                                              {column.numeric.digit_length_distribution && Object.keys(column.numeric.digit_length_distribution).length > 0 && (
                                                <span className="profile-stat profile-digit-lengths"><span className="profile-stat-label">digit lengths</span> {Object.entries(column.numeric.digit_length_distribution).map(([length, count]) => <span key={length} className="profile-dep-chips">{length}: {formatStat(count)}</span>)}</span>
                                              )}
                                            </div>
                                          </div>
                                        )}

                                        {column.text && (
                                          <div className="profile-detail-block">
                                            <strong>Text</strong>

                                            <div className="profile-stat-group">
                                              <span className="profile-stat"><span className="profile-stat-label">length min–max</span> {formatStat(column.text.min_length)}–{formatStat(column.text.max_length)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">mean length</span> {formatStat(column.text.mean_length)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">median length</span> {formatStat(column.text.median_length)}</span>
                                            </div>
                                            {column.text.patterns && (
                                              <div className="profile-stat-group" style={{ marginTop: 6 }}>
                                                <span className="profile-stat"><span className="profile-stat-label">email-like</span> {formatStat(column.text.patterns.email_like)}</span>
                                                <span className="profile-stat"><span className="profile-stat-label">numeric-like</span> {formatStat(column.text.patterns.numeric_like)}</span>
                                                <span className="profile-stat"><span className="profile-stat-label">date-like</span> {formatStat(column.text.patterns.date_like)}</span>
                                                <span className="profile-stat"><span className="profile-stat-label">postal-like</span> {formatStat(column.text.patterns.postal_like)}</span>
                                                <span className="profile-stat"><span className="profile-stat-label">alphanumeric</span> {formatStat(column.text.patterns.alphanumeric_like)}</span>
                                                <span className="profile-stat"><span className="profile-stat-label">special chars</span> {formatStat(column.text.patterns.contains_special_character)}</span>
                                                <span className="profile-stat"><span className="profile-stat-label">contains whitespace</span> {formatStat(column.text.patterns.contains_whitespace)}</span>
                                              </div>
                                            )}
                                            {(column.text.constant || column.text.near_constant) && (
                                              <div className="profile-stat-group" style={{ marginTop: 6 }}>
                                                {column.text.constant && <span className="profile-stat"><span className="badge badge-warning">Constant (single value)</span></span>}
                                                {!column.text.constant && column.text.near_constant && <span className="profile-stat"><span className="badge badge-warning">Near-constant</span> <span className="cell-sub">top value {formatStat(column.text.top_value_share_percentage ?? 0)}% of rows</span></span>}
                                              </div>
                                            )}
                                            {(column.identifier_like || column.text.code_like) && (
                                              <div className="profile-stat-group" style={{ marginTop: 6 }}>
                                                {column.identifier_like && (
                                                  <span className="profile-stat"><span className="badge badge-accent">Identifier-like</span> <span className="cell-sub">{(column.identifier_like_reasons || []).join('; ') || 'shape evidence'}</span></span>
                                                )}
                                                {column.identifier_repeats && <span className="profile-stat"><span className="profile-stat-label">repeats values</span> yes (natural key, not unique)</span>}
                                                {column.text.code_like && <span className="profile-stat"><span className="badge badge-accent">Code-like</span></span>}
                                              </div>
                                            )}
                                            {column.text.shape && (
                                              <div className="profile-stat-group" style={{ marginTop: 6 }}>
                                                {column.datetime && column.datetime.detected_format ? (
                                                  <span className="profile-stat"><span className="profile-stat-label">format</span> <span className="mono">{String(column.datetime.detected_format)}</span></span>
                                                ) : (
                                                  <>
                                                    <span className="profile-stat"><span className="profile-stat-label">dominant shape</span> <span className="mono">{formatStat(column.text.shape.dominant_display_shape ?? column.text.shape.dominant_shape)}</span> ({formatStat(column.text.shape.dominant_coverage_percentage)}% coverage{column.text.shape.is_regular ? ', regular' : ''})</span>
                                                    {(column.text.shape.top_shapes || []).length > 1 && (
                                                      <span className="profile-stat profile-shapes"><span className="profile-stat-label">top shapes</span> {(column.text.shape.top_shapes || []).map((shape: any) => <span key={shape.shape} className="profile-dep-chips"><span className="mono">{shape.display_shape ?? shape.shape}</span> {formatStat(shape.percentage)}%</span>)}</span>
                                                    )}
                                                    {column.text.shape.suggested_regex && <span className="profile-stat"><span className="profile-stat-label">suggested regex</span> <span className="mono">{String(column.text.shape.suggested_regex)}</span></span>}
                                                    {column.text.shape.skipped_reason && <span className="profile-stat cell-sub">shapes skipped: {formatStat(column.text.shape.skipped_reason)}</span>}
                                                  </>
                                                )}
                                              </div>
                                            )}
                                            {column.text.separators && (
                                              <div className="profile-stat-group" style={{ marginTop: 6 }}>
                                                <span className="profile-stat"><span className="profile-stat-label">separators</span> {formatStat(column.text.separators.summary)}</span>
                                              </div>
                                            )}
                                            {(column.text.disguised_missing_count > 0 || column.text.leading_trailing_whitespace_count > 0 || column.unhashable_values || (typeof column.text.case_variant_groups === 'number' && column.text.case_variant_groups > 0) || column.text.normalized_distinct_count !== undefined) && (
                                              <div className="profile-stat-group" style={{ marginTop: 6 }}>
                                                {column.text.disguised_missing_count > 0 && (
                                                  <span className="profile-stat"><span className="badge badge-warning">Disguised missing</span> <span className="cell-sub">{formatCount(column.text.disguised_missing_count)} values ({formatStat(column.text.disguised_missing_percentage ?? 0)}%){(column.text.disguised_missing_values || []).length > 0 && <>: {(column.text.disguised_missing_values || []).map((value: any) => `"${String(value)}"`).join(', ')}</>}</span></span>
                                                )}
                                                {column.text.leading_trailing_whitespace_count > 0 && (
                                                  <span className="profile-stat"><span className="badge badge-warning">Leading/trailing whitespace</span> {formatCount(column.text.leading_trailing_whitespace_count)} values</span>
                                                )}
                                                {column.text.normalized_distinct_count !== undefined && column.text.normalized_distinct_count !== column.distinct_count && (
                                                  <span className="profile-stat"><span className="profile-stat-label">distinct after normalization</span> {formatCount(column.text.normalized_distinct_count)} of {formatCount(column.distinct_count)}</span>
                                                )}
                                                {typeof column.text.case_variant_groups === 'number' && column.text.case_variant_groups > 0 && (
                                                  <span className="profile-stat"><span className="badge badge-warning">Case variants</span> <span className="cell-sub">{formatCount(column.text.case_variant_groups)} group(s)</span></span>
                                                )}
                                                {column.unhashable_values && <span className="profile-stat"><span className="badge badge-warning">Unhashable values</span> <span className="cell-sub">counted as text (lists/dicts)</span></span>}
                                              </div>
                                            )}
                                          </div>
                                        )}

                                        {column.datetime && (
                                          <div className="profile-detail-block">
                                            <strong>Datetime evidence</strong>

                                            <div className="profile-stat-group">
                                              <span className="profile-stat"><span className="profile-stat-label">date-like</span> {formatStat(column.datetime.date_like_count)} ({formatStat(column.datetime.date_like_percentage)}%)</span>
                                              <span className="profile-stat"><span className="profile-stat-label">format valid</span> {formatStat(column.datetime.format_valid_percentage)}%</span>
                                              <span className="profile-stat"><span className="profile-stat-label">min</span> {formatStat(column.datetime.min)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">max</span> {formatStat(column.datetime.max)}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">monotonic increasing</span> {column.datetime.monotonic_increasing ? 'yes' : 'no'}</span>
                                              <span className="profile-stat"><span className="profile-stat-label">monotonic decreasing</span> {column.datetime.monotonic_decreasing ? 'yes' : 'no'}</span>
                                              {column.datetime.detected_format && (
                                                <span className="profile-stat"><span className="profile-stat-label">detected format</span> <span className="mono">{String(column.datetime.detected_format)}</span>{column.datetime.format_ambiguous && <> (ambiguous, confidence {formatStat(column.datetime.format_confidence_percentage)}%)</>}</span>
                                              )}
                                              {column.datetime.has_time_component !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">has time component</span> {column.datetime.has_time_component ? 'yes' : 'no'}</span>
                                              )}
                                              {column.datetime.future_date_percentage !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">future dates</span> {formatStat(column.datetime.future_date_percentage)}%</span>
                                              )}
                                              {column.datetime.span_days !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">span</span> {formatCount(column.datetime.span_days)} days, {formatCount(column.datetime.distinct_dates)} distinct dates</span>
                                              )}
                                              {column.datetime.null_or_unparseable_count !== undefined && (
                                                <span className="profile-stat"><span className="profile-stat-label">null or unparseable</span> {formatCount(column.datetime.null_or_unparseable_count)}</span>
                                              )}
                                            </div>
                                          </div>
                                        )}

                                        {column.categorical && (
                                          <div className="profile-detail-block">
                                            <strong>Categorical</strong>

                                            <div className="profile-stat-group">
                                              <span className="profile-stat"><span className="profile-stat-label">category count</span> {formatStat(column.categorical.category_count)}</span>
                                              {(column.categorical.top_values || []).slice(0, 5).map((topValue: any) => (
                                                <span className="profile-stat" key={String(topValue.value)}>
                                                  <span className="profile-stat-label">{String(topValue.value)}</span> {formatStat(topValue.count)} ({formatStat(topValue.percentage)}%)
                                                </span>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              </>
            )}

            {profilingReady && (
              <div className="card-footer">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => goToStage(4)}
                >
                  Move to Semantic Understanding →
                </button>
              </div>
            )}
          </section>
        </main>
      )}

      {/* =========================================================
          STAGE 05 — SEMANTIC UNDERSTANDING
          ========================================================= */}
      {currentStage === 4 && contextData && (
        <SemanticStage
          key={`semantic-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(5)
          }}
        />
      )}

      {/* =========================================================
          STAGE 06 — RELATIONSHIP DISCOVERY
          ========================================================= */}
      {currentStage === 5 && contextData && (
        <RelationshipsStage
          key={`relationships-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(6)
          }}
        />
      )}

      {/* =========================================================
          STAGES 07 — METRIC & RULE RECOMMENDATION + VALIDATION
          ========================================================= */}
      {currentStage === 6 && contextData && (
        <RulesStage
          key={`rules-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(7)
          }}
        />
      )}

      {/* =========================================================
          STAGE 08 — AUTHORITATIVE EXECUTION
          ========================================================= */}
      {currentStage === 7 && contextData && (
        <ExecutionStage
          key={`execution-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(8)
          }}
        />
      )}

      {/* =========================================================
          STAGE 09 — SCORING & RCA
          ========================================================= */}
      {currentStage === 8 && contextData && (
        <ScoringStage
          key={`scoring-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(9)
          }}
        />
      )}

      {/* =========================================================
          STAGE 10 — REMEDIATION & REASSESSMENT
          ========================================================= */}
      {currentStage === 9 && contextData && (
        <RemediationStage
          key={`remediation-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(10)
          }}
        />
      )}

      {/* =========================================================
          STAGE 11 — MONITORING
          ========================================================= */}
      {currentStage === 10 && contextData && (
        <MonitoringStage
          key={`monitoring-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(11)
          }}
        />
      )}

      {/* =========================================================
          STAGE 12 — FEEDBACK & OFFLINE LEARNING
          ========================================================= */}
      {currentStage === 11 && contextData && (
        <FeedbackStage
          key={`feedback-${contextData.dataset_id}`}
          datasetId={contextData.dataset_id}
          onStageComplete={() => {
            void loadStageProgress(contextData.dataset_id)
            goToStage(0)
          }}
        />
      )}
            {datasetToDelete && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !deletingDataset) {
              setDatasetToDelete(null)
            }
          }}
        >
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-dataset-title"
            aria-describedby="delete-dataset-description"
          >
            <div className="modal-header">
              <h2 id="delete-dataset-title">Delete dataset?</h2>
            </div>

            <div className="modal-body">
              <p className="modal-dataset-name">
                {datasetToDelete.dataset_name}
              </p>

              <p
                id="delete-dataset-description"
                className="modal-description"
              >
                This will remove the dataset from Registered Datasets.
                Its versions, lineage, and raw files will be preserved.
              </p>
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-ghost"
                disabled={deletingDataset}
                onClick={() => setDatasetToDelete(null)}
              >
                Cancel
              </button>

              <button
                type="button"
                className="btn btn-danger"
                disabled={deletingDataset}
                onClick={() => void deleteDataset()}
              >
                {deletingDataset ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}


export default App
