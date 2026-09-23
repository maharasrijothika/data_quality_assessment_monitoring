import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'
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

const API_BASE_URL = 'http://127.0.0.1:8000'

const stages = [
  '01 Ingestion',
  '02 Context',
  '03 Version',
  '04 Profiling',
  '05 Metrics & Rules',
  '06 Validation',
  '07 Execution',
  '08 RCA',
  '09 Remediation',
  '10 Monitoring',
]

// Stages 01–04 are implemented; the rest stay locked in the navigation.
const LAST_AVAILABLE_STAGE = 3

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

  // Registered datasets + per-dataset stage progress seen in this browser session.
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [datasetsError, setDatasetsError] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')
  const [savedContextIds, setSavedContextIds] = useState<Set<number>>(new Set())
  const [profiledIds, setProfiledIds] = useState<Set<number>>(new Set())

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
      const response = await fetch(`${API_BASE_URL}/datasets`)

      if (!response.ok) {
        throw new Error(
          response.status === 404 || response.status === 405
            ? 'The backend has no dataset list endpoint yet (GET /datasets).'
            : `Dataset list request failed with status ${response.status}.`,
        )
      }

      setDatasets(normalizeDatasets(await response.json()))
    } catch (err) {
      console.error('Dataset list request failed:', err)

      setDatasetsError(
        err instanceof Error
          ? err.message
          : 'Could not connect to the backend.',
      )
    } finally {
      setLoadingDatasets(false)
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
      setProfiledIds((current) => new Set(current).add(datasetId))
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

    await loadContext(datasetId)
  }

  const goToStage = (index: number) => {
    setError('')
    setContextMessage('')

    if (index === 0) {
      setCurrentStage(0)
      return
    }

    if (!contextData || index > LAST_AVAILABLE_STAGE) {
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
      setSavedContextIds((current) =>
        new Set(current).add(contextData.dataset_id),
      )
      setCurrentStage(2)
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

  /* ---------- Derived view state (no new business logic) ---------- */

  // Context counts as provided once it was saved this session or has any content.
  const contextProvided =
    contextData !== null &&
    (savedContextIds.has(contextData.dataset_id) ||
      [
        contextData.description,
        contextData.domain,
        contextData.update_cadence,
      ].some((value) => Boolean(value?.trim())))

  const isStageDone = (index: number): boolean => {
    if (contextData === null) return false

    switch (index) {
      case 0:
      case 2:
        return true
      case 1:
        return contextProvided
      case 3:
        return profiledIds.has(contextData.dataset_id)
      default:
        return false
    }
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
              const isAvailable =
                index === 0 ||
                (index <= LAST_AVAILABLE_STAGE && contextData !== null)
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
                      : index > LAST_AVAILABLE_STAGE
                        ? 'Not available yet'
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
                  <Notice tone="danger" title="All uploaded files already exist">
                    <p className="notice-text">{uploadAnalysis.message}</p>

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

                          {isExpanded && (
                            <div className="table-expand">
                              <div className="table-scroll">
                                <div className="table table--profile">
                                  <div className="table-head">
                                    <span>Column</span>
                                    <span className="num">Nulls</span>
                                    <span className="num">Distinct values</span>                                    <span className="num">Empty strings</span>
                                    <span className="num">Whitespace only</span>
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
                                            Identifier evidence
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

                                      <div className="detail-cell">
                                        {column.numeric && (
                                          <div className="detail-line">
                                            <strong>Numeric</strong>
                                            <span>
                                              Min {formatStat(column.numeric.min)}
                                            </span>
                                            <span>
                                              Max {formatStat(column.numeric.max)}
                                            </span>
                                            <span>
                                              Mean {formatStat(column.numeric.mean)}
                                            </span>

                                          </div>
                                        )}

                                        {column.text && (
                                          <div className="detail-line">
                                            <strong>Text</strong>
                                            <span>
                                              Length{' '}
                                              {formatStat(column.text.min_length)}
                                              –
                                              {formatStat(column.text.max_length)}
                                            </span>
                                            <span>
                                              Mean length{' '}
                                              {formatStat(
                                                column.text.mean_length,
                                              )}
                                            </span>
                                          </div>
                                        )}

                                        {column.datetime && (
                                          <div className="detail-line">
                                            <strong>Datetime</strong>
                                            <span>
                                              {formatStat(column.datetime.min)}{' '}
                                              to{' '}
                                              {formatStat(column.datetime.max)}
                                            </span>
                                          </div>
                                        )}

                                        {!column.numeric &&
                                          !column.text &&
                                          !column.datetime && (
                                            <span className="cell-sub">—</span>
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
          </section>
        </main>
      )}
    </div>
  )
}

export default App
