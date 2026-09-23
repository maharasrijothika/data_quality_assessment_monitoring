import type { ReactNode } from 'react'

/* ---------------------------------------------------------------------------
   Shared presentational components for Stages 01–04.
   Purely visual: no state, no data fetching.
   --------------------------------------------------------------------------- */

const iconProps = {
  width: 16,
  height: 16,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
} as const

export function IconCheck() {
  return (
    <svg {...iconProps}>
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

export function IconChevron() {
  return (
    <svg {...iconProps}>
      <polyline points="9 6 15 12 9 18" />
    </svg>
  )
}

export function IconUpload() {
  return (
    <svg {...iconProps} width={20} height={20}>
      <path d="M12 16V4" />
      <polyline points="7 9 12 4 17 9" />
      <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
    </svg>
  )
}

export function IconFolder() {
  return (
    <svg {...iconProps} width={20} height={20}>
      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    </svg>
  )
}

type PageHeaderProps = {
  title: string
  description: string
  onBack?: () => void
}

export function PageHeader({ title, description, onBack }: PageHeaderProps) {
  return (
    <header className="page-header">
      {onBack && (
        <button type="button" className="back-link" onClick={onBack}>
          <IconChevron />
          All datasets
        </button>
      )}
      <h1 className="page-title">{title}</h1>
      <p className="page-description">{description}</p>
    </header>
  )
}

type CardHeaderProps = {
  title: string
  description?: string
  action?: ReactNode
}

export function CardHeader({ title, description, action }: CardHeaderProps) {
  return (
    <div className="card-header">
      <div>
        <h2 className="card-title">{title}</h2>
        {description && <p className="card-description">{description}</p>}
      </div>
      {action && <div className="card-header-action">{action}</div>}
    </div>
  )
}

type NoticeProps = {
  tone: 'success' | 'warning' | 'danger'
  title: string
  children?: ReactNode
}

export function Notice({ tone, title, children }: NoticeProps) {
  return (
    <div
      className={`notice notice-${tone}`}
      role={tone === 'danger' ? 'alert' : 'status'}
    >
      <p className="notice-title">
        {tone === 'success' && <IconCheck />}
        {title}
      </p>
      {children}
    </div>
  )
}

type SummaryGridProps = {
  items: ReadonlyArray<readonly [string, ReactNode]>
}

export function SummaryGrid({ items }: SummaryGridProps) {
  return (
    <dl className="summary-grid">
      {items.map(([label, value]) => (
        <div className="summary-item" key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}
