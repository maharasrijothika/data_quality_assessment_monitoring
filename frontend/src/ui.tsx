import type { ReactNode } from 'react'

/* ---------------------------------------------------------------------------
   Shared presentational components (extended design system).
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

export function IconWarning() {
  return (
    <svg {...iconProps}>
      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
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
  tone: 'success' | 'warning' | 'danger' | 'info'
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
        {tone === 'warning' && <IconWarning />}
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

/* Status badge mapping shared by all later stages. */

export type StatusTone = 'neutral' | 'success' | 'warning' | 'danger' | 'accent'

export function StatusBadge({
  tone,
  children,
}: {
  tone: StatusTone
  children: ReactNode
}) {
  const classNames: Record<StatusTone, string> = {
    neutral: 'badge',
    success: 'badge badge-success',
    warning: 'badge badge-warning',
    danger: 'badge badge-danger',
    accent: 'badge badge-accent',
  }

  return <span className={classNames[tone]}>{children}</span>
}

export function confidenceTone(level: string): StatusTone {
  switch (level) {
    case 'Strong':
    case 'Approved':
      return 'success'
    case 'Probable':
      return 'accent'
    case 'Ambiguous':
      return 'warning'
    default:
      return 'neutral'
  }
}

export function ruleStatusTone(status: string): StatusTone {
  switch (status) {
    case 'approved':
      return 'success'
    case 'rejected':
      return 'danger'
    case 'VALID':
      return 'success'
    case 'NEEDS_REVIEW':
      return 'warning'
    case 'INVALID':
      return 'danger'
    default:
      return 'neutral'
  }
}

export function scoreTone(score: number): StatusTone {
  if (score >= 95) return 'success'
  if (score >= 85) return 'accent'
  if (score >= 70) return 'warning'
  return 'danger'
}
