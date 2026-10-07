export const API_BASE_URL = 'http://127.0.0.1:8000'

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, detail: unknown) {
    super(
      typeof detail === 'string'
        ? detail
        : `Request failed with status ${status}.`,
    )
    this.status = status
    this.detail = detail
  }
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)

  if (!response.ok) {
    const data = await response.json().catch(() => null)
    throw new ApiError(response.status, data?.detail ?? response.statusText)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export function apiErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (typeof error.detail === 'string') return error.detail
    if (error.detail && typeof error.detail === 'object' && 'message' in error.detail) {
      return String((error.detail as { message: unknown }).message)
    }
  }

  if (error instanceof Error) {
    return error.message.includes('fetch')
      ? 'Could not connect to the backend. Make sure the FastAPI server is running on port 8000.'
      : error.message
  }

  return 'An unexpected error occurred.'
}

export const jsonInit = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})
