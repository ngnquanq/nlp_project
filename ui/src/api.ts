export type HealthResponse = {
  status: 'ready' | 'model_unavailable'
  model_id: string
  device: string
  parameter_count: number | null
  message: string | null
}

export type TranslationResponse = {
  translation: string
  normalized_input: string
  source_token_count: number
  target_token_count: number
  unknown_tokens: string[]
  latency_ms: number
  model_id: string
}

type ErrorResponse = {
  error?: {
    code?: string
    message?: string
    request_id?: string
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code = 'REQUEST_FAILED',
    readonly requestId?: string,
  ) {
    super(message)
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let body: ErrorResponse = {}
  try {
    body = (await response.json()) as ErrorResponse
  } catch {
    // Keep the stable fallback below for non-JSON failures.
  }
  return new ApiError(
    body.error?.message ?? 'Không thể kết nối với dịch vụ dịch.',
    body.error?.code,
    body.error?.request_id,
  )
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch('/api/health', { signal })
  if (!response.ok) throw await parseError(response)
  return response.json() as Promise<HealthResponse>
}

export async function translateText(
  text: string,
  signal?: AbortSignal,
): Promise<TranslationResponse> {
  const response = await fetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  })
  if (!response.ok) throw await parseError(response)
  return response.json() as Promise<TranslationResponse>
}
