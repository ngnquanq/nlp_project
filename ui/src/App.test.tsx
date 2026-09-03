import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { HealthResponse, ModelHealth, ModelStatus } from './api'

const E1_LIMITS = {
  max_characters: 4000,
  max_units: 256,
  unit: 'position' as const,
  unit_label: 'vị trí',
  client_estimate: false,
  chars_per_unit: null,
}

const E2_LIMITS = {
  max_characters: 4000,
  max_units: 512,
  unit: 'token' as const,
  unit_label: 'token',
  client_estimate: true,
  chars_per_unit: 3.39,
}

function e1(status: ModelStatus = 'ready', message: string | null = null): ModelHealth {
  return {
    key: 'e1',
    model_id: 'e1_fairseq_vi_zh_v1',
    label: 'E1',
    sublabel: 'Fairseq Transformer',
    status,
    device: 'cpu',
    parameter_count: status === 'ready' ? 67652608 : null,
    message,
    limits: E1_LIMITS,
    reports_unknown_tokens: true,
    slow_first_request: false,
  }
}

function e2(status: ModelStatus = 'not_loaded', message: string | null = null): ModelHealth {
  return {
    key: 'e2',
    model_id: 'e2_qwen3_8b_qlora_vi_zh_v1',
    label: 'E2',
    sublabel: 'Qwen3-8B QLoRA 4-bit',
    status,
    device: 'cuda',
    parameter_count: null,
    message,
    limits: E2_LIMITS,
    reports_unknown_tokens: false,
    slow_first_request: true,
  }
}

function health(...models: ModelHealth[]): HealthResponse {
  const usable = models.filter((model) => model.status !== 'unavailable')
  return {
    status:
      usable.length === 0
        ? 'model_unavailable'
        : usable.length === models.length
          ? 'ready'
          : 'degraded',
    default_model: 'e1',
    models,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const E1_TRANSLATION = {
  translation: '王遣使如北',
  normalized_input: 'Vua sai sứ sang phương Bắc',
  source_token_count: 6,
  target_token_count: 6,
  unknown_tokens: ['phương'],
  latency_ms: 184,
  model_id: 'e1_fairseq_vi_zh_v1',
}

const E2_TRANSLATION = {
  translation: '帝遣使如宋',
  normalized_input: 'Vua sai sứ sang phương Bắc',
  source_token_count: 12,
  target_token_count: 5,
  unknown_tokens: [],
  latency_ms: 4200,
  model_id: 'e2_qwen3_8b_qlora_vi_zh_v1',
}

describe('E1/E2 MT interface', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(health(e1(), e2()))))
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it('shows model readiness and keeps translation disabled for blank input', async () => {
    render(<App />)
    expect(await screen.findByText('E1 sẵn sàng')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Dịch sang Hán văn/i })).toBeDisabled()
  })

  it('translates, reports metadata, OOV tokens, and copies the result', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock
      .mockResolvedValueOnce(jsonResponse(health(e1(), e2())))
      .mockResolvedValueOnce(jsonResponse(E1_TRANSLATION))

    const user = userEvent.setup()
    const writeText = vi.spyOn(navigator.clipboard, 'writeText')
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    await user.type(
      screen.getByLabelText('Văn bản tiếng Việt cần dịch'),
      'Vua sai sứ sang phương Bắc',
    )
    await user.click(screen.getByRole('button', { name: /Dịch sang Hán văn/i }))

    expect(await screen.findByText('王遣使如北')).toBeInTheDocument()
    expect(screen.getByText(/Token ngoài từ điển/)).toBeInTheDocument()
    expect(screen.getByText('184 ms')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Sao chép' }))
    expect(writeText).toHaveBeenCalledWith('王遣使如北')
    expect(screen.getByRole('button', { name: 'Đã sao chép' })).toBeInTheDocument()
  })

  it('submits with Ctrl+Enter', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock
      .mockResolvedValueOnce(jsonResponse(health(e1(), e2())))
      .mockResolvedValueOnce(
        jsonResponse({ ...E1_TRANSLATION, translation: '王', unknown_tokens: [] }),
      )
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    await user.type(screen.getByLabelText('Văn bản tiếng Việt cần dịch'), 'Vua')
    await user.keyboard('{Control>}{Enter}{/Control}')
    expect(await screen.findByText('王')).toBeInTheDocument()
  })

  it('explains unavailable private artifacts', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(
        health(
          e1('unavailable', 'Set MT_CHECKPOINT_PATH and MT_DATA_BIN_PATH before translating.'),
        ),
      ),
    )
    render(<App />)
    expect(await screen.findByText('Thiếu model')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('MT_CHECKPOINT_PATH')
  })

  it('blocks text beyond the E1 encoded-position limit', async () => {
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    await user.type(
      screen.getByLabelText('Văn bản tiếng Việt cần dịch'),
      Array.from({ length: 256 }, () => 'từ').join(' '),
    )
    expect(screen.getByRole('alert')).toHaveTextContent('256 vị trí')
    expect(screen.getByRole('button', { name: /Dịch sang Hán văn/i })).toBeDisabled()
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
  })

  it('offers both models with E1 selected by default', async () => {
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    const options = screen.getAllByRole('radio')
    expect(options).toHaveLength(2)
    expect(options[0]).toHaveAttribute('aria-checked', 'true')
    expect(options[1]).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByText('Qwen3-8B QLoRA 4-bit')).toBeInTheDocument()
  })

  it('sends the selected model and never re-probes health on selection', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock
      .mockResolvedValueOnce(jsonResponse(health(e1(), e2())))
      .mockResolvedValueOnce(jsonResponse(E2_TRANSLATION))

    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    await user.click(screen.getByRole('radio', { name: /E2/ }))
    await screen.findByText('E2 chờ nạp')

    await user.type(screen.getByLabelText('Văn bản tiếng Việt cần dịch'), 'Vua sai sứ')
    await user.click(screen.getByRole('button', { name: /Dịch sang Hán văn/i }))

    expect(await screen.findByText('帝遣使如宋')).toBeInTheDocument()
    const body = JSON.parse(String(fetchMock.mock.calls[1][1]?.body))
    expect(body).toEqual({ text: 'Vua sai sứ', model: 'e2' })
    // Health on mount + one translate. Selecting a model must not fetch.
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('switches the counter to the estimated E2 token budget', async () => {
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    await user.type(screen.getByLabelText('Văn bản tiếng Việt cần dịch'), 'Vua sai sứ')

    expect(screen.getByText('4/256 vị trí')).toBeInTheDocument()
    await user.click(screen.getByRole('radio', { name: /E2/ }))
    // 10 characters / 3.39 -> 3, and the tilde marks it as an estimate.
    expect(screen.getByText('~3/512 token')).toBeInTheDocument()
  })

  it('keeps a not_loaded model submittable so lazy loading can trigger', async () => {
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    await user.click(screen.getByRole('radio', { name: /E2/ }))
    await user.type(screen.getByLabelText('Văn bản tiếng Việt cần dịch'), 'Vua sai sứ')
    expect(screen.getByRole('button', { name: /Dịch sang Hán văn/i })).toBeEnabled()
  })

  it('marks an unavailable model and refuses to select it', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(health(e1(), e2('unavailable', 'Chạy `make ui-api-e2`.'))),
    )
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')

    const e2Option = screen.getByRole('radio', { name: /E2/ })
    expect(e2Option).toHaveAttribute('aria-disabled', 'true')
    await user.click(e2Option)

    expect(e2Option).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByText('E1 sẵn sàng')).toBeInTheDocument()
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
  })

  it('restores the stored model and falls back when it is unavailable', async () => {
    window.localStorage.setItem('mt.model', 'e2')
    render(<App />)
    expect(await screen.findByText('E2 chờ nạp')).toBeInTheDocument()
    cleanup()

    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(health(e1(), e2('unavailable', 'Sidecar chưa chạy.'))),
    )
    render(<App />)
    expect(await screen.findByText('E1 sẵn sàng')).toBeInTheDocument()
  })

  it('moves the selection with ArrowRight', async () => {
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    screen.getByRole('radio', { name: /E1/ }).focus()
    await user.keyboard('{ArrowRight}')
    expect(await screen.findByText('E2 chờ nạp')).toBeInTheDocument()
  })
})
