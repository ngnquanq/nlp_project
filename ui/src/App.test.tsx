import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const readyHealth = {
  status: 'ready',
  model_id: 'e1_fairseq_vi_zh_v1',
  device: 'cpu',
  parameter_count: 67652608,
  message: null,
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('E1 MT interface', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(readyHealth)))
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('shows model readiness and keeps translation disabled for blank input', async () => {
    render(<App />)
    expect(await screen.findByText('E1 sẵn sàng')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Dịch sang Hán văn/i })).toBeDisabled()
  })

  it('translates, reports metadata, OOV tokens, and copies the result', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock
      .mockResolvedValueOnce(jsonResponse(readyHealth))
      .mockResolvedValueOnce(
        jsonResponse({
          translation: '王遣使如北',
          normalized_input: 'Vua sai sứ sang phương Bắc',
          source_token_count: 6,
          target_token_count: 6,
          unknown_tokens: ['phương'],
          latency_ms: 184,
          model_id: 'e1_fairseq_vi_zh_v1',
        }),
    )

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
      .mockResolvedValueOnce(jsonResponse(readyHealth))
      .mockResolvedValueOnce(
        jsonResponse({
          translation: '王', normalized_input: 'Vua', source_token_count: 1,
          target_token_count: 1, unknown_tokens: [], latency_ms: 10,
          model_id: 'e1_fairseq_vi_zh_v1',
        }),
      )
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('E1 sẵn sàng')
    const source = screen.getByLabelText('Văn bản tiếng Việt cần dịch')
    await user.type(source, 'Vua')
    await user.keyboard('{Control>}{Enter}{/Control}')
    expect(await screen.findByText('王')).toBeInTheDocument()
  })

  it('explains unavailable private artifacts', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({
        ...readyHealth,
        status: 'model_unavailable',
        parameter_count: null,
        message: 'Set MT_CHECKPOINT_PATH and MT_DATA_BIN_PATH before translating.',
      }),
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
})
