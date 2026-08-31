import { FormEvent, KeyboardEvent, useEffect, useId, useRef, useState } from 'react'
import {
  ApiError,
  getHealth,
  translateText,
  type HealthResponse,
  type TranslationResponse,
} from './api'

const MAX_CHARACTERS = 4_000
const MAX_ENCODED_POSITIONS = 256

function normalizeForCount(value: string): string {
  return value.normalize('NFC').replace(/\s+/g, ' ').trim()
}

function encodedPositions(value: string): number {
  const normalized = normalizeForCount(value)
  return normalized ? normalized.split(' ').length + 1 : 0
}

function formatParameters(value: number | null): string {
  return value ? `${(value / 1_000_000).toFixed(1)}M tham số` : 'E1 Fairseq'
}

function StatusMark({ health }: { health: HealthResponse | null }) {
  const ready = health?.status === 'ready'
  const label = !health ? 'Đang kiểm tra model' : ready ? 'E1 sẵn sàng' : 'Thiếu model'
  return (
    <div className={`model-status ${ready ? 'is-ready' : ''}`} role="status">
      <span className="status-dot" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export default function App() {
  const textareaId = useId()
  const resultHeadingId = useId()
  const [text, setText] = useState('')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [result, setResult] = useState<TranslationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const resultRef = useRef<HTMLElement>(null)

  const positions = encodedPositions(text)
  const characterCount = normalizeForCount(text).length
  const inputTooLong =
    positions > MAX_ENCODED_POSITIONS || characterCount > MAX_CHARACTERS
  const modelReady = health?.status === 'ready'

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then((response) => {
        setHealth(response)
        if (response.status === 'model_unavailable') {
          setHealthError(response.message ?? 'E1 chưa được nạp.')
        }
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setHealthError('Không kết nối được với API tại máy này.')
      })
    return () => controller.abort()
  }, [])

  async function submit() {
    if (!modelReady || !text.trim() || inputTooLong || loading) return
    setLoading(true)
    setError(null)
    setCopied(false)
    try {
      const response = await translateText(text)
      setResult(response)
      requestAnimationFrame(() => resultRef.current?.focus())
    } catch (reason) {
      setResult(null)
      setError(
        reason instanceof ApiError
          ? reason.message
          : 'Không thể kết nối với dịch vụ dịch.',
      )
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    void submit()
  }

  function handleKeyboard(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      void submit()
    }
  }

  function handleTextChange(value: string) {
    setText(value)
    setResult(null)
    setError(null)
    setCopied(false)
  }

  async function copyTranslation() {
    if (!result) return
    await navigator.clipboard.writeText(result.translation)
    setCopied(true)
  }

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="brand-lockup" aria-label="Việt sang Hán">
          <span className="brand-viet">VIỆT</span>
          <span className="brand-line" aria-hidden="true" />
          <span className="brand-han" lang="zh">漢</span>
        </div>
        <StatusMark health={health} />
      </header>

      <main>
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow">Dịch máy · Thí nghiệm E1</p>
          <h1 id="page-title">
            Một bàn dịch cho
            <span> văn bản sử Việt.</span>
          </h1>
          <p className="lede">
            Nhập một đoạn tiếng Việt để mô hình Transformer chuyển sang Hán văn.
            Kết quả là dự đoán của mô hình, không phải bản hiệu đính chuyên gia.
          </p>
        </section>

        {healthError && (
          <aside className="setup-notice" role="alert">
            <span className="notice-label">Model chưa sẵn sàng</span>
            <p>{healthError}</p>
            <code>MT_CHECKPOINT_PATH · MT_DATA_BIN_PATH</code>
          </aside>
        )}

        <form className="translation-desk" onSubmit={handleSubmit}>
          <section className="source-panel" aria-labelledby="source-heading">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Nguồn · VI</p>
                <h2 id="source-heading">Nguyên văn</h2>
              </div>
              <span className={inputTooLong ? 'counter is-over' : 'counter'}>
                {positions}/{MAX_ENCODED_POSITIONS} vị trí
              </span>
            </div>

            <label className="sr-only" htmlFor={textareaId}>
              Văn bản tiếng Việt cần dịch
            </label>
            <textarea
              id={textareaId}
              value={text}
              onChange={(event) => handleTextChange(event.target.value)}
              onKeyDown={handleKeyboard}
              placeholder="Nhập câu hoặc đoạn văn tiếng Việt…"
              maxLength={MAX_CHARACTERS + 1}
              aria-describedby="source-guidance source-count"
              rows={11}
            />

            <div className="source-footer">
              <p id="source-guidance">
                E1 phù hợp nhất với văn bản lịch sử đã tách token và ít dấu câu.
              </p>
              <span id="source-count">{characterCount.toLocaleString('vi-VN')} ký tự</span>
            </div>

            {inputTooLong && (
              <p className="inline-error" role="alert">
                Rút gọn văn bản để không vượt quá 256 vị trí mã hoá.
              </p>
            )}

            <button
              className="translate-button"
              type="submit"
              disabled={!modelReady || !text.trim() || inputTooLong || loading}
            >
              <span>{loading ? 'Đang dịch…' : 'Dịch sang Hán văn'}</span>
              <span className="shortcut" aria-hidden="true">⌘ ↵</span>
            </button>
          </section>

          <section
            className={`result-panel ${result ? 'has-result' : ''}`}
            aria-labelledby={resultHeadingId}
            aria-busy={loading}
            aria-live="polite"
            ref={resultRef}
            tabIndex={-1}
          >
            <div className="panel-heading result-heading">
              <div>
                <p className="panel-kicker">Đích · ZH</p>
                <h2 id={resultHeadingId}>Hán văn dự đoán</h2>
              </div>
              {result && (
                <button className="copy-button" type="button" onClick={copyTranslation}>
                  {copied ? 'Đã sao chép' : 'Sao chép'}
                </button>
              )}
            </div>

            <div className="folio">
              <span className="folio-rule rule-one" aria-hidden="true" />
              <span className="folio-rule rule-two" aria-hidden="true" />
              <span className="folio-rule rule-three" aria-hidden="true" />
              {loading ? (
                <div className="loading-state" role="status">
                  <span className="loading-mark" aria-hidden="true">譯</span>
                  <p>E1 đang dựng bản dịch…</p>
                </div>
              ) : result ? (
                <div className="translation-content">
                  <p className="han-text" lang="zh">{result.translation}</p>
                  <span className="result-seal" aria-label="Bản dịch máy">譯</span>
                </div>
              ) : error ? (
                <div className="empty-state error-state" role="alert">
                  <p className="empty-title">Chưa thể tạo bản dịch</p>
                  <p>{error}</p>
                </div>
              ) : (
                <div className="empty-state">
                  <p className="empty-title">Bản dịch sẽ xuất hiện tại đây</p>
                  <p>Dùng nút Dịch hoặc nhấn Ctrl/Cmd + Enter.</p>
                </div>
              )}
            </div>

            {result && (
              <div className="result-meta">
                <span>{result.latency_ms.toLocaleString('vi-VN')} ms</span>
                <span>{result.source_token_count} token nguồn</span>
                <span>{result.target_token_count} ký tự đích</span>
                <span>{formatParameters(health?.parameter_count ?? null)}</span>
              </div>
            )}

            {result && result.unknown_tokens.length > 0 && (
              <div className="oov-warning" role="status">
                <strong>Token ngoài từ điển:</strong>{' '}
                <span>{result.unknown_tokens.join(', ')}</span>
              </div>
            )}
          </section>
        </form>
      </main>

      <footer>
        <span>{health?.model_id ?? 'e1_fairseq_vi_zh_v1'}</span>
        <span>Fairseq Transformer · beam 7 · local only</span>
      </footer>
    </div>
  )
}
