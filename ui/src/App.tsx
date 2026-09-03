import { FormEvent, KeyboardEvent, useEffect, useId, useRef, useState } from 'react'
import {
  ApiError,
  getHealth,
  translateText,
  type HealthResponse,
  type ModelHealth,
  type ModelKey,
  type TranslationResponse,
} from './api'

const MODEL_STORAGE_KEY = 'mt.model'
const SLOW_NOTICE_AFTER_SECONDS = 10

type ModelCopy = {
  guidance: string
  footerNote: string
}

const MODEL_COPY: Record<ModelKey, ModelCopy> = {
  e1: {
    guidance: 'E1 phù hợp nhất với văn bản lịch sử đã tách token và ít dấu câu.',
    footerNote: 'Fairseq Transformer · beam 7 · local only',
  },
  e2: {
    guidance:
      'E2 dùng cùng prompt với pipeline offline; văn bản lịch sử ít dấu câu vẫn cho kết quả tốt nhất.',
    footerNote: 'Qwen3-8B QLoRA 4-bit · beam 4 · local only',
  },
}

function isModelKey(value: unknown): value is ModelKey {
  return value === 'e1' || value === 'e2'
}

function readStoredModel(): ModelKey {
  try {
    const stored = window.localStorage.getItem(MODEL_STORAGE_KEY)
    if (isModelKey(stored)) return stored
  } catch {
    // Private windows and blocked site data both land here; the default is fine.
  }
  return 'e1'
}

function normalizeForCount(value: string): string {
  return value.normalize('NFC').replace(/\s+/g, ' ').trim()
}

function encodedPositions(value: string): number {
  const normalized = normalizeForCount(value)
  return normalized ? normalized.split(' ').length + 1 : 0
}

/** Units the browser can show for a model it cannot tokenize itself. */
function estimatedUnits(value: string, charsPerUnit: number | null): number {
  const normalized = normalizeForCount(value)
  if (!normalized) return 0
  return Math.ceil(normalized.length / (charsPerUnit && charsPerUnit > 0 ? charsPerUnit : 3.39))
}

function formatParameters(model: ModelHealth | null): string {
  if (!model) return 'Đang kiểm tra model'
  return model.parameter_count
    ? `${(model.parameter_count / 1_000_000).toFixed(1)}M tham số`
    : model.sublabel
}

function statusLabel(model: ModelHealth | null, health: HealthResponse | null): string {
  if (!health || !model) return 'Đang kiểm tra model'
  if (model.status === 'ready') return `${model.label} sẵn sàng`
  if (model.status === 'not_loaded') return `${model.label} chờ nạp`
  return 'Thiếu model'
}

function StatusMark({
  health,
  model,
}: {
  health: HealthResponse | null
  model: ModelHealth | null
}) {
  const ready = model?.status === 'ready'
  return (
    <div className={`model-status ${ready ? 'is-ready' : ''}`} role="status">
      <span className="status-dot" aria-hidden="true" />
      <span>{statusLabel(model, health)}</span>
    </div>
  )
}

export default function App() {
  const textareaId = useId()
  const resultHeadingId = useId()
  const modelLabelId = useId()
  const [text, setText] = useState('')
  const [model, setModel] = useState<ModelKey>(readStoredModel)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [result, setResult] = useState<TranslationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [copied, setCopied] = useState(false)
  const resultRef = useRef<HTMLElement>(null)
  const requestRef = useRef<AbortController | null>(null)

  const models = health?.models ?? []
  const active = models.find((entry) => entry.key === model) ?? null
  const limits = active?.limits ?? null

  const characterCount = normalizeForCount(text).length
  const units = limits?.client_estimate
    ? estimatedUnits(text, limits.chars_per_unit)
    : encodedPositions(text)

  const overCharacters = limits ? characterCount > limits.max_characters : false
  const overUnits = limits
    ? limits.client_estimate
      ? // The browser cannot run Qwen BPE, so it only blocks well past the
        // limit and lets the server issue the authoritative rejection.
        units > limits.max_units * 1.25
      : units > limits.max_units
    : false
  const inputTooLong = overCharacters || overUnits

  // not_loaded must stay submittable, otherwise lazy loading never triggers.
  const modelUsable = active !== null && active.status !== 'unavailable'

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then((response) => {
        setHealth(response)
        const stored = response.models.find((entry) => entry.key === model)
        // A stale stored choice would otherwise lock the user out.
        if (!stored || stored.status === 'unavailable') {
          const fallback = response.models.find(
            (entry) => entry.key === response.default_model && entry.status !== 'unavailable',
          )
          if (fallback) setModel(fallback.key)
        }
        if (response.status === 'model_unavailable') {
          const first = response.models[0]
          setHealthError(first?.message ?? 'E1 chưa được nạp.')
        }
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setHealthError('Không kết nối được với API tại máy này.')
      })
    return () => controller.abort()
    // Runs once: selecting a model must never re-probe the API.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_STORAGE_KEY, model)
    } catch {
      // Persisting the choice is a convenience, never a requirement.
    }
  }, [model])

  useEffect(() => {
    if (!loading) {
      setElapsed(0)
      return
    }
    const started = Date.now()
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - started) / 1000))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [loading])

  useEffect(() => () => requestRef.current?.abort(), [])

  async function submit() {
    if (!modelUsable || !text.trim() || inputTooLong || loading) return
    requestRef.current?.abort()
    const controller = new AbortController()
    requestRef.current = controller
    setLoading(true)
    setError(null)
    setCopied(false)
    try {
      const response = await translateText(text, model, controller.signal)
      setResult(response)
      requestAnimationFrame(() => resultRef.current?.focus())
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') return
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

  function selectModel(entry: ModelHealth) {
    if (entry.status === 'unavailable' || entry.key === model) return
    requestRef.current?.abort()
    setModel(entry.key)
    setResult(null)
    setError(null)
    setCopied(false)
  }

  function handleSegmentKeys(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
    event.preventDefault()
    const index = models.findIndex((entry) => entry.key === model)
    if (index < 0 || models.length < 2) return
    const step = event.key === 'ArrowRight' ? 1 : -1
    const next = models[(index + step + models.length) % models.length]
    selectModel(next)
  }

  async function copyTranslation() {
    if (!result) return
    await navigator.clipboard.writeText(result.translation)
    setCopied(true)
  }

  function modelNote(): string {
    if (!active) return 'Đang lấy trạng thái mô hình…'
    if (active.status === 'unavailable') {
      return active.message ?? `${active.label} chưa sẵn sàng.`
    }
    if (active.status === 'not_loaded') {
      return `${active.label} sẽ được nạp ở lần dịch đầu tiên.`
    }
    return `${active.label} · ${active.sublabel} · ${active.device}`
  }

  function overLimitMessage(): string {
    if (!limits) return 'Rút gọn văn bản.'
    if (overCharacters) {
      return `Rút gọn văn bản để không vượt quá ${limits.max_characters.toLocaleString('vi-VN')} ký tự.`
    }
    return limits.unit === 'position'
      ? `Rút gọn văn bản để không vượt quá ${limits.max_units} vị trí mã hoá.`
      : `Rút gọn văn bản để không vượt quá ${limits.max_units} token prompt.`
  }

  const copy = MODEL_COPY[model]
  const showSlowNotice =
    loading && (active?.slow_first_request ?? false) && elapsed >= SLOW_NOTICE_AFTER_SECONDS

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="brand-lockup" aria-label="Việt sang Hán">
          <span className="brand-viet">VIỆT</span>
          <span className="brand-line" aria-hidden="true" />
          <span className="brand-han" lang="zh">漢</span>
        </div>
        <StatusMark health={health} model={active} />
      </header>

      <main>
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow">Dịch máy · Thí nghiệm E1 &amp; E2</p>
          <h1 id="page-title">
            Một bàn dịch cho
            <span> văn bản sử Việt.</span>
          </h1>
          <p className="lede">
            Nhập một đoạn tiếng Việt rồi chọn mô hình để chuyển sang Hán văn.
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
                {limits?.client_estimate ? '~' : ''}
                {units}/{limits?.max_units ?? 0} {limits?.unit_label ?? 'vị trí'}
              </span>
            </div>

            <div className="model-switch" role="radiogroup" aria-labelledby={modelLabelId}>
              <p className="panel-kicker" id={modelLabelId}>Mô hình</p>
              <div className="segmented">
                {models.map((entry) => (
                  <button
                    key={entry.key}
                    type="button"
                    role="radio"
                    aria-checked={entry.key === model}
                    aria-disabled={entry.status === 'unavailable'}
                    tabIndex={entry.key === model ? 0 : -1}
                    className={
                      'segment' +
                      (entry.key === model ? ' is-active' : '') +
                      (entry.status === 'unavailable' ? ' is-unavailable' : '')
                    }
                    onClick={() => selectModel(entry)}
                    onKeyDown={handleSegmentKeys}
                  >
                    <span className="segment-label">{entry.label}</span>
                    <span className="segment-sub">{entry.sublabel}</span>
                  </button>
                ))}
              </div>
              <p className="model-note" role="status">{modelNote()}</p>
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
              maxLength={(limits?.max_characters ?? 4000) + 1}
              aria-describedby="source-guidance source-count"
              rows={11}
            />

            <div className="source-footer">
              <p id="source-guidance">{copy.guidance}</p>
              <span id="source-count">{characterCount.toLocaleString('vi-VN')} ký tự</span>
            </div>

            {inputTooLong && (
              <p className="inline-error" role="alert">{overLimitMessage()}</p>
            )}

            <button
              className="translate-button"
              type="submit"
              disabled={!modelUsable || !text.trim() || inputTooLong || loading}
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
                  <p>{active?.label ?? 'Model'} đang dựng bản dịch…</p>
                  {showSlowNotice && (
                    <p className="loading-note">
                      Lần dịch đầu phải nạp mô hình 8B 4-bit nên chậm hơn; các lần
                      sau sẽ nhanh. Đã chờ {elapsed}s.
                    </p>
                  )}
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
                <span>{formatParameters(active)}</span>
              </div>
            )}

            {result && active?.reports_unknown_tokens && result.unknown_tokens.length > 0 && (
              <div className="oov-warning" role="status">
                <strong>Token ngoài từ điển:</strong>{' '}
                <span>{result.unknown_tokens.join(', ')}</span>
              </div>
            )}
          </section>
        </form>
      </main>

      <footer>
        <span>{active?.model_id ?? 'e1_fairseq_vi_zh_v1'}</span>
        <span>{copy.footerNote}</span>
      </footer>
    </div>
  )
}
