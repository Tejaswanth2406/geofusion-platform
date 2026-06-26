/**
 * GeoFusion AI — Explainability Panel
 * Shows why a result was matched: confidence gauge, sensor flow, reasons list
 */

function ConfidenceGauge({ pct }) {
  const radius = 30
  const circ   = 2 * Math.PI * radius
  const offset = circ - (pct / 100) * circ

  const color = pct >= 80
    ? 'var(--accent-emerald)'
    : pct >= 60
    ? 'var(--accent-cyan)'
    : pct >= 40
    ? 'var(--accent-amber)'
    : 'var(--accent-rose)'

  return (
    <div className="confidence-gauge" aria-label={`Confidence: ${pct}%`}>
      <svg viewBox="0 0 80 80" width="80" height="80">
        {/* Track */}
        <circle cx="40" cy="40" r={radius} fill="none" stroke="var(--border-subtle)" strokeWidth="6" />
        {/* Fill */}
        <circle
          cx="40" cy="40" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.16,1,0.3,1)', filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div className="confidence-text">
        <span>{pct}%</span>
        <span className="confidence-label">conf.</span>
      </div>
    </div>
  )
}

export default function ExplainPanel({ result, explainability, queryInfo }) {
  if (!result && !explainability) {
    return (
      <div className="empty-state">
        <span className="empty-icon">🔬</span>
        <p className="empty-title">Select a result</p>
        <p className="empty-subtitle">Click any result card to see explainability details</p>
      </div>
    )
  }

  const explain = explainability || {}
  const pct     = Math.round(explain.confidence_pct ?? (result?.similarity ?? 0) * 100)
  const reasons = explain.matched_because || []
  const querySensor  = explain.query_sensor  || queryInfo?.sensor  || '—'
  const targetSensor = explain.target_sensor || result?.sensor     || '—'
  const distance     = typeof explain.embedding_distance === 'number'
    ? explain.embedding_distance.toFixed(4)
    : '—'

  return (
    <div className="explain-panel scale-in" id="explain-panel">

      {/* Header with gauge */}
      <div className="explain-header">
        <div>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            Match Analysis
          </h3>
          {result && (
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {result.tile_id || result.id || 'Unknown tile'}
            </p>
          )}
        </div>
        <ConfidenceGauge pct={pct} />
      </div>

      {/* Sensor cross-modal flow */}
      <div className="sensor-flow" id="sensor-flow">
        <div className="sensor-flow-badge">
          {querySensor === 'optical' ? '🌍' : querySensor === 'sar' ? '📡' : '🌈'}
          &nbsp;{querySensor}
        </div>
        <span className="sensor-flow-arrow">→</span>
        <div className="sensor-flow-badge">
          {targetSensor === 'optical' ? '🌍' : targetSensor === 'sar' ? '📡' : '🌈'}
          &nbsp;{targetSensor}
        </div>
        <div style={{ marginLeft: 'auto', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          cross-modal
        </div>
      </div>

      {/* Embedding distance */}
      <div style={{ display: 'flex', gap: '0.75rem' }}>
        <div style={{ flex: 1, padding: '0.6rem 0.8rem', background: 'var(--bg-glass)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', color: 'var(--accent-cyan)', fontWeight: 700 }}>{distance}</div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Embed Dist</div>
        </div>
        <div style={{ flex: 1, padding: '0.6rem 0.8rem', background: 'var(--bg-glass)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', color: 'var(--accent-violet)', fontWeight: 700 }}>{pct}%</div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Similarity</div>
        </div>
      </div>

      {/* Match reasons */}
      {reasons.length > 0 && (
        <div>
          <p style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>
            Matched Because
          </p>
          <div className="explain-reasons">
            {reasons.map((reason, i) => (
              <div
                key={i}
                className="explain-reason"
                style={{ animationDelay: `${i * 0.06}s` }}
              >
                <span className="explain-reason-dot" />
                <span>{reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metadata extras */}
      {result?.metadata && (
        <div style={{ padding: '0.75rem', background: 'var(--bg-glass)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {result.metadata.location && (
            <div style={{ marginBottom: '0.25rem' }}>📍 {result.metadata.location}</div>
          )}
          {result.metadata.date && (
            <div style={{ marginBottom: '0.25rem' }}>📅 {result.metadata.date}</div>
          )}
          {result.metadata.cloud_cover !== undefined && (
            <div>☁️ Cloud cover: {result.metadata.cloud_cover}%</div>
          )}
        </div>
      )}
    </div>
  )
}
