/**
 * GeoFusion AI — Results Grid Component
 * Renders Top-K retrieval results as satellite tile cards
 */

// Deterministic pseudo-random color palette for each tile
const TILE_PALETTES = [
  ['#0d1b2a', '#1b4332', '#0a3622'],
  ['#1a1a2e', '#16213e', '#0f3460'],
  ['#2d1b69', '#1a1a2e', '#11002f'],
  ['#1c1c1c', '#2d2d2d', '#1a1a1a'],
  ['#0b3d0b', '#1a5c1a', '#0d2b0d'],
  ['#1a0533', '#2d0a5e', '#1a0533'],
  ['#051923', '#003554', '#006494'],
  ['#240046', '#3c096c', '#5a189a'],
]

const SENSOR_ICONS = {
  optical:       '🌍',
  sar:           '📡',
  multispectral: '🌈',
  dem:           '⛰️',
}

const SENSOR_BADGE_CLASS = {
  optical:       'badge-optical',
  sar:           'badge-sar',
  multispectral: 'badge-multi',
  dem:           'badge-dem',
}

function SatTile({ index, sensor }) {
  const palette = TILE_PALETTES[index % TILE_PALETTES.length]
  const icon = SENSOR_ICONS[sensor] || '🛰️'

  // Generate a simple pseudo-random SVG pattern
  const seed = index * 17 + 5
  const stripes = Array.from({ length: 6 }, (_, i) => ({
    y: (i * 16 + (seed % 8)) % 100,
    opacity: 0.1 + (((seed + i) % 5) * 0.06),
    height: 4 + (i % 4),
  }))

  return (
    <div
      className="sat-tile"
      style={{ background: `linear-gradient(135deg, ${palette[0]} 0%, ${palette[1]} 50%, ${palette[2]} 100%)` }}
      aria-hidden="true"
    >
      <svg
        width="100%"
        height="100%"
        style={{ position: 'absolute', inset: 0 }}
        preserveAspectRatio="none"
      >
        {stripes.map((s, i) => (
          <rect
            key={i}
            x="0"
            y={`${s.y}%`}
            width="100%"
            height={s.height}
            fill="white"
            opacity={s.opacity}
          />
        ))}
        {/* Grid overlay */}
        {Array.from({ length: 4 }, (_, i) => (
          <line
            key={`v${i}`}
            x1={`${(i + 1) * 20}%`} y1="0"
            x2={`${(i + 1) * 20}%`} y2="100%"
            stroke="white" strokeOpacity="0.05" strokeWidth="1"
          />
        ))}
      </svg>
      <span style={{ position: 'relative', zIndex: 1, fontSize: '1.75rem', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.5))' }}>
        {icon}
      </span>
    </div>
  )
}

function ResultCard({ result, index, selected, onClick }) {
  const sim = typeof result.similarity === 'number' ? result.similarity : result.score ?? 0
  const pct = Math.round(sim * 100)
  const sensor = result.sensor || result.metadata?.sensor || 'optical'
  const tileId = result.tile_id || result.id || `tile-${index}`

  return (
    <div
      className={`result-card ${selected ? 'selected' : ''}`}
      onClick={onClick}
      id={`result-card-${index}`}
      role="button"
      tabIndex={0}
      aria-label={`Result ${index + 1}: ${tileId}, similarity ${pct}%`}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      style={{ animationDelay: `${index * 0.04}s` }}
    >
      <div className="result-tile">
        <div className="result-rank">#{index + 1}</div>
        <div className={`result-sensor-badge ${SENSOR_BADGE_CLASS[sensor] || 'badge-optical'}`}>
          {sensor}
        </div>
        <SatTile index={index} sensor={sensor} />
      </div>

      <div className="result-meta">
        <div className="result-id">{tileId}</div>

        <div className="result-sim-row">
          <span className="result-sim-label">Similarity</span>
          <span className="result-sim-value">{pct}%</span>
        </div>
        <div className="sim-bar-track">
          <div
            className="sim-bar-fill"
            style={{ width: `${pct}%` }}
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>

        {result.metadata?.location && (
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            📍 {result.metadata.location}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ResultsGrid({ results, onSelectResult, selectedIndex }) {
  if (!results || results.length === 0) {
    return (
      <div className="empty-state">
        <span className="empty-icon">🛰️</span>
        <p className="empty-title">No results yet</p>
        <p className="empty-subtitle">Upload a satellite image and run retrieval</p>
      </div>
    )
  }

  return (
    <div className="results-grid fade-in" id="results-grid" role="list">
      {results.map((result, i) => (
        <ResultCard
          key={result.tile_id || result.id || i}
          result={result}
          index={i}
          selected={selectedIndex === i}
          onClick={() => onSelectResult(i)}
        />
      ))}
    </div>
  )
}
