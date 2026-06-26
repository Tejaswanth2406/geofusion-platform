/**
 * GeoFusion AI — Service Status Bar
 * Polls GET /health every 30s and shows service status dots
 */
import { useState, useEffect } from 'react'

const SERVICES = [
  { key: 'embedding',  label: 'Embedding Service',  port: 8001, icon: '🧠' },
  { key: 'retrieval',  label: 'Retrieval (FAISS)',   port: 8002, icon: '🔍' },
  { key: 'evaluation', label: 'Evaluation Service',  port: 8005, icon: '📊' },
]

export default function StatusBar({ token }) {
  const [services, setServices] = useState({})
  const [overall, setOverall]   = useState('unknown')
  const [lastCheck, setLastCheck] = useState(null)

  const checkHealth = async () => {
    try {
      const res = await fetch('/health', {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
      if (res.ok) {
        const data = await res.json()
        setServices(data.services || {})
        setOverall(data.status || 'unknown')
        setLastCheck(new Date())
      } else {
        setOverall('down')
      }
    } catch {
      setOverall('down')
    }
  }

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30_000)
    return () => clearInterval(interval)
  }, [token]) // eslint-disable-line react-hooks/exhaustive-deps

  const statusLabel = {
    healthy:  'All Systems Up',
    degraded: 'Degraded',
    down:     'Services Down',
    unknown:  'Checking…',
  }

  return (
    <div className="card" style={{ padding: '1rem' }} id="status-panel">
      <div className="card-header" style={{ marginBottom: '0.75rem' }}>
        <div className="card-icon">💡</div>
        <span className="card-title">System Status</span>
        <span
          className="chip"
          style={{ marginLeft: 'auto', cursor: 'pointer' }}
          onClick={checkHealth}
          title="Click to refresh"
        >
          {overall === 'healthy'
            ? <><span className="chip-emerald" style={{ color: 'var(--accent-emerald)' }}>🟢</span></>
            : overall === 'down'
            ? <><span style={{ color: 'var(--accent-rose)' }}>🔴</span></>
            : <><span style={{ color: 'var(--accent-amber)' }}>🟡</span></>}
          &nbsp;{statusLabel[overall] || overall}
        </span>
      </div>

      <div className="status-bar">
        {SERVICES.map(({ key, label, icon }) => {
          const val = services[key] || 'unknown'
          return (
            <div key={key} className="status-item" id={`status-${key}`}>
              <span style={{ fontSize: '0.9rem' }}>{icon}</span>
              <span className="status-name">{label}</span>
              <span className={`status-dot ${val}`} />
              <span className={`status-value ${val}`} style={{ marginLeft: '8px' }}>{val}</span>
            </div>
          )
        })}
      </div>

      {lastCheck && (
        <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.6rem', textAlign: 'right' }}>
          Last checked {lastCheck.toLocaleTimeString()}
        </p>
      )}
    </div>
  )
}
