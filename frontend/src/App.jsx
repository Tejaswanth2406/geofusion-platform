/**
 * GeoFusion AI Platform — Main App
 * =====================================
 * Full-featured satellite intelligence dashboard:
 *  - JWT login / logout
 *  - Drag-and-drop image upload
 *  - Sensor + retrieval mode config
 *  - Top-K results grid with similarity scores
 *  - Explainability panel for selected result
 *  - Live service health status bar
 */

import { useState } from 'react'
import LoginPage   from './components/LoginPage.jsx'
import UploadPanel from './components/UploadPanel.jsx'
import ResultsGrid from './components/ResultsGrid.jsx'
import ExplainPanel from './components/ExplainPanel.jsx'
import StatusBar   from './components/StatusBar.jsx'

export default function App() {
  // Auth state
  const [token, setToken]       = useState(() => sessionStorage.getItem('gf_token') || '')
  const [role, setRole]         = useState(() => sessionStorage.getItem('gf_role')  || '')
  const [username, setUsername] = useState(() => sessionStorage.getItem('gf_user')  || '')

  // Search results state
  const [results, setResults]         = useState(null)
  const [explainability, setExplain]  = useState(null)
  const [queryInfo, setQueryInfo]     = useState(null)
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [searchMeta, setSearchMeta]   = useState(null) // { count, time_ms, index_size }
  const [error, setError]             = useState('')

  // ─── Auth handlers ────────────────────────────────────────────────────────
  const handleLogin = (tok, userRole, user) => {
    setToken(tok)
    setRole(userRole)
    setUsername(user)
    sessionStorage.setItem('gf_token', tok)
    sessionStorage.setItem('gf_role',  userRole)
    sessionStorage.setItem('gf_user',  user)
  }

  const handleLogout = () => {
    setToken('')
    setRole('')
    setUsername('')
    sessionStorage.clear()
    setResults(null)
    setExplain(null)
    setError('')
  }

  // ─── Results handler ──────────────────────────────────────────────────────
  const handleResults = (data, info) => {
    const items = data.results || []
    setResults(items)
    setExplain(data.explainability || null)
    setQueryInfo(info)
    setSelectedIdx(0)
    setSearchMeta({
      count:      items.length,
      time_ms:    data.retrieval_time_ms,
      index_size: data.index_size,
      request_id: data.request_id,
    })
    setError('')
  }

  // ─── Render login ─────────────────────────────────────────────────────────
  if (!token) {
    return <LoginPage onLogin={handleLogin} />
  }

  const selectedResult = results?.[selectedIdx] || null

  return (
    <div className="app-layout">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="app-header-brand">
          <div className="app-header-logo">🛰️</div>
          <div>
            <div className="app-header-title">GeoFusion AI</div>
            <div className="app-header-subtitle">Satellite Intelligence Platform</div>
          </div>
        </div>

        <div className="app-header-actions">
          {/* Role badge */}
          <span className={`chip ${role === 'admin' ? 'chip-cyan' : 'chip-violet'}`} id="role-badge">
            {role === 'admin' ? '👑' : '👤'} {role}
          </span>

          {/* Search results summary */}
          {searchMeta && (
            <span
              className="chip chip-emerald"
              style={{ display: 'none' }}
              id="search-meta-chip"
            >
              {searchMeta.count} results · {searchMeta.time_ms}ms
            </span>
          )}

          {/* User avatar + logout */}
          <div className="user-avatar" id="user-avatar" title={username}>
            {username.slice(0, 2).toUpperCase()}
          </div>
          <button
            id="logout-btn"
            className="btn btn-ghost btn-sm"
            onClick={handleLogout}
            title="Sign out"
          >
            ⏏ Sign Out
          </button>
        </div>
      </header>

      {/* ── Main ── */}
      <div className="app-main">

        {/* ── Left Sidebar ── */}
        <aside className="sidebar">
          {/* Upload + config panel */}
          <UploadPanel
            onResults={handleResults}
            onError={setError}
            token={token}
          />

          {/* System health */}
          <StatusBar token={token} />
        </aside>

        {/* ── Right main content ── */}
        <main className="main-content">

          {/* Error banner */}
          {error && (
            <div
              className="card"
              style={{ borderColor: 'rgba(244,63,94,0.4)', background: 'rgba(244,63,94,0.07)' }}
              role="alert"
              id="error-banner"
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '1.25rem' }}>⚠️</span>
                <div>
                  <p style={{ color: 'var(--accent-rose)', fontWeight: 600, fontSize: '0.875rem', margin: 0 }}>Retrieval Error</p>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: 0 }}>{error}</p>
                </div>
                <button
                  className="btn btn-ghost btn-sm"
                  style={{ marginLeft: 'auto' }}
                  onClick={() => setError('')}
                >✕</button>
              </div>
            </div>
          )}

          {/* Stats banner — shown after search */}
          {searchMeta && (
            <div className="stats-banner fade-in" id="stats-banner">
              <div className="stat-item">
                <div className="stat-value">{searchMeta.count}</div>
                <div className="stat-label">Results Found</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{searchMeta.time_ms != null ? `${searchMeta.time_ms}ms` : '—'}</div>
                <div className="stat-label">Retrieval Time</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{queryInfo?.sensor || '—'}</div>
                <div className="stat-label">Query Sensor</div>
              </div>
            </div>
          )}

          {/* Results grid + explainability split */}
          {results ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '1.25rem', alignItems: 'start' }}>
              {/* Results grid */}
              <div className="card" style={{ padding: '1.25rem' }}>
                <div className="card-header">
                  <div className="card-icon">🗂️</div>
                  <span className="card-title">
                    Top-{results.length} Results
                  </span>
                  <span
                    className="chip chip-emerald"
                    style={{ marginLeft: 'auto' }}
                    id="mode-chip"
                  >
                    {queryInfo?.mode === 'cross' ? '🔀 Cross-Modal' : '🔍 Same-Sensor'}
                  </span>
                </div>
                <ResultsGrid
                  results={results}
                  onSelectResult={setSelectedIdx}
                  selectedIndex={selectedIdx}
                />
              </div>

              {/* Explainability panel */}
              <div className="card" id="explain-card">
                <div className="card-header">
                  <div className="card-icon">🔬</div>
                  <span className="card-title">Explainability</span>
                </div>
                <ExplainPanel
                  result={selectedResult}
                  explainability={explainability}
                  queryInfo={queryInfo}
                />
              </div>
            </div>
          ) : (
            /* Welcome / empty state */
            <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }} id="welcome-card">
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '3rem 2rem', textAlign: 'center', gap: '1.25rem' }}>
                <div style={{
                  width: 80, height: 80,
                  background: 'var(--grad-primary)',
                  borderRadius: 'var(--radius-lg)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '2.25rem',
                  boxShadow: 'var(--shadow-glow)'
                }}>🛰️</div>

                <div>
                  <h2 style={{ marginBottom: '0.5rem' }}>Multi-Sensor Satellite Retrieval</h2>
                  <p style={{ maxWidth: 480, lineHeight: 1.7 }}>
                    Upload a satellite image to search across optical, SAR, and multispectral sensor modalities
                    using our FAISS-indexed shared embedding space.
                  </p>
                </div>

                {/* Feature highlights */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', width: '100%', maxWidth: 600, marginTop: '0.5rem' }}>
                  {[
                    { icon: '🧠', title: 'ViT Encoders',   desc: 'Vision Transformer + ResNet dual encoder' },
                    { icon: '⚡', title: 'FAISS Index',    desc: 'Sub-50ms ANN search over millions of tiles' },
                    { icon: '🔬', title: 'Explainability', desc: 'Match reasons + confidence scores' },
                  ].map(f => (
                    <div
                      key={f.title}
                      style={{
                        padding: '1rem',
                        background: 'var(--bg-glass)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-md)',
                        textAlign: 'center',
                      }}
                    >
                      <div style={{ fontSize: '1.5rem', marginBottom: '0.4rem' }}>{f.icon}</div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>{f.title}</div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{f.desc}</div>
                    </div>
                  ))}
                </div>

                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  Signed in as <strong style={{ color: 'var(--accent-cyan)' }}>{username}</strong> · Role: <strong style={{ color: 'var(--accent-violet)' }}>{role}</strong>
                </p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
