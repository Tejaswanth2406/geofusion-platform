/**
 * GeoFusion AI — Login Page Component
 * JWT authentication against /auth/token
 */
import { useState } from 'react'

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await fetch('/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Invalid credentials')
      }
      const data = await res.json()
      onLogin(data.access_token, data.role || 'analyst', username)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fillDemo = (u, p) => { setUsername(u); setPassword(p); setError('') }

  return (
    <div className="login-page">
      <div className="login-box">
        {/* Logo */}
        <div className="login-logo">🛰️</div>
        <h1 className="login-title" style={{ fontSize: '1.75rem' }}>GeoFusion AI</h1>
        <p className="login-subtitle">Multi-Sensor Satellite Intelligence Platform</p>

        {/* Form */}
        <form className="login-form" onSubmit={handleSubmit} id="login-form">
          <div className="form-group">
            <label className="form-label" htmlFor="login-username">Username</label>
            <input
              id="login-username"
              className="form-input"
              type="text"
              placeholder="Enter username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="username"
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="login-password">Password</label>
            <input
              id="login-password"
              className="form-input"
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />
          </div>

          {error && <div className="login-error" role="alert">{error}</div>}

          <button
            id="login-submit"
            type="submit"
            className="btn btn-primary btn-lg"
            disabled={loading || !username || !password}
          >
            {loading ? <><span className="spinner" /> Authenticating…</> : '🔐 Sign In'}
          </button>
        </form>

        {/* Demo credentials hint */}
        <div className="login-hint">
          <p style={{ marginBottom: '0.5rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Demo Credentials</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => fillDemo('admin', 'geofusion_demo_2026')}
              style={{ all: 'unset', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}
            >
              <span className="chip chip-cyan">admin</span>
              <strong>admin</strong> / <strong>geofusion_demo_2026</strong>
              <span style={{ marginLeft: 'auto', fontSize: '0.65rem', color: 'var(--accent-emerald)' }}>Full access</span>
            </button>
            <button
              type="button"
              onClick={() => fillDemo('analyst', 'analyst_pass')}
              style={{ all: 'unset', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}
            >
              <span className="chip chip-violet">analyst</span>
              <strong>analyst</strong> / <strong>analyst_pass</strong>
              <span style={{ marginLeft: 'auto', fontSize: '0.65rem', color: 'var(--accent-amber)' }}>Read only</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
