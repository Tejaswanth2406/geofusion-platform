/**
 * GeoFusion AI — Upload Panel
 * Drag-and-drop satellite image input, sensor/mode config, search trigger
 */
import { useState, useCallback } from 'react'

const SENSORS = [
  { value: 'optical',      label: 'Optical (Sentinel-2)',     icon: '🌍' },
  { value: 'sar',          label: 'SAR (Sentinel-1 VV/VH)',   icon: '📡' },
  { value: 'multispectral',label: 'Multispectral (Landsat-8)', icon: '🌈' },
]

const MODES = [
  { value: 'cross', label: 'Cross-Modal' },
  { value: 'same',  label: 'Same-Sensor' },
]

export default function UploadPanel({ onResults, onError, token }) {
  const [file, setFile]     = useState(null)
  const [sensor, setSensor] = useState('optical')
  const [mode, setMode]     = useState('cross')
  const [topK, setTopK]     = useState(10)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const handleFileDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer?.files[0] || e.target.files?.[0]
    if (dropped) setFile(dropped)
  }, [])

  const handleSearch = async () => {
    if (!file) return
    setLoading(true)
    onError('')

    const formData = new FormData()
    formData.append('image', file)
    formData.append('sensor', sensor)
    formData.append('top_k', String(topK))
    formData.append('retrieval_mode', mode)
    formData.append('explain', 'true')

    try {
      const res = await fetch('/api/v1/retrieve', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      onResults(data, { sensor, mode, topK, filename: file.name })
    } catch (err) {
      onError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const selectedSensor = SENSORS.find(s => s.value === sensor)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

      {/* Upload zone */}
      <div className="card" style={{ padding: '1rem' }}>
        <div className="card-header">
          <div className="card-icon">📡</div>
          <span className="card-title">Satellite Image</span>
        </div>

        <div
          id="upload-dropzone"
          className={`upload-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleFileDrop}
        >
          <input
            type="file"
            id="file-input"
            accept=".tif,.tiff,.png,.jpg,.jpeg,.geotiff"
            onChange={handleFileDrop}
            aria-label="Upload satellite image"
          />
          {file ? (
            <div className="upload-filename">
              <span>✅</span>
              <span>{file.name}</span>
            </div>
          ) : (
            <>
              <span className="upload-icon">🛰️</span>
              <p className="upload-text">Drag & drop or <strong style={{ color: 'var(--accent-cyan)' }}>browse</strong></p>
              <p className="upload-subtext">.tif · .tiff · .png · .jpg · GeoTIFF</p>
            </>
          )}
        </div>

        {file && (
          <button
            className="btn btn-ghost btn-sm"
            style={{ marginTop: '0.6rem', width: '100%' }}
            onClick={() => setFile(null)}
          >
            ✕ Clear image
          </button>
        )}
      </div>

      {/* Sensor type */}
      <div className="card" style={{ padding: '1rem' }}>
        <div className="card-header">
          <div className="card-icon">🌐</div>
          <span className="card-title">Sensor Type</span>
        </div>
        <div className="form-group">
          <label className="form-label" htmlFor="sensor-select">Input sensor modality</label>
          <select
            id="sensor-select"
            className="form-select"
            value={sensor}
            onChange={(e) => setSensor(e.target.value)}
          >
            {SENSORS.map(s => (
              <option key={s.value} value={s.value}>{s.icon} {s.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Retrieval mode */}
      <div className="card" style={{ padding: '1rem' }}>
        <div className="card-header">
          <div className="card-icon">🔀</div>
          <span className="card-title">Retrieval Mode</span>
        </div>
        <div className="segmented-control" role="group" aria-label="Retrieval mode">
          {MODES.map(m => (
            <button
              key={m.value}
              id={`mode-${m.value}`}
              className={mode === m.value ? 'active' : ''}
              onClick={() => setMode(m.value)}
              type="button"
            >
              {m.label}
            </button>
          ))}
        </div>
        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.6rem' }}>
          {mode === 'cross'
            ? '🔄 Search for matching tiles across different sensors'
            : '🔍 Search for matching tiles from same sensor type'}
        </p>
      </div>

      {/* Top-K */}
      <div className="card" style={{ padding: '1rem' }}>
        <div className="card-header">
          <div className="card-icon">🏆</div>
          <span className="card-title">Results Count</span>
        </div>
        <div className="slider-group">
          <div className="slider-header">
            <label className="form-label" htmlFor="topk-slider">Top-K results</label>
            <span className="slider-value" id="topk-value">{topK}</span>
          </div>
          <input
            id="topk-slider"
            type="range"
            min="1"
            max="50"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            aria-label="Number of results"
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            <span>1</span><span>50</span>
          </div>
        </div>
      </div>

      {/* Search button */}
      <button
        id="search-btn"
        className="btn btn-primary btn-lg"
        onClick={handleSearch}
        disabled={!file || loading}
        style={{ marginTop: '0.25rem' }}
      >
        {loading
          ? <><span className="spinner" /> Retrieving…</>
          : <>🔍 Run Retrieval</>}
      </button>

      {!file && (
        <p style={{ textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Upload a satellite image to begin
        </p>
      )}
    </div>
  )
}
