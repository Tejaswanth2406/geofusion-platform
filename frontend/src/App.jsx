/**
 * GeoFusion AI — Dashboard (scaffold)
 * ------------------------------------
 * Minimal placeholder showing the intended UX:
 *   - Upload satellite image (drag & drop)
 *   - Select sensor type (optical / sar / multispectral)
 *   - Select retrieval mode (optical -> sar, sar -> optical, same-sensor)
 *   - Display Top-K results with similarity scores + explainability
 *
 * Wire this up to the API Gateway at POST /api/v1/retrieve.
 * Bootstrap with Vite: `npm create vite@latest . -- --template react`
 */

import { useState } from "react";

const API_BASE = import.meta.env?.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [sensor, setSensor] = useState("optical");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("sensor", sensor);
    formData.append("top_k", "10");
    formData.append("retrieval_mode", "cross");
    formData.append("explain", "true");

    try {
      const res = await fetch(`${API_BASE}/api/v1/retrieve`, {
        method: "POST",
        body: formData,
      });
      setResults(await res.json());
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", maxWidth: 720, margin: "40px auto" }}>
      <h1>GeoFusion AI</h1>
      <p>Multi-Sensor Satellite Intelligence Retrieval Engine</p>

      <input type="file" onChange={(e) => setFile(e.target.files[0])} />

      <div style={{ margin: "12px 0" }}>
        <label>Sensor: </label>
        <select value={sensor} onChange={(e) => setSensor(e.target.value)}>
          <option value="optical">Optical</option>
          <option value="sar">SAR</option>
          <option value="multispectral">Multispectral</option>
        </select>
      </div>

      <button onClick={handleSearch} disabled={!file || loading}>
        {loading ? "Searching..." : "Search"}
      </button>

      {results && (
        <pre style={{ background: "#f4f4f4", padding: 16, marginTop: 24 }}>
          {JSON.stringify(results, null, 2)}
        </pre>
      )}
    </div>
  );
}
