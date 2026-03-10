from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .run_logger import RunLogger, new_run_id
from .pipeline import AuthenticityPipeline
from .schemas import (
    AnalysisRequest,
    AnalysisResponse,
    Category,
    CategoryScore,
    MediaType,
    ModeProfile,
    OverrideHints,
)

try:
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse
except ModuleNotFoundError:  # pragma: no cover - exercised when FastAPI installed.
    FastAPI = None  # type: ignore[assignment]

_PIPELINE: AuthenticityPipeline | None = None


def _get_pipeline() -> AuthenticityPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = AuthenticityPipeline()
    return _PIPELINE


def _parse_request(payload: dict[str, Any]) -> AnalysisRequest:
    media_type = MediaType(payload["media_type"])

    raw_scores = payload.get("provided_scores", {})
    provided_scores: dict[Category, CategoryScore] = {}
    for raw_category, raw_score in raw_scores.items():
        category = Category(raw_category)
        provided_scores[category] = CategoryScore(
            risk=float(raw_score["risk"]),
            confidence=float(raw_score["confidence"]),
            rationale=raw_score.get("rationale"),
        )

    raw_hints = payload.get("override_hints", {})
    override_hints = OverrideHints(
        provenance_declares_synthetic_or_edited=bool(
            raw_hints.get("provenance_declares_synthetic_or_edited", False)
        ),
        watermark_strong_positive=bool(raw_hints.get("watermark_strong_positive", False)),
        provenance_declares_authentic=bool(raw_hints.get("provenance_declares_authentic", False)),
        forensic_contradiction=bool(raw_hints.get("forensic_contradiction", False)),
    )
    media_uri = payload.get("media_uri")
    if isinstance(media_uri, str):
        media_uri = media_uri.strip() or None
    else:
        media_uri = None
    raw_mode = str(payload.get("mode_profile", "") or "").strip().lower()
    mode_profile: ModeProfile | None = None
    if raw_mode:
        try:
            mode_profile = ModeProfile(raw_mode)
        except ValueError:
            mode_profile = None
    return AnalysisRequest(
        media_type=media_type,
        media_uri=media_uri,
        provided_scores=provided_scores,
        override_hints=override_hints,
        mode_profile=mode_profile,
    )


def _serialize_response(response: AnalysisResponse) -> dict[str, Any]:
    return {
        "media_type": response.media_type.value,
        "verdict": response.verdict.value,
        "predicted_class": response.predicted_class,
        "class_probabilities": response.class_probabilities,
        "overall_risk": response.overall_risk,
        "overall_confidence": response.overall_confidence,
        "applied_override": response.applied_override,
        "category_scores": {
            category.value: {
                "risk": score.risk,
                "confidence": score.confidence,
                "rationale": score.rationale,
            }
            for category, score in response.category_scores.items()
        },
        "category_contributions": [
            {
                "category": contribution.category.value,
                "weighted_contribution": contribution.weighted_contribution,
                "effective_weight": contribution.effective_weight,
                "score": {
                    "risk": contribution.score.risk,
                    "confidence": contribution.score.confidence,
                    "rationale": contribution.score.rationale,
                },
            }
            for contribution in response.category_contributions
        ],
    }


def _with_run_meta(result: dict[str, Any], run_id: str, run_file: Path) -> dict[str, Any]:
    wrapped = dict(result)
    wrapped["run_id"] = run_id
    wrapped["log_file"] = str(run_file)
    return wrapped


def analyze_payload(
    payload: dict[str, Any],
    *,
    source: str = "api",
    run_id: str | None = None,
) -> dict[str, Any]:
    run_logger = RunLogger()
    run_id = run_id or new_run_id()
    pipeline = _get_pipeline()
    try:
        request = _parse_request(payload)
        response = pipeline.analyze(request)
        serialized = _serialize_response(response)
        run_file = run_logger.log_run(
            run_id=run_id,
            source=source,
            payload=payload,
            result=serialized,
            error=None,
        )
        return _with_run_meta(serialized, run_id, run_file)
    except Exception as exc:
        error_payload = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        run_file = run_logger.log_run(
            run_id=run_id,
            source=source,
            payload=payload,
            result=None,
            error=error_payload,
        )
        return {
            "error": error_payload,
            "run_id": run_id,
            "log_file": str(run_file),
        }


def _safe_float(value: Optional[str]) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if parsed < 0 or parsed > 100:
        return None
    return parsed


def _add_optional_score(
    provided_scores: dict[str, dict[str, float]],
    category: str,
    risk_raw: Optional[str],
    conf_raw: Optional[str],
) -> None:
    risk = _safe_float(risk_raw)
    conf = _safe_float(conf_raw)
    if risk is None or conf is None:
        return
    provided_scores[category] = {"risk": risk, "confidence": conf}


def _render_web_page() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Detector Local UI</title>
  <style>
    :root {
      --bg: radial-gradient(circle at 15% 15%, #d8f5ff 0%, #f6f2eb 45%, #f3fbf5 100%);
      --card: rgba(255, 255, 255, 0.86);
      --ink: #1f2937;
      --muted: #4b5563;
      --accent: #0f766e;
      --accent-2: #115e59;
      --border: rgba(15, 118, 110, 0.18);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background: var(--bg);
      min-height: 100vh;
      padding: clamp(12px, 2.2vw, 28px);
    }
    .container {
      max-width: 1080px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: clamp(12px, 2vw, 20px);
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: clamp(14px, 2vw, 18px);
      backdrop-filter: blur(3px);
      box-shadow: 0 18px 50px rgba(17, 94, 89, 0.12);
      min-width: 0;
    }
    h1 {
      grid-column: 1 / -1;
      margin: 0 0 4px 0;
      font-size: clamp(24px, 3.2vw, 32px);
      letter-spacing: 0.2px;
      line-height: 1.12;
    }
    .subtitle {
      grid-column: 1 / -1;
      color: var(--muted);
      margin-bottom: 4px;
      font-size: clamp(13px, 1.35vw, 15px);
    }
    label { font-weight: 600; display: block; margin-top: 12px; font-size: 14px; }
    input[type=text], input[type=number], input[type=file] {
      width: 100%;
      padding: 10px 11px;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 14px;
      margin-top: 6px;
      background: #ffffff;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 4px;
    }
    button {
      margin-top: 16px;
      border: 0;
      padding: 11px 14px;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .metric {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: rgba(255,255,255,0.9);
      min-width: 0;
    }
    .metric .k {
      display: block;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.2px;
      text-transform: uppercase;
    }
    .metric .v {
      display: block;
      margin-top: 4px;
      font-size: 16px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .score-table-wrap {
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: auto;
      background: rgba(255,255,255,0.92);
      margin-bottom: 10px;
      max-height: min(30vh, 280px);
    }
    .score-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .score-table th, .score-table td {
      padding: 8px;
      border-bottom: 1px solid rgba(15, 118, 110, 0.14);
      text-align: left;
      white-space: nowrap;
    }
    .score-table th {
      position: sticky;
      top: 0;
      background: #ecfeff;
      z-index: 1;
    }
    pre {
      font-family: var(--mono);
      font-size: clamp(11px, 1.3vw, 12.5px);
      line-height: 1.35;
      background: #0f172a;
      color: #e5e7eb;
      padding: 12px;
      border-radius: 12px;
      max-height: min(52vh, 560px);
      overflow: auto;
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .status {
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    ul {
      list-style: none;
      padding: 0;
      margin: 0;
      max-height: min(30vh, 250px);
      overflow: auto;
    }
    li {
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-top: 8px;
      padding: 9px;
      font-size: 12px;
      background: rgba(255,255,255,0.88);
      overflow-wrap: anywhere;
    }
    .mono { font-family: var(--mono); }
    @media (max-width: 1024px) {
      .container { grid-template-columns: 1fr; }
      h1, .subtitle { grid-column: auto; }
    }
    @media (max-width: 640px) {
      body { padding: 12px; }
      .panel { border-radius: 14px; }
      .row { grid-template-columns: 1fr; }
      button { width: 100%; }
      #refresh_logs { width: auto; }
      pre { max-height: 42vh; }
      .summary-grid { grid-template-columns: 1fr; }
      .score-table th, .score-table td { padding: 7px 6px; font-size: 11px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>AI Detector Local UI</h1>
    <div class="subtitle">Image-only test UI. Upload an image and every submission is logged under <span class="mono">logs/runs</span>.</div>
    <div class="panel">
      <label>Upload Image</label>
      <input id="media_file" type="file" accept="image/*" />

      <label>Or Image Absolute Path (optional fallback)</label>
      <input id="media_uri" type="text" placeholder="/Users/you/path/to/image.png" />

      <label>Optional Manual Category Scores (leave blank to auto-run model or neutral fallback)</label>
      <div class="row">
        <input id="provenance_risk" type="number" min="0" max="100" step="0.01" placeholder="Provenance risk" />
        <input id="provenance_conf" type="number" min="0" max="100" step="0.01" placeholder="Provenance confidence" />
      </div>
      <div class="row">
        <input id="metadata_risk" type="number" min="0" max="100" step="0.01" placeholder="Metadata risk" />
        <input id="metadata_conf" type="number" min="0" max="100" step="0.01" placeholder="Metadata confidence" />
      </div>
      <div class="row">
        <input id="prnu_risk" type="number" min="0" max="100" step="0.01" placeholder="PRNU risk" />
        <input id="prnu_conf" type="number" min="0" max="100" step="0.01" placeholder="PRNU confidence" />
      </div>
      <label>Mode Profile</label>
      <select id="mode_profile" style="width:100%;padding:10px 11px;border:1px solid var(--border);border-radius:10px;font-size:14px;margin-top:6px;background:#fff;">
        <option value="balanced" selected>balanced</option>
        <option value="strict">strict</option>
      </select>
      <button id="run_btn">Analyze Image</button>
      <div class="status" id="status">Ready.</div>
    </div>

    <div class="panel">
      <div class="result-header">
        <strong>Result</strong>
        <button id="refresh_logs" style="margin-top:0;padding:8px 10px;font-size:12px;">Refresh Logs</button>
      </div>
      <div class="summary-grid" id="summary_grid">
        <div class="metric">
          <span class="k">Verdict</span>
          <span class="v" id="metric_verdict">n/a</span>
        </div>
        <div class="metric">
          <span class="k">Predicted Class</span>
          <span class="v" id="metric_class">n/a</span>
        </div>
        <div class="metric">
          <span class="k">Mode</span>
          <span class="v" id="metric_mode">n/a</span>
        </div>
        <div class="metric">
          <span class="k">Overall Risk</span>
          <span class="v" id="metric_risk">n/a</span>
        </div>
        <div class="metric">
          <span class="k">Overall Confidence</span>
          <span class="v" id="metric_conf">n/a</span>
        </div>
      </div>
      <div class="score-table-wrap">
        <table class="score-table">
          <thead>
            <tr>
              <th>Category</th>
              <th>Risk</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody id="score_rows">
            <tr><td colspan="3">No run yet.</td></tr>
          </tbody>
        </table>
      </div>
      <pre id="result_box">{}</pre>
      <div class="status">Recent runs:</div>
      <ul id="recent_runs"></ul>
    </div>
  </div>
  <script>
    function renderScoreSummary(data) {
      const verdictEl = document.getElementById("metric_verdict");
      const classEl = document.getElementById("metric_class");
      const modeEl = document.getElementById("metric_mode");
      const riskEl = document.getElementById("metric_risk");
      const confEl = document.getElementById("metric_conf");
      const rowsEl = document.getElementById("score_rows");

      if (!data || data.error) {
        verdictEl.textContent = "error";
        classEl.textContent = "n/a";
        modeEl.textContent = "n/a";
        riskEl.textContent = "n/a";
        confEl.textContent = "n/a";
        rowsEl.innerHTML = "<tr><td colspan='3'>No category scores.</td></tr>";
        return;
      }

      verdictEl.textContent = data.verdict || "n/a";
      classEl.textContent = data.predicted_class || "n/a";
      modeEl.textContent = document.getElementById("mode_profile").value || "n/a";
      riskEl.textContent = data.overall_risk != null ? `${data.overall_risk}` : "n/a";
      confEl.textContent = data.overall_confidence != null ? `${data.overall_confidence}` : "n/a";

      const scores = data.category_scores || {};
      const entries = Object.entries(scores);
      if (entries.length === 0) {
        rowsEl.innerHTML = "<tr><td colspan='3'>No category scores.</td></tr>";
        return;
      }
      entries.sort((a, b) => (b[1].risk || 0) - (a[1].risk || 0));
      rowsEl.innerHTML = entries.map(([name, score]) => {
        const risk = score && score.risk != null ? score.risk : "n/a";
        const conf = score && score.confidence != null ? score.confidence : "n/a";
        return `<tr><td>${name}</td><td>${risk}</td><td>${conf}</td></tr>`;
      }).join("");
    }

    async function refreshLogs() {
      const listEl = document.getElementById("recent_runs");
      try {
        const resp = await fetch(`${window.location.origin}/v1/logs/recent?limit=15`);
        const data = await resp.json();
        listEl.innerHTML = "";
        (data.runs || []).forEach((run) => {
          const li = document.createElement("li");
          li.innerHTML = `
            <div><strong>${run.run_id}</strong></div>
            <div class="mono">verdict=${run.verdict || "n/a"} | risk=${run.overall_risk || "n/a"} | conf=${run.overall_confidence || "n/a"}</div>
            <div class="mono">${run.run_file}</div>
          `;
          listEl.appendChild(li);
        });
      } catch (err) {
        listEl.innerHTML = "<li>Failed to load recent runs.</li>";
      }
    }

    async function runAnalysis() {
      const status = document.getElementById("status");
      const resultBox = document.getElementById("result_box");
      status.textContent = "Analyzing...";
      const fileInput = document.getElementById("media_file");
      const file = fileInput.files && fileInput.files.length > 0 ? fileInput.files[0] : null;
      const mediaUri = document.getElementById("media_uri").value.trim();
      if (!file && !mediaUri) {
        status.textContent = "Choose a file or provide an absolute path.";
        return;
      }

      const formData = new FormData();
      if (file) {
        formData.append("file", file);
      }
      if (mediaUri) {
        formData.append("media_uri", mediaUri);
      }
      formData.append("provenance_risk", document.getElementById("provenance_risk").value);
      formData.append("provenance_conf", document.getElementById("provenance_conf").value);
      formData.append("metadata_risk", document.getElementById("metadata_risk").value);
      formData.append("metadata_conf", document.getElementById("metadata_conf").value);
      formData.append("prnu_risk", document.getElementById("prnu_risk").value);
      formData.append("prnu_conf", document.getElementById("prnu_conf").value);
      formData.append("mode_profile", document.getElementById("mode_profile").value);
      try {
        const resp = await fetch(`${window.location.origin}/v1/analyze/upload`, {
          method: "POST",
          body: formData
        });
        let data = null;
        const bodyText = await resp.text();
        try {
          data = bodyText ? JSON.parse(bodyText) : {};
        } catch (parseErr) {
          data = { error: `Non-JSON response (status ${resp.status}): ${bodyText.slice(0, 220)}` };
        }
        renderScoreSummary(data);
        resultBox.textContent = JSON.stringify(data, null, 2);
        status.textContent = data.error ? "Completed with error (logged)." : "Completed and logged.";
        await refreshLogs();
      } catch (err) {
        renderScoreSummary({ error: String(err) });
        resultBox.textContent = JSON.stringify({ error: String(err) }, null, 2);
        status.textContent = "Request failed.";
      }
    }

    document.getElementById("run_btn").addEventListener("click", runAnalysis);
    document.getElementById("refresh_logs").addEventListener("click", refreshLogs);
    refreshLogs();
  </script>
</body>
</html>"""


if FastAPI is not None:
    app = FastAPI(
        title="AI Detector API",
        version="0.1.0",
        description=(
            "Authenticity-risk analyzer for image/video with layered scoring "
            "and fusion matrix."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/analyze")
    def analyze(payload: dict[str, Any]) -> dict[str, Any]:
        return analyze_payload(payload, source="api")

    @app.post("/v1/analyze/upload")
    async def analyze_upload(
        file: UploadFile | None = File(default=None),
        media_uri: str = Form(default=""),
        provenance_risk: str = Form(default=""),
        provenance_conf: str = Form(default=""),
        metadata_risk: str = Form(default=""),
        metadata_conf: str = Form(default=""),
        prnu_risk: str = Form(default=""),
        prnu_conf: str = Form(default=""),
        mode_profile: str = Form(default="balanced"),
    ) -> dict[str, Any]:
        logger = RunLogger()
        run_id = new_run_id()

        chosen_media_uri = media_uri.strip() if media_uri else ""
        if file is not None and file.filename:
            content = await file.read()
            if not content:
                return {"error": {"type": "ValueError", "message": "uploaded file is empty"}}
            saved = logger.store_upload(file.filename, content, run_id=run_id)
            chosen_media_uri = str(saved)

        if not chosen_media_uri:
            return {"error": {"type": "ValueError", "message": "provide an image file or media_uri"}}

        provided_scores: dict[str, dict[str, float]] = {}
        _add_optional_score(provided_scores, "provenance", provenance_risk, provenance_conf)
        _add_optional_score(provided_scores, "metadata", metadata_risk, metadata_conf)
        _add_optional_score(provided_scores, "prnu", prnu_risk, prnu_conf)

        payload = {
            "media_type": "image",
            "media_uri": chosen_media_uri,
            "provided_scores": provided_scores,
            "mode_profile": mode_profile,
        }
        return analyze_payload(payload, source="web_upload", run_id=run_id)

    @app.get("/v1/logs/recent")
    def recent_logs(limit: int = 20) -> dict[str, Any]:
        logger = RunLogger()
        return {"runs": logger.recent_runs(limit=limit)}

    @app.get("/v1/logs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        logger = RunLogger()
        path = logger.run_path(run_id)
        if not path.exists():
            return {"error": f"run_id not found: {run_id}"}
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse)
    async def web_ui(_: Request) -> HTMLResponse:
        return HTMLResponse(_render_web_page())

else:
    app = None
