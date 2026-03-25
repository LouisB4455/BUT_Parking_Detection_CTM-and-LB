import csv
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

RESULT_CSV = "resultats_modele_final.csv"
MANUAL_CSV = "check_manuel_results.csv"
HOST = "127.0.0.1"
PORT = 8050
IMAGE_PREFIX = "ModeleFinal_"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, ".."))

MODEL_SCRIPT = os.path.join(CURRENT_DIR, "analyse_modele_final.py")
CLEAN_SCRIPT = os.path.join(CURRENT_DIR, "nettoyer_check_manuel.py")
MANUAL_SCRIPT = os.path.normpath(
  os.path.join(
    ROOT_DIR,
    "module_d_analyse_des_resultats",
    "Script d'Analyse des résultats pour Analyse 1.py",
  )
)

STATE_LOCK = threading.Lock()
STATE = {
  "model_running": False,
  "last_model_run": "-",
  "last_model_exit_code": None,
  "last_model_log": "",
  "manual_running": False,
  "last_manual_start": "-",
  "last_manual_pid": None,
}

ERROR_LABELS = {
    "err1": "Voiture non detectee",
    "err2": "Fausse detection",
    "err3": "Stationnement sauvage",
    "err4": "Voiture partielle",
    "err5": "Image inexploitable",
    "err6": "Obstacle non voiture",
    "err7": "Place non visible",
    "err8": "Voiture sur 2 places",
    "err9": "Double detection",
    "err10": "Erreur cadrage de place",
}


def normalize_image_name(name: str) -> str:
    base = os.path.basename(name or "")
    if base.startswith(IMAGE_PREFIX):
        return base[len(IMAGE_PREFIX) :]
    return base


def parse_timestamp(name: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4})", name or "")
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M")
    except ValueError:
        return None


def to_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def load_results():
    if not os.path.exists(RESULT_CSV):
        return []

    rows = []
    with open(RESULT_CSV, mode="r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = normalize_image_name(row.get("image", ""))
            total_places = to_int(row.get("total_places", 0), 0)
            occupied_places = to_int(row.get("occupied_places", 0), 0)
            free_places = to_int(row.get("free_places", 0), 0)
            rows.append(
                {
                    "image": row.get("image", ""),
                    "image_key": key,
                    "ts": parse_timestamp(key),
                    "free_places": free_places,
                    "occupied_places": occupied_places,
                    "total_places": total_places,
                    "cars_detected": to_int(row.get("cars_detected", 0), 0),
                    "illegal_parked": to_int(row.get("illegal_parked", 0), 0),
                    "alignment_ok": to_int(row.get("alignment_ok", 0), 0),
                    "inlier_ratio": to_float(row.get("inlier_ratio", 0.0), 0.0),
                    "occupancy_rate": (occupied_places / total_places * 100.0) if total_places > 0 else 0.0,
                }
            )

    rows.sort(key=lambda r: (r["ts"] is None, r["ts"] or datetime.max, r["image_key"]))
    return rows


def load_manual_latest():
    if not os.path.exists(MANUAL_CSV):
        return {}

    latest = {}
    with open(MANUAL_CSV, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_image_name(row.get("image", ""))
            if not key:
                continue
            data = {"places_detectees": to_int(row.get("places_detectees", 0), 0)}
            for i in range(1, 11):
                data[f"err{i}"] = to_int(row.get(f"err{i}", 0), 0)
            latest[key] = data
    return latest


def build_payload():
    results = load_results()
    manual = load_manual_latest()

    n = len(results)
    latest = results[-1] if n else None
    latest_occupancy_rate = round(latest["occupancy_rate"], 2) if latest else 0.0
    latest_occupied = int(latest["occupied_places"]) if latest else 0
    latest_free = int(latest["free_places"]) if latest else 0
    latest_illegal = int(latest["illegal_parked"]) if latest else 0
    latest_image = latest["image_key"] if latest else "-"

    labels = []
    occupied = []
    free = []
    cars = []
    illegal = []
    inlier = []
    occupancy = []

    gaps = []
    error_totals = {k: 0 for k in ERROR_LABELS.keys()}

    for r in results:
        labels.append((r["ts"].strftime("%H:%M") if r["ts"] else r["image_key"]))
        occupied.append(r["occupied_places"])
        free.append(r["free_places"])
        cars.append(r["cars_detected"])
        illegal.append(r["illegal_parked"])
        inlier.append(r["inlier_ratio"])
        occupancy.append(round(r["occupancy_rate"], 2))

        m = manual.get(r["image_key"])
        if m:
            gaps.append(
                {
                    "image": r["image_key"],
                    "modele_free_places": r["free_places"],
                    "manuel_places_detectees": m["places_detectees"],
                    "ecart": r["free_places"] - m["places_detectees"],
                }
            )
            for i in range(1, 11):
                error_totals[f"err{i}"] += m[f"err{i}"]

    return {
        "kpis": {
        "latest_image": latest_image,
        "latest_occupancy_rate": latest_occupancy_rate,
        "latest_occupied": latest_occupied,
        "latest_free": latest_free,
        "latest_illegal": latest_illegal,
        },
        "series": {
            "labels": labels,
            "occupied": occupied,
            "free": free,
            "cars": cars,
            "illegal": illegal,
            "inlier": inlier,
            "occupancy": occupancy,
        },
        "error_totals": [
            {"code": k, "label": ERROR_LABELS[k], "value": error_totals[k]} for k in ERROR_LABELS.keys()
        ],
        "gaps": gaps,
    }


def build_errors_payload():
    manual = load_manual_latest()
    totals = {k: 0 for k in ERROR_LABELS.keys()}

    rows = []
    for image_key, m in manual.items():
        entry = {"image": image_key, "places_detectees": m.get("places_detectees", 0)}
        total_err = 0
        for i in range(1, 11):
            code = f"err{i}"
            val = int(m.get(code, 0))
            totals[code] += val
            entry[code] = val
            total_err += val
        entry["total_err"] = total_err
        rows.append(entry)

    rows.sort(key=lambda r: r["image"])

    return {
        "totals": [{"code": k, "label": ERROR_LABELS[k], "value": totals[k]} for k in ERROR_LABELS],
        "rows": rows,
    }


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_model_pipeline_background():
    py = sys.executable
    cmd_clean = [py, CLEAN_SCRIPT, "--csv", MANUAL_CSV]
    cmd_model = [py, MODEL_SCRIPT]

    logs = []
    exit_code = 0

    try:
        clean = subprocess.run(cmd_clean, cwd=CURRENT_DIR, capture_output=True, text=True)
        logs.append("$ " + " ".join(cmd_clean))
        logs.append(clean.stdout.strip())
        if clean.stderr.strip():
            logs.append(clean.stderr.strip())
        if clean.returncode != 0:
            exit_code = clean.returncode

        if exit_code == 0:
            model = subprocess.run(cmd_model, cwd=CURRENT_DIR, capture_output=True, text=True)
            logs.append("$ " + " ".join(cmd_model))
            logs.append(model.stdout.strip())
            if model.stderr.strip():
                logs.append(model.stderr.strip())
            exit_code = model.returncode
    except Exception as e:
        logs.append(f"Erreur execution pipeline: {e}")
        exit_code = 1

    with STATE_LOCK:
        STATE["model_running"] = False
        STATE["last_model_run"] = _now_str()
        STATE["last_model_exit_code"] = exit_code
        STATE["last_model_log"] = "\n".join([x for x in logs if x])


def start_model_pipeline() -> dict:
    with STATE_LOCK:
        if STATE["model_running"]:
            return {"ok": False, "message": "Le modele est deja en cours d'execution."}
        STATE["model_running"] = True

    t = threading.Thread(target=_run_model_pipeline_background, daemon=True)
    t.start()
    return {"ok": True, "message": "Execution du modele lancee."}


def start_manual_correction() -> dict:
    if not os.path.exists(MANUAL_SCRIPT):
        return {"ok": False, "message": f"Script manuel introuvable: {MANUAL_SCRIPT}"}

    py = sys.executable
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(
            [py, MANUAL_SCRIPT],
            cwd=os.path.dirname(MANUAL_SCRIPT),
            creationflags=creationflags,
        )
        with STATE_LOCK:
            STATE["manual_running"] = True
            STATE["last_manual_start"] = _now_str()
            STATE["last_manual_pid"] = proc.pid
        return {"ok": True, "message": f"Correction manuelle lancee (PID {proc.pid})."}
    except Exception as e:
        return {"ok": False, "message": f"Impossible de lancer la correction manuelle: {e}"}


def get_status_payload() -> dict:
    with STATE_LOCK:
        return {
            "model_running": STATE["model_running"],
            "last_model_run": STATE["last_model_run"],
            "last_model_exit_code": STATE["last_model_exit_code"],
            "last_model_log": STATE["last_model_log"],
            "manual_running": STATE["manual_running"],
            "last_manual_start": STATE["last_manual_start"],
            "last_manual_pid": STATE["last_manual_pid"],
        }


HTML_MAIN_PAGE = """<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Monitoring Officiel - Modele Final</title>
  <style>
    body { font-family: Segoe UI, Tahoma, sans-serif; margin: 0; background: #f4f7fb; color: #1f2a37; }
    header { padding: 16px 20px; background: linear-gradient(120deg, #0b3d91, #1478d4); color: white; }
    main { padding: 18px; display: grid; gap: 16px; }
    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
    .card { background: white; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .k-title { font-size: 12px; opacity: 0.7; }
    .k-value { font-size: 28px; font-weight: 700; margin-top: 6px; }
    .row { display: grid; grid-template-columns: 1fr; gap: 12px; }
    @media (min-width: 1024px) { .row { grid-template-columns: 2fr 1fr; } }
    canvas { width: 100%; height: 320px; border-radius: 10px; background: #fff; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }
    th, td { border-bottom: 1px solid #eef2f7; padding: 8px 10px; text-align: left; font-size: 13px; }
    th { background: #f8fafc; }
    .bar-wrap { display: grid; gap: 6px; }
    .bar-line { display: grid; grid-template-columns: 220px 1fr 56px; gap: 8px; align-items: center; font-size: 13px; }
    .bar-bg { height: 12px; background: #e6edf6; border-radius: 10px; overflow: hidden; }
    .bar-fg { height: 12px; background: #1478d4; }
    .legend { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0 10px; font-size: 13px; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
    .actions { display:flex; gap:10px; flex-wrap: wrap; margin-top: 12px; }
    button { border: 0; border-radius: 8px; padding: 10px 12px; font-weight: 600; cursor: pointer; }
    .btn-main { background: #0b3d91; color: white; }
    .btn-secondary { background: #1478d4; color: white; }
    .btn-neutral { background: #e6edf6; color: #1f2a37; }
    .status-box { white-space: pre-wrap; max-height: 180px; overflow: auto; background:#0f172a; color:#e2e8f0; padding:10px; border-radius:8px; font-family: Consolas, monospace; font-size:12px; }
  </style>
</head>
<body>
  <header>
    <h2 style=\"margin:0\">Monitoring Officiel Parking - Modele Final</h2>
    <div class=\"actions\">
      <button class=\"btn-main\" onclick=\"goErrors()\">Page Monitoring Erreurs</button>
      <button class=\"btn-secondary\" onclick=\"runManual()\">Lancer Correction Manuelle</button>
      <button class=\"btn-secondary\" onclick=\"runModel()\">Lancer Modele + Rafraichir</button>
      <button class=\"btn-neutral\" onclick=\"refreshAll()\">Rafraichir Maintenant</button>
    </div>
  </header>
  <main>
    <section class=\"card\">
      <h3 style=\"margin-top:0\">Etat des actions</h3>
      <div id=\"actionStatus\">Chargement...</div>
      <div id=\"logBox\" class=\"status-box\" style=\"margin-top:8px\"></div>
    </section>

    <section class=\"kpis\" id=\"kpis\"></section>

    <section class=\"row\">
      <div class=\"card\">
        <h3 style=\"margin-top:0\">Courbe des places occupees/libres</h3>
        <div class=\"legend\">
          <span class=\"legend-item\"><span class=\"legend-dot\" style=\"background:#e11d48\"></span>Occupees</span>
          <span class=\"legend-item\"><span class=\"legend-dot\" style=\"background:#16a34a\"></span>Libres</span>
          <span class=\"legend-item\"><span class=\"legend-dot\" style=\"background:#2563eb\"></span>Voitures detectees</span>
        </div>
        <canvas id=\"linePlaces\" width=\"1200\" height=\"320\"></canvas>
      </div>
      <div class=\"card\">
        <h3 style=\"margin-top:0\">Erreurs manuelles</h3>
        <div class=\"bar-wrap\" id=\"bars\"></div>
      </div>
    </section>

    <section class=\"row\">
      <div class=\"card\">
        <h3 style=\"margin-top:0\">Taux d'occupation (%)</h3>
        <div class=\"legend\">
          <span class=\"legend-item\"><span class=\"legend-dot\" style=\"background:#0ea5e9\"></span>% places occupees</span>
        </div>
        <canvas id=\"lineOcc\" width=\"1200\" height=\"320\"></canvas>
      </div>
      <div class=\"card\">
        <h3 style=\"margin-top:0\">Ecart modele vs manuel</h3>
        <table>
          <thead>
            <tr><th>Image</th><th>Modele libres</th><th>Manuel detectees</th><th>Ecart</th></tr>
          </thead>
          <tbody id=\"gaps\"></tbody>
        </table>
      </div>
    </section>
  </main>

<script>
function drawLineChart(canvasId, labels, series, ymin, ymax) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const pad = {l: 42, r: 14, t: 16, b: 30};

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = '#dbe5f0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (h - pad.t - pad.b) * (i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
  }

  const n = labels.length;
  const xAt = (i) => n <= 1 ? pad.l : pad.l + i * ((w - pad.l - pad.r) / (n - 1));
  const yAt = (v) => {
    const ratio = (v - ymin) / (ymax - ymin || 1);
    return h - pad.b - ratio * (h - pad.t - pad.b);
  };

  series.forEach(s => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.values.forEach((v, i) => {
      const x = xAt(i);
      const y = yAt(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = s.color;
    s.values.forEach((v, i) => {
      const x = xAt(i);
      const y = yAt(v);
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
    });
  });

  ctx.fillStyle = '#4a5565';
  ctx.font = '11px Segoe UI';
  if (n > 0) {
    const step = Math.max(1, Math.floor(n / 6));
    for (let i = 0; i < n; i += step) {
      const x = xAt(i);
      ctx.fillText(labels[i], x - 12, h - 8);
    }
  }
}

function maxValue(arrs) {
  let m = 0;
  arrs.forEach(a => a.forEach(v => { if (v > m) m = v; }));
  return m;
}

function goErrors() {
  window.location.href = '/errors';
}

function runManual() {
  fetch('/api/run-manual', {method: 'POST'})
    .then(r => r.json())
    .then(d => { alert(d.message); refreshStatus(); })
    .catch(() => alert('Erreur lancement correction manuelle'));
}

function runModel() {
  fetch('/api/run-model', {method: 'POST'})
    .then(r => r.json())
    .then(d => { alert(d.message); refreshStatus(); })
    .catch(() => alert('Erreur lancement modele'));
}

function refreshStatus() {
  fetch('/api/status').then(r => r.json()).then(s => {
    const txt =
      'Modele en cours: ' + (s.model_running ? 'OUI' : 'NON') + '\\n' +
      'Dernier run modele: ' + s.last_model_run + '\\n' +
      'Code retour modele: ' + s.last_model_exit_code + '\\n' +
      'Correction manuelle lancee: ' + s.last_manual_start + ' (PID: ' + s.last_manual_pid + ')';
    document.getElementById('actionStatus').innerText = txt;
    document.getElementById('logBox').innerText = s.last_model_log || 'Aucun log pour le moment.';
  });
}

function refreshData() {
  fetch('/api/data').then(r => r.json()).then(data => {
    const k = data.kpis;
  const kpi = [
    ['Image la plus recente', k.latest_image],
    ['% places occupees (direct)', k.latest_occupancy_rate + '%'],
    ['Nb places occupees', k.latest_occupied],
    ['Nb places libres', k.latest_free],
    ['Voitures en place illegale', k.latest_illegal]
  ];
  const kpis = document.getElementById('kpis');
  kpis.innerHTML = '';
  kpi.forEach(item => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = '<div class="k-title">' + item[0] + '</div><div class="k-value">' + item[1] + '</div>';
    kpis.appendChild(div);
  });

  const s = data.series;
  drawLineChart(
    'linePlaces',
    s.labels,
    [
      {name: 'Occupees', color: '#e11d48', values: s.occupied},
      {name: 'Libres', color: '#16a34a', values: s.free},
      {name: 'Voitures', color: '#2563eb', values: s.cars}
    ],
    0,
    Math.max(1, maxValue([s.occupied, s.free, s.cars]))
  );

  drawLineChart(
    'lineOcc',
    s.labels,
    [
      {name: 'Occupation', color: '#0ea5e9', values: s.occupancy}
    ],
    0,
    100
  );

  const bars = document.getElementById('bars');
  bars.innerHTML = '';
  const totals = data.error_totals;
  const maxErr = Math.max(1, ...totals.map(x => x.value));
  totals.forEach(e => {
    const line = document.createElement('div');
    line.className = 'bar-line';
    const pct = (e.value / maxErr) * 100;
    line.innerHTML = '<div>' + e.label + '</div>' +
      '<div class="bar-bg"><div class="bar-fg" style="width:' + pct + '%"></div></div>' +
      '<div>' + e.value + '</div>';
    bars.appendChild(line);
  });

  const gaps = document.getElementById('gaps');
  gaps.innerHTML = '';
  data.gaps.forEach(g => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + g.image + '</td><td>' + g.modele_free_places + '</td><td>' + g.manuel_places_detectees + '</td><td>' + g.ecart + '</td>';
    gaps.appendChild(tr);
  });
  });
}

function refreshAll() {
  refreshStatus();
  refreshData();
}

refreshAll();
setInterval(refreshStatus, 3000);
setInterval(refreshData, 5000);
</script>
</body>
</html>
"""


HTML_ERRORS_PAGE = """<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Monitoring Erreurs - Modele Final</title>
  <style>
    body { font-family: Segoe UI, Tahoma, sans-serif; margin: 0; background: #f4f7fb; color: #1f2a37; }
    header { padding: 16px 20px; background: linear-gradient(120deg, #8b1e3f, #c2410c); color: white; }
    main { padding: 18px; display: grid; gap: 16px; }
    .actions { display:flex; gap:10px; flex-wrap: wrap; margin-top: 12px; }
    button { border: 0; border-radius: 8px; padding: 10px 12px; font-weight: 600; cursor: pointer; }
    .btn-main { background: #7f1d1d; color: white; }
    .btn-secondary { background: #b91c1c; color: white; }
    .card { background: white; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .bar-wrap { display: grid; gap: 6px; }
    .bar-line { display: grid; grid-template-columns: 220px 1fr 56px; gap: 8px; align-items: center; font-size: 13px; }
    .bar-bg { height: 12px; background: #fee2e2; border-radius: 10px; overflow: hidden; }
    .bar-fg { height: 12px; background: #dc2626; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }
    th, td { border-bottom: 1px solid #eef2f7; padding: 8px 10px; text-align: left; font-size: 13px; }
    th { background: #f8fafc; }
  </style>
</head>
<body>
  <header>
    <h2 style=\"margin:0\">Monitoring Erreurs</h2>
    <div class=\"actions\">
      <button class=\"btn-main\" onclick=\"goMain()\">Retour Monitoring Principal</button>
      <button class=\"btn-secondary\" onclick=\"runManual()\">Lancer Correction Manuelle</button>
    </div>
  </header>
  <main>
    <section class=\"card\">
      <h3 style=\"margin-top:0\">Totaux erreurs manuelles</h3>
      <div class=\"bar-wrap\" id=\"bars\"></div>
    </section>
    <section class=\"card\">
      <h3 style=\"margin-top:0\">Detail par image</h3>
      <table>
        <thead>
          <tr>
            <th>Image</th><th>Total err</th>
            <th>err1</th><th>err2</th><th>err3</th><th>err4</th><th>err5</th>
            <th>err6</th><th>err7</th><th>err8</th><th>err9</th><th>err10</th>
          </tr>
        </thead>
        <tbody id=\"rows\"></tbody>
      </table>
    </section>
  </main>
  <script>
  function goMain() {
    window.location.href = '/';
  }

  function runManual() {
    fetch('/api/run-manual', {method: 'POST'})
      .then(r => r.json())
      .then(d => alert(d.message))
      .catch(() => alert('Erreur lancement correction manuelle'));
  }

  function refreshErrors() {
    fetch('/api/errors').then(r => r.json()).then(data => {
      const bars = document.getElementById('bars');
      bars.innerHTML = '';
      const maxErr = Math.max(1, ...data.totals.map(x => x.value));
      data.totals.forEach(e => {
        const line = document.createElement('div');
        line.className = 'bar-line';
        const pct = (e.value / maxErr) * 100;
        line.innerHTML = '<div>' + e.label + '</div>' +
          '<div class=\"bar-bg\"><div class=\"bar-fg\" style=\"width:' + pct + '%\"></div></div>' +
          '<div>' + e.value + '</div>';
        bars.appendChild(line);
      });

      const rows = document.getElementById('rows');
      rows.innerHTML = '';
      data.rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + r.image + '</td><td>' + r.total_err + '</td>' +
          '<td>' + r.err1 + '</td><td>' + r.err2 + '</td><td>' + r.err3 + '</td><td>' + r.err4 + '</td><td>' + r.err5 + '</td>' +
          '<td>' + r.err6 + '</td><td>' + r.err7 + '</td><td>' + r.err8 + '</td><td>' + r.err9 + '</td><td>' + r.err10 + '</td>';
        rows.appendChild(tr);
      });
    });
  }

  refreshErrors();
  setInterval(refreshErrors, 5000);
  </script>
</body>
</html>
"""


class MonitoringHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send(200, HTML_MAIN_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/errors":
            self._send(200, HTML_ERRORS_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/data":
            payload = build_payload()
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return

        if path == "/api/errors":
            payload = build_errors_payload()
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return

        if path == "/api/status":
            payload = get_status_payload()
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return

        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/run-model":
            payload = start_model_pipeline()
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return

        if path == "/api/run-manual":
            payload = start_manual_correction()
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return

        self._send(404, b"Not found", "text/plain; charset=utf-8")


def main():
    os.chdir(CURRENT_DIR)
    server = ThreadingHTTPServer((HOST, PORT), MonitoringHandler)
    print(f"Monitoring local: http://{HOST}:{PORT}")
    print("Aucun input requis: interface principale + page erreurs + boutons d'action backend")
    server.serve_forever()


if __name__ == "__main__":
    main()
