from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="Credit Card Fraud Detection API")

_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_model_path = os.path.join(_base_dir, 'models', 'ensemble_isolation_forest.joblib')

_artifacts = None
try:
    _artifacts = joblib.load(_model_path)
    print("Ensemble model loaded successfully.")
except Exception as e:
    print(f"Warning: could not load model — {e}")
    print("Run  python -m src.models.train_ensemble  to train the model first.")


# ── Input schema ─────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int


# ── Feature engineering (mirrors train_ensemble.py) ──────────────────────────

def _engineer(t: dict, arts: dict) -> np.ndarray:
    le            = arts['label_encoder']
    amount_max    = arts['amount_max']
    velocity_90th = arts['velocity_90th']
    trust_10th    = arts['trust_10th']
    scaler        = arts['scaler']

    merchant = t['merchant_category']
    if merchant not in le.classes_:
        merchant = le.classes_[0]
    merchant_enc = int(le.transform([merchant])[0])

    amount   = t['amount']
    hour     = t['transaction_hour']
    foreign  = t['foreign_transaction']
    mismatch = t['location_mismatch']
    trust    = t['device_trust_score']
    velocity = t['velocity_last_24h']
    age      = t['cardholder_age']

    amount_log            = float(np.log1p(amount))
    risk_score            = amount / amount_max + mismatch + foreign
    amount_velocity_ratio = amount / (velocity + 1)
    trust_deficit         = 100 - trust
    trust_velocity        = trust_deficit * velocity
    is_night              = int(hour < 6)
    combined_risk         = foreign * mismatch
    high_velocity         = int(velocity > velocity_90th)
    low_trust             = int(trust < trust_10th)
    risk_x_amount         = risk_score * amount_log

    row = [[
        amount, hour, foreign, mismatch, trust, velocity, age, merchant_enc,
        amount_log, trust_deficit, risk_score, amount_velocity_ratio,
        trust_velocity, is_night, combined_risk, high_velocity, low_trust, risk_x_amount,
    ]]
    return scaler.transform(row)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if _artifacts is None:
        return {"error": "Model not loaded. Run train_ensemble.py first."}

    X_new = _engineer(transaction.model_dump(), _artifacts)

    models       = _artifacts['models']
    score_ranges = _artifacts['score_ranges']
    threshold    = _artifacts['threshold']

    scores = []
    for clf, (s_max, s_min) in zip(models, score_ranges):
        raw  = clf.decision_function(X_new)[0]
        norm = (s_max - raw) / (s_max - s_min + 1e-9)
        scores.append(float(norm))

    anomaly_score = float(np.mean(scores))
    is_fraud      = int(anomaly_score >= threshold)

    return {
        "is_fraud":     is_fraud,
        "anomaly_score": round(anomaly_score, 6),
        "threshold":     round(threshold, 6),
        "label":         "FRAUD" if is_fraud else "Legitimate",
    }


# ── Frontend HTML ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;
       min-height:100vh;display:flex;align-items:center;justify-content:center;padding:1.5rem}
  .card{background:#1e293b;border-radius:1rem;padding:2rem 2.25rem;
        box-shadow:0 25px 60px rgba(0,0,0,.55);width:100%;max-width:640px}
  h1{font-size:1.45rem;font-weight:700;color:#f8fafc;margin-bottom:.2rem}
  .sub{font-size:.82rem;color:#64748b;margin-bottom:1.6rem}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  .field{display:flex;flex-direction:column;gap:.35rem}
  .full{grid-column:1/-1}
  label{font-size:.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;
        letter-spacing:.06em}
  input,select{background:#0f172a;border:1.5px solid #334155;border-radius:.5rem;
               color:#e2e8f0;padding:.6rem .8rem;font-size:.9rem;width:100%;
               transition:border-color .18s}
  input:focus,select:focus{outline:none;border-color:#6366f1}
  select option{background:#1e293b}
  input[type=range]{padding:.25rem 0;accent-color:#6366f1;cursor:pointer}
  .range-row{display:flex;align-items:center;gap:.75rem}
  .range-row input{flex:1}
  .range-val{font-size:.9rem;color:#6366f1;font-weight:700;min-width:2rem;text-align:right}
  button{width:100%;margin-top:1.5rem;background:#6366f1;color:#fff;border:none;
         border-radius:.5rem;padding:.8rem;font-size:1rem;font-weight:600;cursor:pointer;
         transition:background .2s,transform .1s}
  button:hover{background:#4f46e5}
  button:active{transform:scale(.98)}
  button:disabled{background:#334155;cursor:not-allowed;transform:none}
  #result{margin-top:1.25rem;border-radius:.6rem;padding:1.1rem 1.25rem;display:none}
  #result.fraud{background:#450a0a;border:1.5px solid #dc2626}
  #result.legit{background:#052e16;border:1.5px solid #16a34a}
  #result.err{background:#1c1107;border:1.5px solid #d97706}
  .r-title{font-size:1.05rem;font-weight:700}
  .r-fraud .r-title{color:#f87171}
  .r-legit .r-title{color:#4ade80}
  .r-err  .r-title{color:#fbbf24}
  .r-detail{font-size:.8rem;color:#94a3b8;margin-top:.35rem}
  .bar-wrap{margin-top:.8rem}
  .bar-lbl{font-size:.72rem;color:#94a3b8;display:flex;justify-content:space-between;
            margin-bottom:.3rem}
  .bar-bg{background:#0f172a;border-radius:999px;height:8px;overflow:hidden}
  .bar-fill{height:100%;border-radius:999px;transition:width .5s ease}
  .badge{display:inline-block;font-size:.7rem;font-weight:700;padding:.15rem .5rem;
         border-radius:999px;margin-left:.5rem;vertical-align:middle}
  .badge-fraud{background:#dc2626;color:#fff}
  .badge-legit{background:#16a34a;color:#fff}
</style>
</head>
<body>
<div class="card">
  <h1>Credit Card Fraud Detection</h1>
  <p class="sub">Isolation Forest Ensemble &middot; 18 features &middot; ROC-AUC 0.98 &middot; Recall &ge; 0.99</p>

  <form id="txForm">
    <div class="grid">

      <div class="field">
        <label>Amount ($)</label>
        <input type="number" name="amount" placeholder="e.g. 150.00" step="0.01" min="0" required>
      </div>

      <div class="field">
        <label>Transaction Hour (0&ndash;23)</label>
        <input type="number" name="transaction_hour" placeholder="e.g. 14" min="0" max="23" required>
      </div>

      <div class="field full">
        <label>Merchant Category</label>
        <select name="merchant_category">
          <option value="Electronics">Electronics</option>
          <option value="Travel">Travel</option>
          <option value="Grocery">Grocery</option>
          <option value="Food">Food</option>
          <option value="Entertainment">Entertainment</option>
          <option value="Dining">Dining</option>
          <option value="Retail">Retail</option>
          <option value="Healthcare">Healthcare</option>
        </select>
      </div>

      <div class="field">
        <label>Foreign Transaction</label>
        <select name="foreign_transaction">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>

      <div class="field">
        <label>Location Mismatch</label>
        <select name="location_mismatch">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>

      <div class="field full">
        <label>Device Trust Score &mdash; <span id="trustLbl">62</span></label>
        <div class="range-row">
          <input type="range" name="device_trust_score" min="25" max="99" value="62"
                 oninput="document.getElementById('trustLbl').textContent=this.value">
          <span class="range-val" id="trustNum">62</span>
        </div>
      </div>

      <div class="field">
        <label>Velocity Last 24h</label>
        <input type="number" name="velocity_last_24h" placeholder="e.g. 2" min="0" max="20" required>
      </div>

      <div class="field">
        <label>Cardholder Age</label>
        <input type="number" name="cardholder_age" placeholder="e.g. 35" min="18" max="100" required>
      </div>

    </div>
    <button type="submit" id="submitBtn">Predict</button>
  </form>

  <div id="result"></div>
</div>

<script>
const trustRange = document.querySelector('input[name=device_trust_score]');
trustRange.addEventListener('input', () => {
  document.getElementById('trustLbl').textContent = trustRange.value;
  document.getElementById('trustNum').textContent = trustRange.value;
});

document.getElementById('txForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  const resultEl = document.getElementById('result');

  btn.disabled = true;
  btn.textContent = 'Analyzing…';
  resultEl.style.display = 'none';

  const fd = new FormData(e.target);
  const payload = {
    amount:               parseFloat(fd.get('amount')),
    transaction_hour:     parseInt(fd.get('transaction_hour')),
    merchant_category:    fd.get('merchant_category'),
    foreign_transaction:  parseInt(fd.get('foreign_transaction')),
    location_mismatch:    parseInt(fd.get('location_mismatch')),
    device_trust_score:   parseInt(fd.get('device_trust_score')),
    velocity_last_24h:    parseInt(fd.get('velocity_last_24h')),
    cardholder_age:       parseInt(fd.get('cardholder_age')),
  };

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await resp.json();

    if (data.error) {
      resultEl.className = 'err';
      resultEl.style.display = 'block';
      resultEl.innerHTML = `<div class="r-err"><div class="r-title">⚠️ Model not ready</div>
        <div class="r-detail">${data.error}</div></div>`;
      return;
    }

    const isFraud = data.is_fraud === 1;
    const pct     = Math.min(100, Math.max(0, data.anomaly_score * 100)).toFixed(1);
    const barClr  = isFraud ? '#dc2626' : '#16a34a';
    const cls     = isFraud ? 'fraud' : 'legit';
    const rCls    = isFraud ? 'r-fraud' : 'r-legit';
    const icon    = isFraud ? '⚠️ FRAUD DETECTED' : '✅ Legitimate Transaction';
    const badge   = isFraud
      ? '<span class="badge badge-fraud">HIGH RISK</span>'
      : '<span class="badge badge-legit">LOW RISK</span>';

    resultEl.className = cls;
    resultEl.style.display = 'block';
    resultEl.innerHTML = `
      <div class="${rCls}">
        <div class="r-title">${icon}${badge}</div>
        <div class="r-detail">
          Anomaly score: <strong>${data.anomaly_score.toFixed(4)}</strong>
          &nbsp;&middot;&nbsp;
          Threshold: ${data.threshold.toFixed(4)}
        </div>
        <div class="bar-wrap">
          <div class="bar-lbl"><span>Risk level</span><span>${pct}%</span></div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${pct}%;background:${barClr}"></div>
          </div>
        </div>
      </div>`;
  } catch (err) {
    resultEl.className = 'err';
    resultEl.style.display = 'block';
    resultEl.innerHTML = `<div class="r-err"><div class="r-title">Network error</div>
      <div class="r-detail">${err.message}</div></div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict';
  }
});
</script>
</body>
</html>"""
