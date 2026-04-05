# ============================================================
# Block 12: Flask Web App V2
# File: app_v2.py
# Run: python app_v2.py
# Open: http://127.0.0.1:5000
# ============================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load model pipeline ───────────────────────────────────────
import importlib.util, os, sys

spec = importlib.util.spec_from_file_location(
           "loan_model_v2",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_model_v2.py")
       )
mod = importlib.util.module_from_spec(spec)   # ← was load_from_spec
spec.loader.exec_module(mod)

# Pull into local scope
X             = mod.X
scaler        = mod.scaler
trad_models   = mod.trad_models
alt_models    = mod.alt_models
trad_features = mod.trad_features
alt_features  = mod.alt_features

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Loan Approval Predictor V2</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #080c14;
  --surface:   #0e1420;
  --surface2:  #141c2e;
  --surface3:  #1a2438;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(255,255,255,0.12);
  --accent:    #4f8ef7;
  --accent2:   #2563eb;
  --green:     #10b981;
  --green2:    #059669;
  --red:       #f43f5e;
  --amber:     #f59e0b;
  --purple:    #8b5cf6;
  --text:      #e2e8f0;
  --muted:     #4b5563;
  --label:     #8b95a8;
  --radius:    14px;
}

body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  padding: 48px 16px 80px;
}

/* ── Animated background grid ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(79,142,247,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(79,142,247,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

/* ── Header ── */
.header {
  text-align: center;
  margin-bottom: 40px;
  position: relative;
  z-index: 1;
}
.header-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(79,142,247,0.1);
  border: 1px solid rgba(79,142,247,0.2);
  border-radius: 20px;
  padding: 5px 14px;
  font-size: 0.72rem;
  color: var(--accent);
  letter-spacing: 0.8px;
  text-transform: uppercase;
  font-weight: 500;
  margin-bottom: 16px;
}
.header h1 {
  font-size: 2.4rem;
  font-weight: 700;
  letter-spacing: -1px;
  line-height: 1.1;
  margin-bottom: 10px;
}
.header h1 span {
  background: linear-gradient(135deg, #4f8ef7, #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.header p {
  font-size: 0.88rem;
  color: var(--label);
  max-width: 420px;
  margin: 0 auto 20px;
  line-height: 1.6;
}
.stat-row {
  display: flex;
  justify-content: center;
  gap: 24px;
  flex-wrap: wrap;
}
.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.stat-val {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent);
  font-family: 'DM Mono', monospace;
}
.stat-label {
  font-size: 0.68rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

/* ── Layout ── */
.layout {
  max-width: 1000px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  position: relative;
  z-index: 1;
}

/* ── Card ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 22px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.card-icon {
  width: 34px; height: 34px;
  border-radius: 8px;
  background: rgba(79,142,247,0.1);
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
}
.card-title {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--label);
}

/* ── Form fields ── */
.field { margin-bottom: 14px; }
.field label {
  display: block;
  font-size: 0.73rem;
  font-weight: 500;
  color: var(--label);
  margin-bottom: 6px;
  letter-spacing: 0.2px;
}
input, select {
  width: 100%;
  padding: 10px 13px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-size: 0.875rem;
  font-family: 'DM Sans', sans-serif;
  transition: border-color 0.15s, box-shadow 0.15s;
  appearance: none;
}
input:focus, select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(79,142,247,0.12);
}
select {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%234b5563' d='M5 6L0 0h10z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 30px;
  cursor: pointer;
}
select option { background: #141c2e; }

/* Loan type selector */
.loan-type-group {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
  margin-bottom: 14px;
}
.loan-type-btn {
  padding: 10px 6px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--label);
  font-size: 0.78rem;
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
  cursor: pointer;
  text-align: center;
  transition: all 0.15s;
}
.loan-type-btn.active {
  background: rgba(79,142,247,0.12);
  border-color: var(--accent);
  color: var(--accent);
}
.loan-type-label {
  font-size: 0.73rem;
  font-weight: 500;
  color: var(--label);
  margin-bottom: 8px;
  display: block;
  letter-spacing: 0.2px;
}

/* CIBIL slider */
.slider-wrap { position: relative; }
input[type=range] {
  -webkit-appearance: none;
  appearance: none;
  height: 5px;
  background: var(--surface3);
  border: none;
  border-radius: 10px;
  padding: 0;
  box-shadow: none;
  cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px; height: 18px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: 2px solid var(--bg);
  box-shadow: 0 0 0 3px rgba(79,142,247,0.2);
}
.slider-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
}
.slider-labels span {
  font-size: 0.67rem;
  color: var(--muted);
  font-family: 'DM Mono', monospace;
}
.cibil-display {
  text-align: center;
  font-family: 'DM Mono', monospace;
  font-size: 1.4rem;
  font-weight: 600;
  margin-bottom: 10px;
  transition: color 0.3s;
}

/* ── Buttons ── */
.btn-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  margin-top: 20px;
}
.btn-predict {
  padding: 12px 20px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border: none;
  border-radius: 9px;
  color: white;
  font-size: 0.9rem;
  font-weight: 600;
  font-family: 'DM Sans', sans-serif;
  cursor: pointer;
  transition: opacity 0.15s, transform 0.1s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.btn-predict:hover  { opacity: 0.9; }
.btn-predict:active { transform: scale(0.98); }
.btn-reset {
  padding: 12px 16px;
  background: transparent;
  border: 1px solid var(--border2);
  border-radius: 9px;
  color: var(--label);
  font-size: 0.85rem;
  font-family: 'DM Sans', sans-serif;
  cursor: pointer;
  transition: all 0.15s;
}
.btn-reset:hover { color: var(--text); border-color: var(--label); }

/* spinner */
.spinner {
  display: none;
  width: 16px; height: 16px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Result Panel ── */
.result-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* placeholder state */
.placeholder {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 40px 20px;
  color: var(--muted);
  gap: 12px;
}
.placeholder-icon { font-size: 2.5rem; opacity: 0.3; }
.placeholder p { font-size: 0.82rem; line-height: 1.6; max-width: 200px; }

/* verdict card */
.verdict-card {
  border-radius: 12px;
  padding: 20px;
  border: 1px solid var(--border);
  display: none;
}
.verdict-approved { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.25); }
.verdict-rejected { background: rgba(244,63,94,0.08);  border-color: rgba(244,63,94,0.25); }
.verdict-review   { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.25); }

.verdict-top { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
.verdict-icon {
  width: 48px; height: 48px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; flex-shrink: 0;
}
.vi-approved { background: rgba(16,185,129,0.15); }
.vi-rejected { background: rgba(244,63,94,0.15); }
.vi-review   { background: rgba(245,158,11,0.15); }

.verdict-label {
  font-size: 1.3rem;
  font-weight: 700;
  letter-spacing: -0.3px;
}
.vl-approved { color: var(--green); }
.vl-rejected { color: var(--red);   }
.vl-review   { color: var(--amber); }

.verdict-sub { font-size: 0.78rem; color: var(--label); margin-top: 2px; }

/* Prob bar */
.prob-row {
  display: flex; justify-content: space-between;
  margin-bottom: 7px;
}
.prob-row span { font-size: 0.72rem; color: var(--label); font-weight: 500; }
.prob-row strong { font-size: 0.72rem; font-family: 'DM Mono', monospace; }
.prob-track {
  height: 7px;
  background: rgba(255,255,255,0.06);
  border-radius: 20px;
  overflow: hidden;
  margin-bottom: 16px;
}
.prob-fill {
  height: 100%;
  border-radius: 20px;
  transition: width 1s cubic-bezier(0.4,0,0.2,1);
  width: 0%;
}
.pf-approved { background: linear-gradient(90deg, #059669, #10b981); }
.pf-rejected { background: linear-gradient(90deg, #be123c, #f43f5e); }
.pf-review   { background: linear-gradient(90deg, #b45309, #f59e0b); }

/* FOIR gauge */
.gauge-section { margin-bottom: 16px; }
.gauge-label-row {
  display: flex; justify-content: space-between;
  margin-bottom: 6px;
}
.gauge-label { font-size: 0.7rem; color: var(--label); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.gauge-value { font-size: 0.7rem; font-family: 'DM Mono', monospace; }
.gauge-track {
  height: 10px;
  background: rgba(255,255,255,0.05);
  border-radius: 20px;
  overflow: visible;
  position: relative;
  margin-bottom: 4px;
}
.gauge-zones {
  position: absolute;
  inset: 0;
  display: flex;
  border-radius: 20px;
  overflow: hidden;
}
.gz-safe   { background: rgba(16,185,129,0.2); flex: 30; }
.gz-mod    { background: rgba(245,158,11,0.2); flex: 20; }
.gz-danger { background: rgba(244,63,94,0.2);  flex: 50; }
.gauge-fill {
  position: absolute;
  top: 0; left: 0;
  height: 100%;
  border-radius: 20px;
  transition: width 1s cubic-bezier(0.4,0,0.2,1);
  width: 0%;
}
.gauge-marker {
  position: absolute;
  top: -3px;
  width: 3px; height: 16px;
  background: white;
  border-radius: 2px;
  transform: translateX(-50%);
  transition: left 1s cubic-bezier(0.4,0,0.2,1);
  box-shadow: 0 0 6px rgba(255,255,255,0.4);
}
.gauge-ticks {
  display: flex;
  justify-content: space-between;
  padding: 0 2px;
}
.gauge-ticks span { font-size: 0.6rem; color: var(--muted); font-family: 'DM Mono', monospace; }

/* CIBIL band */
.cibil-band-section { margin-bottom: 16px; }
.cibil-band-track {
  display: grid;
  grid-template-columns: repeat(5,1fr);
  height: 8px;
  border-radius: 20px;
  overflow: hidden;
  margin-bottom: 5px;
}
.cb { height: 100%; transition: opacity 0.5s; }
.cb1 { background: #f43f5e; }
.cb2 { background: #fb923c; }
.cb3 { background: #f59e0b; }
.cb4 { background: #84cc16; }
.cb5 { background: #10b981; }
.cb.inactive { opacity: 0.15; }
.cibil-band-labels {
  display: grid;
  grid-template-columns: repeat(5,1fr);
  text-align: center;
}
.cibil-band-labels span { font-size: 0.58rem; color: var(--muted); font-family: 'DM Mono', monospace; }

/* Info chips */
.chip-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 16px;
}
.chip {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.chip-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 3px; }
.chip-value { font-size: 0.82rem; font-weight: 600; font-family: 'DM Mono', monospace; }

/* Breakdown */
.breakdown-title {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
  margin-bottom: 8px;
  font-weight: 600;
}
.breakdown-list { list-style: none; }
.breakdown-list li {
  font-size: 0.79rem;
  color: var(--label);
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: flex-start;
  gap: 8px;
  line-height: 1.4;
}
.breakdown-list li:last-child { border-bottom: none; }
.dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
  margin-top: 5px;
}

/* model badge */
.model-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(139,92,246,0.1);
  border: 1px solid rgba(139,92,246,0.2);
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 0.68rem;
  color: var(--purple);
  font-weight: 500;
  margin-bottom: 14px;
}

/* responsive */
@media (max-width: 700px) {
  .layout { grid-template-columns: 1fr; }
  .header h1 { font-size: 1.8rem; }
  .chip-grid { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-badge">&#9679; AI-Powered Loan Assessment</div>
  <h1>Loan Approval <span>Predictor</span></h1>
  <p>Hybrid ML model with bias reduction. Routes applicants based on CIBIL score for fair, accurate decisions.</p>
  <div class="stat-row">
    <div class="stat"><span class="stat-val">94.4%</span><span class="stat-label">Traditional F1</span></div>
    <div class="stat"><span class="stat-val">84.2%</span><span class="stat-label">Alternative F1</span></div>
    <div class="stat"><span class="stat-val">&lt;0.3%</span><span class="stat-label">Gender Influence</span></div>
    <div class="stat"><span class="stat-val">500</span><span class="stat-label">Training Rows</span></div>
  </div>
</div>

<!-- Main Layout -->
<div class="layout">

  <!-- LEFT: Form -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon">&#128203;</div>
      <span class="card-title">Applicant Information</span>
    </div>

    <!-- Loan Type -->
    <span class="loan-type-label">Loan Type</span>
    <div class="loan-type-group">
      <button class="loan-type-btn active" onclick="setLoanType('Personal',this)">&#128179; Personal</button>
      <button class="loan-type-btn" onclick="setLoanType('Home',this)">&#127968; Home</button>
      <button class="loan-type-btn" onclick="setLoanType('Vehicle',this)">&#128663; Vehicle</button>
    </div>
    <input type="hidden" id="Loan_Type" value="Personal"/>

    <!-- CIBIL Score Slider -->
    <div class="field">
      <label>CIBIL Score</label>
      <div class="cibil-display" id="cibil-display">720</div>
      <div class="slider-wrap">
        <input type="range" id="CIBIL_Score" min="300" max="900" value="720"
               oninput="updateCibil(this.value)"/>
        <div class="slider-labels">
          <span>300</span><span>500</span><span>650</span><span>750</span><span>900</span>
        </div>
      </div>
    </div>

    <!-- Row fields -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div class="field">
        <label>Gender</label>
        <select id="Gender"><option>Male</option><option>Female</option></select>
      </div>
      <div class="field">
        <label>Married</label>
        <select id="Married"><option>Yes</option><option>No</option></select>
      </div>
      <div class="field">
        <label>Dependents</label>
        <select id="Dependents"><option>0</option><option>1</option><option>2</option><option>3+</option></select>
      </div>
      <div class="field">
        <label>Education</label>
        <select id="Education"><option>Graduate</option><option>Not Graduate</option></select>
      </div>
      <div class="field">
        <label>Self Employed</label>
        <select id="Self_Employed"><option>No</option><option>Yes</option></select>
      </div>
      <div class="field">
        <label>Property Area</label>
        <select id="Property_Area"><option>Urban</option><option>Semiurban</option><option>Rural</option></select>
      </div>
      <div class="field">
        <label>Applicant Income (&#8377;/mo)</label>
        <input type="number" id="ApplicantIncome" value="50000" min="0"/>
      </div>
      <div class="field">
        <label>Co-applicant Income (&#8377;/mo)</label>
        <input type="number" id="CoapplicantIncome" value="0" min="0"/>
      </div>
      <div class="field">
        <label>Loan Amount (&#8377; thousands)</label>
        <input type="number" id="LoanAmount" value="200" min="1"/>
      </div>
      <div class="field">
        <label>Loan Term (months)</label>
        <select id="Loan_Amount_Term">
          <option value="12">12 mo (1 yr)</option>
          <option value="24">24 mo (2 yr)</option>
          <option value="36">36 mo (3 yr)</option>
          <option value="48">48 mo (4 yr)</option>
          <option value="60">60 mo (5 yr)</option>
          <option value="84">84 mo (7 yr)</option>
          <option value="120">120 mo (10 yr)</option>
          <option value="180">180 mo (15 yr)</option>
          <option value="240" selected>240 mo (20 yr)</option>
          <option value="300">300 mo (25 yr)</option>
          <option value="360">360 mo (30 yr)</option>
        </select>
      </div>
    </div>

    <div class="field">
      <label>Existing Monthly EMIs (&#8377;)</label>
      <input type="number" id="Existing_EMIs" value="0" min="0"/>
    </div>

    <div class="btn-row">
      <button class="btn-predict" onclick="predict()">
        <span id="btn-text">Predict Approval</span>
        <div class="spinner" id="spinner"></div>
      </button>
      <button class="btn-reset" onclick="resetForm()">Reset</button>
    </div>
  </div>

  <!-- RIGHT: Result -->
  <div class="card result-panel" id="result-panel">
    <div class="card-header">
      <div class="card-icon">&#128200;</div>
      <span class="card-title">Assessment Result</span>
    </div>

    <!-- Placeholder -->
    <div class="placeholder" id="placeholder">
      <div class="placeholder-icon">&#9878;</div>
      <p>Fill in the applicant details and click Predict to see the assessment.</p>
    </div>

    <!-- Verdict (hidden until predicted) -->
    <div id="verdict-card" class="verdict-card">

      <div class="verdict-top">
        <div class="verdict-icon" id="v-icon"></div>
        <div>
          <div class="verdict-label" id="v-label"></div>
          <div class="verdict-sub"  id="v-sub"></div>
        </div>
      </div>

      <!-- Model badge -->
      <div class="model-badge" id="v-model"></div>

      <!-- Probability bar -->
      <div class="prob-row">
        <span>Approval Probability</span>
        <strong id="v-prob-text"></strong>
      </div>
      <div class="prob-track">
        <div class="prob-fill" id="v-prob-fill"></div>
      </div>

      <!-- FOIR Gauge -->
      <div class="gauge-section">
        <div class="gauge-label-row">
          <span class="gauge-label">FOIR (Repayment Burden)</span>
          <span class="gauge-value" id="foir-val"></span>
        </div>
        <div class="gauge-track">
          <div class="gauge-zones">
            <div class="gz-safe"></div>
            <div class="gz-mod"></div>
            <div class="gz-danger"></div>
          </div>
          <div class="gauge-fill" id="foir-fill"></div>
          <div class="gauge-marker" id="foir-marker"></div>
        </div>
        <div class="gauge-ticks">
          <span>0%</span><span>30%</span><span>50%</span><span>75%</span><span>100%+</span>
        </div>
      </div>

      <!-- CIBIL Band -->
      <div class="cibil-band-section">
        <div class="gauge-label-row">
          <span class="gauge-label">CIBIL Band</span>
          <span class="gauge-value" id="cibil-band-text"></span>
        </div>
        <div class="cibil-band-track">
          <div class="cb cb1" id="cb1"></div>
          <div class="cb cb2" id="cb2"></div>
          <div class="cb cb3" id="cb3"></div>
          <div class="cb cb4" id="cb4"></div>
          <div class="cb cb5" id="cb5"></div>
        </div>
        <div class="cibil-band-labels">
          <span>&lt;550</span><span>550-600</span><span>600-650</span><span>650-750</span><span>750+</span>
        </div>
      </div>

      <!-- Chips -->
      <div class="chip-grid">
        <div class="chip">
          <div class="chip-label">Total Income</div>
          <div class="chip-value" id="chip-income"></div>
        </div>
        <div class="chip">
          <div class="chip-label">Monthly EMI</div>
          <div class="chip-value" id="chip-emi"></div>
        </div>
        <div class="chip">
          <div class="chip-label">Routing Reason</div>
          <div class="chip-value" id="chip-route"></div>
        </div>
        <div class="chip">
          <div class="chip-label">Loan Type</div>
          <div class="chip-value" id="chip-loantype"></div>
        </div>
      </div>

      <!-- Breakdown -->
      <div class="breakdown-title">Decision Factors</div>
      <ul class="breakdown-list" id="breakdown-list"></ul>

    </div>
  </div>
</div>

<script>
  let selectedLoanType = 'Personal';

  function setLoanType(type, btn) {
    selectedLoanType = type;
    document.getElementById('Loan_Type').value = type;
    document.querySelectorAll('.loan-type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // Suggest sensible defaults
    const termMap = { Personal: '36', Home: '240', Vehicle: '60' };
    const loanMap = { Personal: '150', Home: '3500', Vehicle: '600' };
    document.getElementById('Loan_Amount_Term').value = termMap[type];
    document.getElementById('LoanAmount').value       = loanMap[type];
  }

  function updateCibil(val) {
    const display = document.getElementById('cibil-display');
    display.textContent = val;
    const color = val >= 750 ? '#10b981'
                : val >= 700 ? '#84cc16'
                : val >= 650 ? '#f59e0b'
                : val >= 600 ? '#fb923c'
                :              '#f43f5e';
    display.style.color = color;
  }
  updateCibil(720);

  async function predict() {
    document.getElementById('btn-text').style.display = 'none';
    document.getElementById('spinner').style.display  = 'block';

    const data = {
      Loan_Type:         document.getElementById('Loan_Type').value,
      Gender:            document.getElementById('Gender').value,
      Married:           document.getElementById('Married').value,
      Dependents:        document.getElementById('Dependents').value,
      Education:         document.getElementById('Education').value,
      Self_Employed:     document.getElementById('Self_Employed').value,
      ApplicantIncome:   parseFloat(document.getElementById('ApplicantIncome').value)||0,
      CoapplicantIncome: parseFloat(document.getElementById('CoapplicantIncome').value)||0,
      LoanAmount:        parseFloat(document.getElementById('LoanAmount').value)||1,
      Loan_Amount_Term:  parseFloat(document.getElementById('Loan_Amount_Term').value)||36,
      CIBIL_Score:       parseFloat(document.getElementById('CIBIL_Score').value)||700,
      Existing_EMIs:     parseFloat(document.getElementById('Existing_EMIs').value)||0,
      Property_Area:     document.getElementById('Property_Area').value
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(data)
      });
      const j = await res.json();
      showResult(j, data);
    } catch(e) {
      alert('Connection error. Make sure app_v2.py is running.');
    }

    document.getElementById('btn-text').style.display = 'inline';
    document.getElementById('spinner').style.display  = 'none';
  }

  function showResult(j, data) {
    document.getElementById('placeholder').style.display   = 'none';
    const vc = document.getElementById('verdict-card');
    vc.style.display = 'block';

    const st = j.status;
    const cfg = {
      approved: { icon:'✅', label:'Loan Approved',          cls:'approved', vi:'vi-approved', vl:'vl-approved', pf:'pf-approved' },
      rejected: { icon:'❌', label:'Loan Rejected',          cls:'rejected', vi:'vi-rejected', vl:'vl-rejected', pf:'pf-rejected' },
      review:   { icon:'🔎', label:'Manual Review Required', cls:'review',   vi:'vi-review',   vl:'vl-review',   pf:'pf-review'   }
    }[st];

    vc.className = `verdict-card verdict-${cfg.cls}`;

    document.getElementById('v-icon').className  = `verdict-icon ${cfg.vi}`;
    document.getElementById('v-icon').textContent = cfg.icon;
    document.getElementById('v-label').className  = `verdict-label ${cfg.vl}`;
    document.getElementById('v-label').textContent = cfg.label;
    document.getElementById('v-sub').textContent   =
        st==='review' ? 'Borderline — recommend human officer review'
                      : `Model confidence: ${(j.probability*100).toFixed(1)}%`;

    document.getElementById('v-model').textContent = '⚡ ' + j.model_used;

    // Prob bar
    const pct = Math.round(j.probability * 100);
    document.getElementById('v-prob-text').textContent = pct + '%';
    document.getElementById('v-prob-text').style.color =
        st==='approved' ? 'var(--green)' : st==='rejected' ? 'var(--red)' : 'var(--amber)';
    const fill = document.getElementById('v-prob-fill');
    fill.className = `prob-fill ${cfg.pf}`;
    setTimeout(() => { fill.style.width = pct + '%'; }, 50);

    // FOIR gauge (cap display at 100%)
    const foirPct = Math.min(j.foir * 100, 100);
    const foirColor = j.foir < 0.30 ? '#10b981' : j.foir < 0.50 ? '#f59e0b' : '#f43f5e';
    document.getElementById('foir-val').textContent  = (j.foir*100).toFixed(1) + '%';
    document.getElementById('foir-val').style.color  = foirColor;
    const gf = document.getElementById('foir-fill');
    gf.style.background = foirColor;
    const gm = document.getElementById('foir-marker');
    setTimeout(() => {
      gf.style.width  = foirPct + '%';
      gm.style.left   = foirPct + '%';
    }, 50);

    // CIBIL band
    const cs = data.CIBIL_Score;
    const band = cs < 550 ? 1 : cs < 600 ? 2 : cs < 650 ? 3 : cs < 750 ? 4 : 5;
    const bandLabels = {1:'Poor (<550)',2:'Weak (550-600)',3:'Fair (600-650)',4:'Good (650-750)',5:'Excellent (750+)'};
    document.getElementById('cibil-band-text').textContent = bandLabels[band];
    for(let i=1;i<=5;i++){
      document.getElementById('cb'+i).classList.toggle('inactive', i !== band);
    }

    // Chips
    const ti  = data.ApplicantIncome + data.CoapplicantIncome;
    const emi = (data.LoanAmount * 1000) / data.Loan_Amount_Term;
    document.getElementById('chip-income').textContent   = 'Rs.' + ti.toLocaleString('en-IN');
    document.getElementById('chip-emi').textContent      = 'Rs.' + Math.round(emi).toLocaleString('en-IN') + '/mo';
    document.getElementById('chip-route').textContent    = cs >= 650 ? 'CIBIL >= 650' : 'CIBIL < 650';
    document.getElementById('chip-loantype').textContent = data.Loan_Type;

    // Breakdown
    const ul = document.getElementById('breakdown-list');
    ul.innerHTML = '';
    (j.breakdown||[]).forEach(item => {
      const li = document.createElement('li');
      li.innerHTML = `<span class="dot"></span>${item}`;
      ul.appendChild(li);
    });
  }

  function resetForm() {
    document.getElementById('Gender').value           = 'Male';
    document.getElementById('Married').value          = 'Yes';
    document.getElementById('Dependents').value       = '0';
    document.getElementById('Education').value        = 'Graduate';
    document.getElementById('Self_Employed').value    = 'No';
    document.getElementById('Property_Area').value    = 'Urban';
    document.getElementById('ApplicantIncome').value  = '50000';
    document.getElementById('CoapplicantIncome').value= '0';
    document.getElementById('LoanAmount').value       = '200';
    document.getElementById('Loan_Amount_Term').value = '240';
    document.getElementById('CIBIL_Score').value      = '720';
    document.getElementById('Existing_EMIs').value    = '0';
    updateCibil(720);
    setLoanType('Personal', document.querySelectorAll('.loan-type-btn')[0]);
    document.getElementById('placeholder').style.display    = 'flex';
    document.getElementById('verdict-card').style.display   = 'none';
  }
</script>
</body>
</html>
"""

def predict_loan(applicant):
    row = {
        'Gender'           : 1 if applicant['Gender']=='Male' else 0,
        'Married'          : 1 if applicant['Married']=='Yes' else 0,
        'Dependents'       : int(str(applicant['Dependents']).replace('+','')),
        'Education'        : 1 if applicant['Education']=='Graduate' else 0,
        'Self_Employed'    : 1 if applicant['Self_Employed']=='Yes' else 0,
        'ApplicantIncome'  : applicant['ApplicantIncome'],
        'CoapplicantIncome': applicant['CoapplicantIncome'],
        'LoanAmount'       : applicant['LoanAmount'],
        'Loan_Amount_Term' : applicant['Loan_Amount_Term'],
        'CIBIL_Score'      : applicant['CIBIL_Score'],
        'Existing_EMIs'    : applicant['Existing_EMIs'],
        'Property_Area'    : {'Rural':0,'Semiurban':1,'Urban':2}[applicant['Property_Area']],
        'LoanType_Home'    : 1 if applicant['Loan_Type']=='Home'     else 0,
        'LoanType_Personal': 1 if applicant['Loan_Type']=='Personal' else 0,
        'LoanType_Vehicle' : 1 if applicant['Loan_Type']=='Vehicle'  else 0,
    }

    row['Total_Income']      = row['ApplicantIncome'] + row['CoapplicantIncome']
    emi_approx               = (row['LoanAmount'] * 1000) / (row['Loan_Amount_Term'] + 1e-6)
    row['FOIR']              = (emi_approx + row['Existing_EMIs']) / (row['Total_Income'] + 1e-6)
    row['Income_Loan_Ratio'] = row['Total_Income'] / (row['LoanAmount'] + 1e-6)
    row['CIBIL_Bucket']      = (0 if applicant['CIBIL_Score']<600 else
                                1 if applicant['CIBIL_Score']<650 else
                                2 if applicant['CIBIL_Score']<700 else
                                3 if applicant['CIBIL_Score']<750 else 4)

    all_cols   = list(X.columns)
    row_df     = pd.DataFrame([row])[all_cols]
    row_scaled = pd.DataFrame(scaler.transform(row_df), columns=all_cols)

    foir_pct = row['FOIR'] * 100
    breakdown = [
        f"CIBIL Score: {applicant['CIBIL_Score']} — "
        f"{'Excellent' if applicant['CIBIL_Score']>=750 else 'Good' if applicant['CIBIL_Score']>=700 else 'Fair' if applicant['CIBIL_Score']>=650 else 'Poor'}",
        f"FOIR: {foir_pct:.1f}% — "
        f"{'Healthy (< 30%)' if foir_pct<30 else 'Moderate (30-50%)' if foir_pct<50 else 'High (> 50%) — risk factor'}",
        f"Total Income: Rs.{int(row['Total_Income']):,}/month",
        f"Approx EMI: Rs.{int(emi_approx):,}/month",
        f"Existing EMIs: Rs.{int(applicant['Existing_EMIs']):,}/month",
        f"Income-to-Loan Ratio: {row['Income_Loan_Ratio']:.2f}",
        f"Loan Type: {applicant['Loan_Type']}",
    ]

    if applicant['CIBIL_Score'] >= 650:
        X_input    = row_scaled[trad_features]
        prob       = float(trad_models["Logistic Regression"].predict_proba(X_input)[0][1])
        model_used = "Logistic Regression (Traditional)"
        breakdown.append("Routed to Traditional Model — CIBIL >= 650")
        breakdown.append("Gender & Married combined influence: < 0.3%")
    else:
        X_input    = row_scaled[alt_features]
        prob       = float(alt_models["Decision Tree"].predict_proba(X_input)[0][1])
        model_used = "Decision Tree (Alternative)"
        breakdown.append("Routed to Alternative Model — CIBIL < 650")
        breakdown.append("Gender & Married not used in this model")

    status = "approved" if prob>=0.55 else "review" if prob>=0.40 else "rejected"

    return {
        "status":     status,
        "probability": prob,
        "model_used":  model_used,
        "foir":        row['FOIR'],
        "breakdown":   breakdown
    }

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    from flask import jsonify
    return jsonify(predict_loan(request.get_json()))

if __name__ == '__main__':
    print("Starting Loan Approval Predictor V2...")
    print("Open: http://127.0.0.1:5000")
    app.run(debug=False)