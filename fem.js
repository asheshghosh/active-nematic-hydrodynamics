"use strict";

const DT = 0.12;
const HISTORY_LIMIT = 520;
const canvas = document.getElementById("field");
const ctx = canvas.getContext("2d", { alpha: false });
const historyCanvas = document.getElementById("history");
const historyCtx = historyCanvas.getContext("2d", { alpha: false });
const pressureCanvas = document.getElementById("pressure");
const pressureCtx = pressureCanvas.getContext("2d", { alpha: false });

const controls = {
  meshSize: document.getElementById("meshSize"),
  activityEnabled: document.getElementById("activityEnabled"),
  activity: document.getElementById("activity"),
  elastic: document.getElementById("elastic"),
  viscosity: document.getElementById("viscosity"),
  momentumAdvection: document.getElementById("momentumAdvection"),
  friction: document.getElementById("friction"),
  alignment: document.getElementById("alignment"),
  noise: document.getElementById("noise"),
  showMesh: document.getElementById("showMesh"),
  showDirectors: document.getElementById("showDirectors"),
  showVelocity: document.getElementById("showVelocity"),
  showDefects: document.getElementById("showDefects"),
  showPressure: document.getElementById("showPressure"),
};

let N = 80;
let size = N * N;
let image = ctx.createImageData(N, N);
let pressureImage = pressureCtx.createImageData(N, N);
let qxx;
let qxy;
let nextQxx;
let nextQxy;
let ux;
let uy;
let nextUx;
let nextUy;
let pressure;
let div;
let scratch;
let lapQxx;
let lapQxy;
let lapUx;
let lapUy;
let mass;
let adjacency;
let triangles;
let latestDefects = [];
let defectStats = { positive: 0, negative: 0, total: 0 };
let energy = 0;
let stepCount = 0;
let running = true;
let history = [];

function node(x, y) {
  return ((y + N) % N) * N + ((x + N) % N);
}

function randn() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function addEdge(weights, a, b, w) {
  weights[a].set(b, (weights[a].get(b) || 0) + w);
  weights[b].set(a, (weights[b].get(a) || 0) + w);
}

function buildMesh(nextN) {
  N = nextN;
  size = N * N;
  canvas.width = N;
  canvas.height = N;
  pressureCanvas.width = N;
  pressureCanvas.height = N;
  image = ctx.createImageData(N, N);
  pressureImage = pressureCtx.createImageData(N, N);

  qxx = new Float32Array(size);
  qxy = new Float32Array(size);
  nextQxx = new Float32Array(size);
  nextQxy = new Float32Array(size);
  ux = new Float32Array(size);
  uy = new Float32Array(size);
  nextUx = new Float32Array(size);
  nextUy = new Float32Array(size);
  pressure = new Float32Array(size);
  div = new Float32Array(size);
  scratch = new Float32Array(size);
  lapQxx = new Float32Array(size);
  lapQxy = new Float32Array(size);
  lapUx = new Float32Array(size);
  lapUy = new Float32Array(size);
  mass = new Float32Array(size);
  triangles = [];

  const weights = Array.from({ length: size }, () => new Map());
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const a = node(x, y);
      const b = node(x + 1, y);
      const c = node(x, y + 1);
      const d = node(x + 1, y + 1);
      triangles.push([a, b, c], [b, d, c]);
      mass[a] += 1 / 3;
      mass[b] += 2 / 3;
      mass[c] += 2 / 3;
      mass[d] += 1 / 3;
      addEdge(weights, a, b, 0.5);
      addEdge(weights, a, c, 0.5);
      addEdge(weights, b, d, 0.5);
      addEdge(weights, c, d, 0.5);
      addEdge(weights, b, c, 1.0);
    }
  }

  adjacency = weights.map((row) => Array.from(row, ([j, w]) => ({ j, w })));
  seed();
}

function seed(kind = "fresh") {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = node(x, y);
      const wave = 0.34 * Math.sin((2 * Math.PI * x) / N) * Math.cos((2 * Math.PI * y) / N);
      const theta = wave + (kind === "kick" ? 0.9 : 0.16) * randn();
      const s = 0.62 + 0.05 * randn();
      qxx[i] = s * Math.cos(2 * theta);
      qxy[i] = s * Math.sin(2 * theta);
      ux[i] = 0;
      uy[i] = 0;
      pressure[i] = 0;
    }
  }
  if (kind !== "kick") {
    stepCount = 0;
    history = [];
  }
}

function ddx(a, x, y) {
  return 0.5 * (a[node(x + 1, y)] - a[node(x - 1, y)]);
}

function ddy(a, x, y) {
  return 0.5 * (a[node(x, y + 1)] - a[node(x, y - 1)]);
}

function advect(a, x, y, vx, vy) {
  return vx * ddx(a, x, y) + vy * ddy(a, x, y);
}

function applyStiffness(src, dest) {
  dest.fill(0);
  for (let i = 0; i < size; i++) {
    let sum = 0;
    for (const edge of adjacency[i]) sum += edge.w * (src[edge.j] - src[i]);
    dest[i] = sum / Math.max(mass[i], 0.001);
  }
}

function zetaValue() {
  if (!controls.activityEnabled.checked) return 0;
  const direction = document.querySelector("input[name='activityType']:checked").value;
  return (direction === "extensile" ? 1 : -1) * Number(controls.activity.value);
}

function projectVelocity() {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = node(x, y);
      div[i] = ddx(nextUx, x, y) + ddy(nextUy, x, y);
      pressure[i] *= 0.82;
    }
  }

  const iterations = N >= 192 ? 10 : N >= 112 ? 14 : 20;
  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < size; i++) {
      let weighted = 0;
      let total = 0;
      for (const edge of adjacency[i]) {
        weighted += edge.w * pressure[edge.j];
        total += edge.w;
      }
      scratch[i] = (weighted - mass[i] * div[i]) / Math.max(total, 0.001);
    }
    pressure.set(scratch);
  }

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = node(x, y);
      ux[i] = nextUx[i] - ddx(pressure, x, y);
      uy[i] = nextUy[i] - ddy(pressure, x, y);
    }
  }
}

function simulate() {
  const zeta = zetaValue();
  const k = Number(controls.elastic.value);
  const viscosity = Number(controls.viscosity.value);
  const lambda = Number(controls.alignment.value);
  const friction = Number(controls.friction.value);
  const noise = Number(controls.noise.value);
  const advectMomentum = controls.momentumAdvection.checked;
  const gamma = 0.62;
  const targetS = 0.66;
  energy = 0;

  applyStiffness(qxx, lapQxx);
  applyStiffness(qxy, lapQxy);
  applyStiffness(ux, lapUx);
  applyStiffness(uy, lapUy);

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = node(x, y);
      const activeX = -zeta * (ddx(qxx, x, y) + ddy(qxy, x, y));
      const activeY = -zeta * (ddx(qxy, x, y) - ddy(qxx, x, y));
      const advX = advectMomentum ? advect(ux, x, y, ux[i], uy[i]) : 0;
      const advY = advectMomentum ? advect(uy, x, y, ux[i], uy[i]) : 0;
      nextUx[i] = ux[i] + DT * (activeX + viscosity * lapUx[i] - friction * ux[i] - advX);
      nextUy[i] = uy[i] + DT * (activeY + viscosity * lapUy[i] - friction * uy[i] - advY);
    }
  }

  projectVelocity();

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = node(x, y);
      const uxx = ddx(ux, x, y);
      const uxy = 0.5 * (ddy(ux, x, y) + ddx(uy, x, y));
      const omega = 0.5 * (ddx(uy, x, y) - ddy(ux, x, y));
      const s2 = qxx[i] * qxx[i] + qxy[i] * qxy[i];
      const bulk = 1.25 * (targetS * targetS - s2);
      const hxx = k * lapQxx[i] + bulk * qxx[i];
      const hxy = k * lapQxy[i] + bulk * qxy[i];
      const nudge = 0.014 * noise * randn();
      nextQxx[i] = qxx[i] + DT * (gamma * hxx - 2 * omega * qxy[i] + lambda * uxx - advect(qxx, x, y, ux[i], uy[i])) + nudge;
      nextQxy[i] = qxy[i] + DT * (gamma * hxy + 2 * omega * qxx[i] + lambda * uxy - advect(qxy, x, y, ux[i], uy[i])) + 0.6 * nudge;
      const mag = Math.hypot(nextQxx[i], nextQxy[i]);
      if (mag > 1.15) {
        nextQxx[i] *= 1.15 / mag;
        nextQxy[i] *= 1.15 / mag;
      }
      energy += 0.5 * k * (hxx * hxx + hxy * hxy) + 0.5 * (ux[i] * ux[i] + uy[i] * uy[i]);
    }
  }

  qxx.set(nextQxx);
  qxy.set(nextQxy);
  stepCount++;
}

function hueColor(v, order) {
  const t = Math.max(-1, Math.min(1, v));
  return [
    Math.round(28 + 188 * Math.max(0, t) + 54 * order),
    Math.round(48 + 132 * (1 - Math.abs(t)) + 60 * order),
    Math.round(66 + 190 * Math.max(0, -t) + 46 * order),
  ];
}

function pressureColor(value) {
  const t = Math.max(-1, Math.min(1, value));
  const neutral = [24, 29, 34];
  const target = t >= 0 ? [255, 111, 126] : [92, 164, 255];
  const a = Math.abs(t);
  return target.map((v, i) => Math.round(neutral[i] + (v - neutral[i]) * a));
}

function angleAt(x, y) {
  return 0.5 * Math.atan2(qxy[node(x, y)], qxx[node(x, y)]);
}

function wrapHalfTurn(a) {
  while (a > Math.PI / 2) a -= Math.PI;
  while (a < -Math.PI / 2) a += Math.PI;
  return a;
}

function findDefects() {
  const defects = [];
  let positive = 0;
  let negative = 0;
  const stride = N > 160 ? 2 : 1;
  for (let y = 0; y < N; y += stride) {
    for (let x = 0; x < N; x += stride) {
      const a0 = angleAt(x, y);
      const a1 = angleAt(x + 1, y);
      const a2 = angleAt(x + 1, y + 1);
      const a3 = angleAt(x, y + 1);
      const winding = wrapHalfTurn(a1 - a0) + wrapHalfTurn(a2 - a1) + wrapHalfTurn(a3 - a2) + wrapHalfTurn(a0 - a3);
      const charge = winding / (2 * Math.PI);
      if (Math.abs(charge) > 0.22) {
        defects.push({ x: x + 0.5, y: y + 0.5, charge });
        if (charge > 0) positive++;
        else negative++;
      }
    }
  }
  latestDefects = defects;
  defectStats = { positive, negative, total: defects.length };
}

function recordHistory() {
  const entry = {
    step: stepCount,
    energy: energy / size,
    positive: defectStats.positive,
    negative: defectStats.negative,
    total: defectStats.total,
  };
  const last = history[history.length - 1];
  if (!last || last.step !== entry.step) history.push(entry);
  if (history.length > HISTORY_LIMIT) history.shift();
}

function drawHistory() {
  const w = historyCanvas.width;
  const h = historyCanvas.height;
  historyCtx.fillStyle = "#0c0f12";
  historyCtx.fillRect(0, 0, w, h);
  historyCtx.strokeStyle = "rgba(255,255,255,0.1)";
  for (let y = 30; y < h; y += 30) {
    historyCtx.beginPath();
    historyCtx.moveTo(42, y);
    historyCtx.lineTo(w - 10, y);
    historyCtx.stroke();
  }
  historyCtx.fillStyle = "#94a2a8";
  historyCtx.font = "12px ui-sans-serif, system-ui";
  historyCtx.fillText("energy", 10, 22);
  historyCtx.fillText("+ defects", 10, 42);
  historyCtx.fillText("- defects", 10, 62);
  if (history.length < 2) return;
  const maxEnergy = Math.max(...history.map((d) => d.energy), 0.001);
  const maxDefects = Math.max(...history.map((d) => d.total), 1);
  drawSeries(history.map((d) => d.energy), (v) => h - 22 - (v / maxEnergy) * (h - 46), "#71d8c9");
  drawSeries(history.map((d) => d.positive), (v) => h - 22 - (v / maxDefects) * (h - 46), "#ffe780");
  drawSeries(history.map((d) => d.negative), (v) => h - 22 - (v / maxDefects) * (h - 46), "#ff6b7d");
}

function drawSeries(values, yFor, color) {
  const w = historyCanvas.width;
  historyCtx.strokeStyle = color;
  historyCtx.lineWidth = 2;
  historyCtx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const x = 44 + ((i + HISTORY_LIMIT - values.length) / (HISTORY_LIMIT - 1)) * (w - 58);
    const y = yFor(values[i]);
    if (i === 0) historyCtx.moveTo(x, y);
    else historyCtx.lineTo(x, y);
  }
  historyCtx.stroke();
}

function drawPressure() {
  if (!controls.showPressure.checked) {
    pressureCanvas.style.display = "none";
    return;
  }
  pressureCanvas.style.display = "block";
  let mean = 0;
  for (let i = 0; i < size; i++) mean += pressure[i];
  mean /= size;
  let scale = 0.0001;
  for (let i = 0; i < size; i++) scale = Math.max(scale, Math.abs(pressure[i] - mean));
  let p = 0;
  for (let i = 0; i < size; i++) {
    const [r, g, b] = pressureColor((pressure[i] - mean) / scale);
    pressureImage.data[p++] = r;
    pressureImage.data[p++] = g;
    pressureImage.data[p++] = b;
    pressureImage.data[p++] = 255;
  }
  pressureCtx.putImageData(pressureImage, 0, 0);
}

function drawMesh() {
  if (!controls.showMesh.checked) return;
  const stride = Math.max(1, Math.round(N / 48));
  ctx.strokeStyle = "rgba(238,247,242,0.12)";
  ctx.lineWidth = Math.max(0.06, N / 2200);
  for (let y = 0; y < N; y += stride) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(N, y);
    ctx.stroke();
  }
  for (let x = 0; x < N; x += stride) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, N);
    ctx.stroke();
  }
}

function drawDirectors() {
  if (!controls.showDirectors.checked) return;
  const stride = Math.max(5, Math.round(N / 25));
  ctx.strokeStyle = "rgba(238,247,242,0.62)";
  ctx.lineWidth = Math.max(0.12, N / 1100);
  ctx.lineCap = "round";
  for (let y = Math.floor(stride / 2); y < N; y += stride) {
    for (let x = Math.floor(stride / 2); x < N; x += stride) {
      const i = node(x, y);
      const theta = 0.5 * Math.atan2(qxy[i], qxx[i]);
      const s = 0.38 * stride + Math.hypot(qxx[i], qxy[i]);
      const dx = Math.cos(theta) * s;
      const dy = Math.sin(theta) * s;
      ctx.beginPath();
      ctx.moveTo(x - dx, y - dy);
      ctx.lineTo(x + dx, y + dy);
      ctx.stroke();
    }
  }
}

function drawVelocity() {
  if (!controls.showVelocity.checked) return;
  const stride = Math.max(8, Math.round(N / 13));
  ctx.strokeStyle = "rgba(113,216,201,0.58)";
  ctx.lineWidth = Math.max(0.12, N / 1300);
  for (let y = Math.floor(stride / 2); y < N; y += stride) {
    for (let x = Math.floor(stride / 2); x < N; x += stride) {
      const i = node(x, y);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 7 * ux[i], y + 7 * uy[i]);
      ctx.stroke();
    }
  }
}

function drawDefects() {
  if (!controls.showDefects.checked) return;
  for (const d of latestDefects) {
    ctx.beginPath();
    ctx.arc(d.x, d.y, Math.max(1.15, N / 78), 0, 2 * Math.PI);
    ctx.fillStyle = d.charge > 0 ? "rgba(255,231,128,0.95)" : "rgba(255,107,125,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 0.22;
    ctx.stroke();
  }
}

function drawField() {
  let p = 0;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = node(x, y);
      const vort = 0.5 * (ddx(uy, x, y) - ddy(ux, x, y));
      const order = Math.min(1, Math.hypot(qxx[i], qxy[i]));
      const [r, g, b] = hueColor(vort * 14, order);
      image.data[p++] = r;
      image.data[p++] = g;
      image.data[p++] = b;
      image.data[p++] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  ctx.save();
  ctx.scale(canvas.width / N, canvas.height / N);
  drawMesh();
  drawDirectors();
  drawVelocity();
  drawDefects();
  ctx.restore();
}

function draw() {
  findDefects();
  recordHistory();
  drawField();
  drawHistory();
  drawPressure();
  document.getElementById("steps").textContent = String(stepCount);
  document.getElementById("defects").textContent = `${defectStats.positive} / ${defectStats.negative}`;
  document.getElementById("energy").textContent = (energy / size).toFixed(3);
}

function syncOutputs() {
  document.getElementById("meshSizeValue").textContent = `${controls.meshSize.value} x ${controls.meshSize.value}`;
  for (const key of ["activity", "elastic", "viscosity", "alignment", "noise"]) {
    document.getElementById(`${key}Value`).textContent = Number(controls[key].value).toFixed(2);
  }
  document.getElementById("frictionValue").textContent = Number(controls.friction.value).toFixed(3);
}

function frame() {
  if (running) {
    const substeps = N >= 160 ? 1 : 2;
    for (let i = 0; i < substeps; i++) simulate();
  }
  draw();
  requestAnimationFrame(frame);
}

document.getElementById("toggle").addEventListener("click", (event) => {
  running = !running;
  event.currentTarget.textContent = running ? "Pause" : "Run";
});
document.getElementById("reset").addEventListener("click", () => seed());
document.getElementById("kick").addEventListener("click", () => seed("kick"));
controls.meshSize.addEventListener("change", () => {
  buildMesh(Number(controls.meshSize.value));
  syncOutputs();
});
for (const control of Object.values(controls)) control.addEventListener("input", syncOutputs);
for (const radio of document.querySelectorAll("input[name='activityType']")) radio.addEventListener("change", syncOutputs);

buildMesh(Number(controls.meshSize.value));
syncOutputs();
frame();
