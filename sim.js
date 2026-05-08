"use strict";

const DT = 0.18;
const HISTORY_LIMIT = 520;
const fieldCanvas = document.getElementById("field");
const fieldCtx = fieldCanvas.getContext("2d", { alpha: false });
const historyCanvas = document.getElementById("history");
const historyCtx = historyCanvas.getContext("2d", { alpha: false });
const pressureCanvas = document.getElementById("pressure");
const pressureCtx = pressureCanvas.getContext("2d", { alpha: false });

const controls = {
  gridSize: document.getElementById("gridSize"),
  activityEnabled: document.getElementById("activityEnabled"),
  activity: document.getElementById("activity"),
  elastic: document.getElementById("elastic"),
  viscosity: document.getElementById("viscosity"),
  momentumAdvection: document.getElementById("momentumAdvection"),
  friction: document.getElementById("friction"),
  alignment: document.getElementById("alignment"),
  noise: document.getElementById("noise"),
  showDirectors: document.getElementById("showDirectors"),
  showVelocity: document.getElementById("showVelocity"),
  showDefects: document.getElementById("showDefects"),
  showPressure: document.getElementById("showPressure"),
};

let N = 112;
let size = N * N;
let image = fieldCtx.createImageData(N, N);
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

let running = true;
let stepCount = 0;
let defectStats = { total: 0, positive: 0, negative: 0 };
let energy = 0;
let latestDefects = [];
let history = [];

function allocateGrid(nextN) {
  N = nextN;
  size = N * N;
  fieldCanvas.width = N;
  fieldCanvas.height = N;
  pressureCanvas.width = N;
  pressureCanvas.height = N;
  image = fieldCtx.createImageData(N, N);
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
  seed();
}

function id(x, y) {
  return ((y + N) % N) * N + ((x + N) % N);
}

function randn() {
  let u = 1 - Math.random();
  let v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function seed(kind = "fresh") {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = id(x, y);
      const wave = 0.35 * Math.sin((2 * Math.PI * x) / N) * Math.cos((2 * Math.PI * y) / N);
      const theta = wave + (kind === "kick" ? 0.9 : 0.18) * randn();
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
  energy = 0;
  latestDefects = [];
  defectStats = { total: 0, positive: 0, negative: 0 };
}

function lap(a, x, y, i) {
  return a[id(x + 1, y)] + a[id(x - 1, y)] + a[id(x, y + 1)] + a[id(x, y - 1)] - 4 * a[i];
}

function ddx(a, x, y) {
  return 0.5 * (a[id(x + 1, y)] - a[id(x - 1, y)]);
}

function ddy(a, x, y) {
  return 0.5 * (a[id(x, y + 1)] - a[id(x, y - 1)]);
}

function advect(a, x, y, vx, vy) {
  return vx * ddx(a, x, y) + vy * ddy(a, x, y);
}

function activityZeta() {
  if (!controls.activityEnabled.checked) return 0;
  const direction = document.querySelector("input[name='activityType']:checked").value;
  const sign = direction === "extensile" ? 1 : -1;
  return sign * Number(controls.activity.value);
}

function projectVelocity() {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = id(x, y);
      div[i] = 0.5 * (nextUx[id(x + 1, y)] - nextUx[id(x - 1, y)] + nextUy[id(x, y + 1)] - nextUy[id(x, y - 1)]);
      pressure[i] *= 0.82;
    }
  }

  const pressureIterations = N >= 768 ? 8 : N >= 384 ? 12 : N >= 192 ? 16 : 22;
  for (let iter = 0; iter < pressureIterations; iter++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = id(x, y);
        scratch[i] = 0.25 * (
          pressure[id(x + 1, y)] +
          pressure[id(x - 1, y)] +
          pressure[id(x, y + 1)] +
          pressure[id(x, y - 1)] -
          div[i]
        );
      }
    }
    pressure.set(scratch);
  }

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = id(x, y);
      ux[i] = nextUx[i] - ddx(pressure, x, y);
      uy[i] = nextUy[i] - ddy(pressure, x, y);
    }
  }
}

function simulate() {
  const zeta = activityZeta();
  const k = Number(controls.elastic.value);
  const viscosity = Number(controls.viscosity.value);
  const lambda = Number(controls.alignment.value);
  const noise = Number(controls.noise.value);
  const advectMomentum = controls.momentumAdvection.checked;
  const friction = Number(controls.friction.value);
  const gamma = 0.68;
  const targetS = 0.66;
  energy = 0;

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = id(x, y);
      const forceX = -zeta * (ddx(qxx, x, y) + ddy(qxy, x, y));
      const forceY = -zeta * (ddx(qxy, x, y) - ddy(qxx, x, y));
      const viscX = viscosity * lap(ux, x, y, i);
      const viscY = viscosity * lap(uy, x, y, i);
      const advX = advectMomentum ? advect(ux, x, y, ux[i], uy[i]) : 0;
      const advY = advectMomentum ? advect(uy, x, y, ux[i], uy[i]) : 0;
      nextUx[i] = ux[i] + DT * (forceX + viscX - friction * ux[i] - advX);
      nextUy[i] = uy[i] + DT * (forceY + viscY - friction * uy[i] - advY);
    }
  }

  projectVelocity();

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = id(x, y);
      const uxx = ddx(ux, x, y);
      const uxy = 0.5 * (ddy(ux, x, y) + ddx(uy, x, y));
      const omega = 0.5 * (ddx(uy, x, y) - ddy(ux, x, y));
      const s2 = qxx[i] * qxx[i] + qxy[i] * qxy[i];
      const bulk = 1.35 * (targetS * targetS - s2);
      const hxx = k * lap(qxx, x, y, i) + bulk * qxx[i];
      const hxy = k * lap(qxy, x, y, i) + bulk * qxy[i];
      const rotateXx = -2 * omega * qxy[i];
      const rotateXy = 2 * omega * qxx[i];
      const alignXx = lambda * uxx;
      const alignXy = lambda * uxy;
      const nudge = noise * 0.015 * randn();

      nextQxx[i] = qxx[i] + DT * (gamma * hxx + rotateXx + alignXx - advect(qxx, x, y, ux[i], uy[i])) + nudge;
      nextQxy[i] = qxy[i] + DT * (gamma * hxy + rotateXy + alignXy - advect(qxy, x, y, ux[i], uy[i])) + nudge * 0.6;

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
  const r = Math.round(28 + 188 * Math.max(0, t) + 54 * order);
  const g = Math.round(48 + 132 * (1 - Math.abs(t)) + 60 * order);
  const b = Math.round(66 + 190 * Math.max(0, -t) + 46 * order);
  return [r, g, b];
}

function angleAt(x, y) {
  return 0.5 * Math.atan2(qxy[id(x, y)], qxx[id(x, y)]);
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
  for (let y = 0; y < N; y += 2) {
    for (let x = 0; x < N; x += 2) {
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
  defectStats = { total: defects.length, positive, negative };
  latestDefects = defects;
  return defects;
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
  historyCtx.clearRect(0, 0, w, h);
  historyCtx.fillStyle = "#0c0f12";
  historyCtx.fillRect(0, 0, w, h);
  historyCtx.strokeStyle = "rgba(255,255,255,0.1)";
  historyCtx.lineWidth = 1;
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
  historyCtx.fillText("steps", w - 50, h - 10);

  if (history.length < 2) return;
  const maxEnergy = Math.max(...history.map((d) => d.energy), 0.001);
  const maxDefects = Math.max(...history.map((d) => d.total), 1);
  const xFor = (index) => 44 + (index / (HISTORY_LIMIT - 1)) * (w - 58);
  const yEnergy = (value) => h - 22 - (value / maxEnergy) * (h - 46);
  const yDefect = (value) => h - 22 - (value / maxDefects) * (h - 46);

  drawSeries(history.map((d) => d.energy), xFor, yEnergy, "#71d8c9");
  drawSeries(history.map((d) => d.positive), xFor, yDefect, "#ffe780");
  drawSeries(history.map((d) => d.negative), xFor, yDefect, "#ff6b7d");
}

function drawSeries(values, xFor, yFor, color) {
  historyCtx.strokeStyle = color;
  historyCtx.lineWidth = 2;
  historyCtx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const x = xFor(i + HISTORY_LIMIT - values.length);
    const y = yFor(values[i]);
    if (i === 0) historyCtx.moveTo(x, y);
    else historyCtx.lineTo(x, y);
  }
  historyCtx.stroke();
}

function pressureColor(value) {
  const t = Math.max(-1, Math.min(1, value));
  const neutral = [24, 29, 34];
  const positive = [255, 111, 126];
  const negative = [92, 164, 255];
  const target = t >= 0 ? positive : negative;
  const a = Math.abs(t);
  return [
    Math.round(neutral[0] + (target[0] - neutral[0]) * a),
    Math.round(neutral[1] + (target[1] - neutral[1]) * a),
    Math.round(neutral[2] + (target[2] - neutral[2]) * a),
  ];
}

function directorAt(x, y, reference = null) {
  const i = id(Math.round(x), Math.round(y));
  let dx = Math.cos(0.5 * Math.atan2(qxy[i], qxx[i]));
  let dy = Math.sin(0.5 * Math.atan2(qxy[i], qxx[i]));
  if (reference && dx * reference.x + dy * reference.y < 0) {
    dx = -dx;
    dy = -dy;
  }
  return { x: dx, y: dy };
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
  for (let i = 0; i < size; i++) {
    scale = Math.max(scale, Math.abs(pressure[i] - mean));
  }

  let p = 0;
  for (let i = 0; i < size; i++) {
    const [r, g, b] = pressureColor((pressure[i] - mean) / scale);
    pressureImage.data[p++] = r;
    pressureImage.data[p++] = g;
    pressureImage.data[p++] = b;
    pressureImage.data[p++] = 255;
  }
  pressureCtx.putImageData(pressureImage, 0, 0);

  pressureCtx.save();
  pressureCtx.scale(pressureCanvas.width / N, pressureCanvas.height / N);
  pressureCtx.fillStyle = "rgba(12, 15, 18, 0.72)";
  pressureCtx.fillRect(0, 0, N, 14);
  pressureCtx.fillStyle = "#eef3f5";
  pressureCtx.font = "5px ui-sans-serif, system-ui";
  pressureCtx.fillText("pressure", 4, 9);
  pressureCtx.fillStyle = "#94a2a8";
  pressureCtx.fillText(`max |p| ${scale.toFixed(3)}`, 34, 9);
  pressureCtx.restore();
}

function drawDirectorRods() {
  const stride = Math.max(4, Math.round(N / 23));
  fieldCtx.strokeStyle = "rgba(238, 247, 242, 0.68)";
  fieldCtx.lineWidth = 0.13;
  for (let y = Math.floor(stride / 2); y < N; y += stride) {
    for (let x = Math.floor(stride / 2); x < N; x += stride) {
      const i = id(x, y);
      const theta = 0.5 * Math.atan2(qxy[i], qxx[i]);
      const s = 0.28 * stride + 1.4 * Math.hypot(qxx[i], qxy[i]);
      const dx = Math.cos(theta) * s;
      const dy = Math.sin(theta) * s;
      fieldCtx.beginPath();
      fieldCtx.moveTo(x - dx, y - dy);
      fieldCtx.lineTo(x + dx, y + dy);
      fieldCtx.stroke();
    }
  }
}

function drawDirectorStreamlines() {
  const stride = Math.max(10, Math.round(N / 15));
  const step = Math.max(0.85, N / 180);
  const steps = Math.max(8, Math.round(N / 32));
  fieldCtx.strokeStyle = "rgba(238, 247, 242, 0.58)";
  fieldCtx.lineWidth = Math.max(0.12, N / 900);

  for (let y = Math.floor(stride / 2); y < N; y += stride) {
    for (let x = Math.floor(stride / 2); x < N; x += stride) {
      const points = [{ x, y }];

      for (const sign of [1, -1]) {
        let px = x;
        let py = y;
        let dir = directorAt(px, py);
        const branch = [];
        for (let s = 0; s < steps; s++) {
          dir = directorAt(px, py, dir);
          px = (px + sign * dir.x * step + N) % N;
          py = (py + sign * dir.y * step + N) % N;
          if (Math.abs(px - x) > N * 0.48 || Math.abs(py - y) > N * 0.48) break;
          branch.push({ x: px, y: py });
        }
        if (sign > 0) points.push(...branch);
        else points.unshift(...branch.reverse());
      }

      fieldCtx.beginPath();
      fieldCtx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) fieldCtx.lineTo(points[i].x, points[i].y);
      fieldCtx.stroke();
    }
  }
}

function drawDirectorStencil() {
  const stride = Math.max(3, Math.round(N / 38));
  fieldCtx.strokeStyle = "rgba(238, 247, 242, 0.42)";
  fieldCtx.lineWidth = Math.max(0.08, N / 1400);
  for (let y = Math.floor(stride / 2); y < N; y += stride) {
    for (let x = Math.floor(stride / 2); x < N; x += stride) {
      const i = id(x, y);
      const theta = 0.5 * Math.atan2(qxy[i], qxx[i]);
      const order = Math.min(1, Math.hypot(qxx[i], qxy[i]));
      const s = (0.5 + 0.55 * order) * stride;
      const dx = Math.cos(theta) * s;
      const dy = Math.sin(theta) * s;
      fieldCtx.beginPath();
      fieldCtx.moveTo(x - dx, y - dy);
      fieldCtx.lineTo(x + dx, y + dy);
      fieldCtx.stroke();
    }
  }
}

function drawDirectors() {
  const mode = document.querySelector("input[name='directorMode']:checked").value;
  if (mode === "streamlines") drawDirectorStreamlines();
  else if (mode === "stencil") drawDirectorStencil();
  else drawDirectorRods();
}

function drawField() {
  let p = 0;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = id(x, y);
      const vort = 0.5 * (ddx(uy, x, y) - ddy(ux, x, y));
      const order = Math.min(1, Math.hypot(qxx[i], qxy[i]));
      const [r, g, b] = hueColor(vort * 14, order);
      image.data[p++] = r;
      image.data[p++] = g;
      image.data[p++] = b;
      image.data[p++] = 255;
    }
  }

  fieldCtx.putImageData(image, 0, 0);
  fieldCtx.save();
  fieldCtx.scale(fieldCanvas.width / N, fieldCanvas.height / N);
  fieldCtx.lineCap = "round";

  if (controls.showDirectors.checked) {
    drawDirectors();
  }

  if (controls.showVelocity.checked) {
    const stride = Math.max(8, Math.round(N / 12));
    fieldCtx.strokeStyle = "rgba(113, 216, 201, 0.58)";
    fieldCtx.lineWidth = 0.16;
    for (let y = Math.floor(stride / 2); y < N; y += stride) {
      for (let x = Math.floor(stride / 2); x < N; x += stride) {
        const i = id(x, y);
        const vx = ux[i] * 7;
        const vy = uy[i] * 7;
        fieldCtx.beginPath();
        fieldCtx.moveTo(x, y);
        fieldCtx.lineTo(x + vx, y + vy);
        fieldCtx.stroke();
      }
    }
  }

  if (controls.showDefects.checked) {
    for (const d of latestDefects) {
      fieldCtx.beginPath();
      fieldCtx.arc(d.x, d.y, Math.max(1.15, N / 78), 0, Math.PI * 2);
      fieldCtx.fillStyle = d.charge > 0 ? "rgba(255, 231, 128, 0.95)" : "rgba(255, 107, 125, 0.95)";
      fieldCtx.fill();
      fieldCtx.strokeStyle = "rgba(0, 0, 0, 0.55)";
      fieldCtx.lineWidth = 0.22;
      fieldCtx.stroke();
    }
  }

  fieldCtx.restore();
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
  const selectedGrid = Number(controls.gridSize.value);
  document.getElementById("gridSizeValue").textContent = `${selectedGrid} x ${selectedGrid}`;
  for (const key of ["activity", "elastic", "viscosity", "alignment", "noise"]) {
    document.getElementById(`${key}Value`).textContent = Number(controls[key].value).toFixed(2);
  }
  document.getElementById("frictionValue").textContent = Number(controls.friction.value).toFixed(3);
}

function frame() {
  if (running) {
    const substeps = N >= 384 ? 1 : N > 128 ? 2 : 3;
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
controls.gridSize.addEventListener("change", () => {
  allocateGrid(Number(controls.gridSize.value));
  syncOutputs();
});
for (const control of Object.values(controls)) {
  control.addEventListener("input", syncOutputs);
}
for (const radio of document.querySelectorAll("input[name='activityType']")) {
  radio.addEventListener("change", syncOutputs);
}

allocateGrid(Number(controls.gridSize.value));
syncOutputs();
frame();
