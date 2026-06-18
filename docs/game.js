/* Air hockey — browser port of the Python physics + trajectory predictor + scripted bot.
 * Runs fully client-side (static GitHub Pages, no backend).
 *
 * Coordinates match the Python project: x in [0, L] (long axis), y in [0, W].
 * Human = LEFT (blue) mallet, defends x=0. Bot = RIGHT (red) mallet, defends x=L.
 */

// ---------------------------------------------------------------- config (TableConfig)
const CFG = {
  length: 2.0, width: 1.0, goalWidth: 0.30,
  puckRadius: 0.03, malletRadius: 0.05,
  puckFriction: 0.20, wallE: 0.90, malletE: 0.95, puckMaxSpeed: 8.0,
  malletMaxSpeed: 4.0, malletMaxAccel: 40.0,
  physicsDt: 1 / 200, decisionDt: 1 / 10,
};
CFG.halfX = CFG.length / 2;
CFG.goalYmin = (CFG.width - CFG.goalWidth) / 2;
CFG.goalYmax = (CFG.width + CFG.goalWidth) / 2;
CFG.puckYlo = CFG.puckRadius;
CFG.puckYhi = CFG.width - CFG.puckRadius;

const GOAL_NONE = 0, GOAL_LEFT = 1, GOAL_RIGHT = 2;
const WIN_SCORE = 7;

const DIFFICULTY = {
  easy:   { speed: 2.6, aggression: 0.7, react: 0.18 },
  normal: { speed: 4.0, aggression: 1.0, react: 0.10 },
  hard:   { speed: 4.0, aggression: 1.25, react: 0.0 },
  rl:     { speed: 4.0, aggression: 1.0,  react: 0.0, rl: true }, // trained SAC policy (policy.js)
};

// ---------------------------------------------------------------- small vec helpers
const v = (x, y) => ({ x, y });
const sub = (a, b) => v(a.x - b.x, a.y - b.y);
const add = (a, b) => v(a.x + b.x, a.y + b.y);
const scale = (a, s) => v(a.x * s, a.y * s);
const len = (a) => Math.hypot(a.x, a.y);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// ---------------------------------------------------------------- predictor (mirror unfolding)
function fold(value, lo, hi) {
  const span = hi - lo;
  if (span <= 0) return lo;
  const z = ((value - lo) % (2 * span) + 2 * span) % (2 * span);
  return z < span ? lo + z : lo + (2 * span - z);
}

function intercept(pos, vel, xLine) {
  if (Math.abs(vel.x) < 1e-6) return null;
  const t = (xLine - pos.x) / vel.x;
  if (t <= 0) return null;
  return { y: fold(pos.y + vel.y * t, CFG.puckYlo, CFG.puckYhi), t };
}

function aimPoint(puckPos, attackLeftGoal) {
  const goalX = attackLeftGoal ? 0.0 : CFG.length;
  return v(goalX, CFG.width / 2);
}

// ---------------------------------------------------------------- physics
const state = {
  puckPos: v(CFG.length / 2, CFG.width / 2),
  puckVel: v(0, 0),
  malletPos: [v(CFG.length * 0.15, CFG.width / 2), v(CFG.length * 0.85, CFG.width / 2)],
  malletVel: [v(0, 0), v(0, 0)],
  lastTouch: -1,
};

function malletXBounds(m) {
  const r = CFG.malletRadius;
  return m === 0 ? [r, CFG.halfX] : [CFG.halfX, CFG.length - r];
}

function homePos(m) {
  return v(m === 0 ? CFG.length * 0.15 : CFG.length * 0.85, CFG.width / 2);
}

function serve(serveTo) {
  state.puckPos = v(CFG.length / 2, CFG.width / 2);
  state.malletPos = [homePos(0), homePos(1)];
  state.malletVel = [v(0, 0), v(0, 0)];
  const speed = 1.5 + Math.random() * 2.5;
  let angle;
  if (serveTo === 0) angle = Math.PI + (Math.random() - 0.5) * 1.2;
  else if (serveTo === 1) angle = (Math.random() - 0.5) * 1.2;
  else angle = Math.random() * 2 * Math.PI - Math.PI;
  state.puckVel = v(Math.cos(angle) * speed, Math.sin(angle) * speed);
  state.lastTouch = -1;
}

function integrateMallets(targets, dt) {
  for (let m = 0; m < 2; m++) {
    let t = targets[m];
    const tn = len(t);
    if (tn > CFG.malletMaxSpeed) t = scale(t, CFG.malletMaxSpeed / tn);
    let dv = sub(t, state.malletVel[m]);
    const dvm = len(dv);
    const maxDv = CFG.malletMaxAccel * dt;
    if (dvm > maxDv) dv = scale(dv, maxDv / dvm);
    state.malletVel[m] = add(state.malletVel[m], dv);
    let np = add(state.malletPos[m], scale(state.malletVel[m], dt));

    const [xlo, xhi] = malletXBounds(m);
    const ylo = CFG.malletRadius, yhi = CFG.width - CFG.malletRadius;
    if (np.x < xlo) { np.x = xlo; state.malletVel[m].x = 0; }
    else if (np.x > xhi) { np.x = xhi; state.malletVel[m].x = 0; }
    if (np.y < ylo) { np.y = ylo; state.malletVel[m].y = 0; }
    else if (np.y > yhi) { np.y = yhi; state.malletVel[m].y = 0; }
    state.malletPos[m] = np;
  }
}

function malletCollision(m) {
  const rsum = CFG.puckRadius + CFG.malletRadius;
  const delta = sub(state.puckPos, state.malletPos[m]);
  const dist = len(delta);
  if (dist >= rsum || dist < 1e-9) {
    if (dist < 1e-9) state.puckPos = add(state.malletPos[m], v(rsum, 0));
    return;
  }
  const n = scale(delta, 1 / dist);
  const vrel = sub(state.puckVel, state.malletVel[m]);
  const vn = vrel.x * n.x + vrel.y * n.y;
  if (vn < 0) state.puckVel = sub(state.puckVel, scale(n, (1 + CFG.malletE) * vn));
  state.puckPos = add(state.malletPos[m], scale(n, rsum));
  state.lastTouch = m;
}

function physicsStep(target0, target1) {
  const dt = CFG.physicsDt;
  integrateMallets([target0, target1], dt);
  state.puckPos = add(state.puckPos, scale(state.puckVel, dt));

  // y walls
  const r = CFG.puckRadius;
  if (state.puckPos.y < r) { state.puckPos.y = r; state.puckVel.y = Math.abs(state.puckVel.y) * CFG.wallE; }
  else if (state.puckPos.y > CFG.width - r) { state.puckPos.y = CFG.width - r; state.puckVel.y = -Math.abs(state.puckVel.y) * CFG.wallE; }

  malletCollision(0);
  malletCollision(1);

  // x walls + goals
  let goal = GOAL_NONE;
  const x = state.puckPos.x, y = state.puckPos.y;
  const inGoalY = y > CFG.goalYmin && y < CFG.goalYmax;
  if (x < r) {
    if (inGoalY) { if (x <= 0) goal = GOAL_LEFT; }
    else { state.puckPos.x = r; state.puckVel.x = Math.abs(state.puckVel.x) * CFG.wallE; }
  } else if (x > CFG.length - r) {
    if (inGoalY) { if (x >= CFG.length) goal = GOAL_RIGHT; }
    else { state.puckPos.x = CFG.length - r; state.puckVel.x = -Math.abs(state.puckVel.x) * CFG.wallE; }
  }

  // friction + speed clamp
  const damp = Math.exp(-CFG.puckFriction * dt);
  state.puckVel = scale(state.puckVel, damp);
  const spd = len(state.puckVel);
  if (spd > CFG.puckMaxSpeed) state.puckVel = scale(state.puckVel, CFG.puckMaxSpeed / spd);
  return goal;
}

// ---------------------------------------------------------------- RL bot (ported SAC actor)
// Runs the trained policy client-side: a tiny MLP (16 -> [256,256] -> 2, tanh) whose
// weights ship in policy.js (generated by scripts/export_policy.py). The policy was
// trained as the LEFT attacker, so — exactly like airhockey.opponents.PolicyOpponent —
// we reflect the world in x to make mallet 1 (right) look like a left attacker, build
// the same 16-d observation as airhockey.env.build_observation, run the net, then
// un-mirror the action back to a world-frame target velocity.

const relu = (x) => (x > 0 ? x : 0);

function mlpForward(obs) {
  let x = obs;
  for (const layer of window.RL_POLICY.layers) {
    const w = layer.w, b = layer.b, out = new Array(b.length);
    for (let i = 0; i < b.length; i++) {
      const row = w[i];
      let s = b[i];
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      out[i] = layer.act === "relu" ? relu(s) : layer.act === "tanh" ? Math.tanh(s) : s;
    }
    x = out;
  }
  return x; // act_dim outputs, already tanh-squashed (deterministic action)
}

const normPos = (p) => [(p.x / CFG.length) * 2 - 1, (p.y / CFG.width) * 2 - 1];
const clip1 = (z) => clamp(z, -1, 1);

// Short-horizon kinematic lookahead with y-wall folding (predictor.position_at).
function positionAt(pos, vel, t) {
  const x = clamp(pos.x + vel.x * t, CFG.puckRadius, CFG.length - CFG.puckRadius);
  const y = fold(pos.y + vel.y * t, CFG.puckYlo, CFG.puckYhi);
  return v(x, y);
}

// Must stay identical to airhockey.env.build_observation (left-attacker frame).
function buildObservation(puckPos, puckVel, selfPos, selfVel, oppPos, oppVel) {
  const vmax = CFG.malletMaxSpeed, pvmax = CFG.puckMaxSpeed;
  const ic = intercept(puckPos, puckVel, CFG.length * 0.12);
  const yInt = ic ? ic.y : puckPos.y;
  const t = ic ? ic.t : 1.5;
  const tNorm = clamp(t / 1.5, 0, 1);
  const ahead = positionAt(puckPos, puckVel, 0.2);
  return [
    ...normPos(puckPos),
    clip1(puckVel.x / pvmax), clip1(puckVel.y / pvmax),
    ...normPos(selfPos),
    clip1(selfVel.x / vmax), clip1(selfVel.y / vmax),
    ...normPos(oppPos),
    clip1(oppVel.x / vmax), clip1(oppVel.y / vmax),
    (yInt / CFG.width) * 2 - 1, tNorm * 2 - 1,
    ...normPos(ahead),
  ];
}

function rlBot() {
  if (!window.RL_POLICY) return scriptedBot(DIFFICULTY.hard); // graceful fallback
  const L = CFG.length;
  const mx = (p) => v(L - p.x, p.y);   // mirror position in x
  const mv = (a) => v(-a.x, a.y);      // mirror velocity in x
  const obs = buildObservation(
    mx(state.puckPos), mv(state.puckVel),
    mx(state.malletPos[1]), mv(state.malletVel[1]),   // mallet 1 -> "self" (left attacker)
    mx(state.malletPos[0]), mv(state.malletVel[0]),   // mallet 0 -> opponent
  );
  const a = mlpForward(obs);  // target velocity in the mirrored frame, units of [-1,1]
  // Un-mirror: scale to a world target velocity and flip x back.
  return v(-a[0] * CFG.malletMaxSpeed, a[1] * CFG.malletMaxSpeed);
}

// ---------------------------------------------------------------- scripted bot (mallet 1, defends x=L)
function scriptedBot(diff) {
  const puck = state.puckPos, pvel = state.puckVel, mpos = state.malletPos[1];
  const attackLeftGoal = true;
  const defenseX = CFG.length * 0.82;
  const onOurSide = puck.x > CFG.halfX;
  const incoming = pvel.x > 0.2;
  const maxV = diff.speed;
  const pr = CFG.puckRadius, mr = CFG.malletRadius;

  let target;
  if (incoming) {
    const ic = intercept(puck, pvel, defenseX);
    target = v(defenseX, ic ? ic.y : puck.y);
  } else if (onOurSide) {
    let aimDir = sub(aimPoint(puck, attackLeftGoal), puck);
    aimDir = scale(aimDir, 1 / (len(aimDir) + 1e-9));
    const behind = sub(puck, scale(aimDir, pr + mr));
    if (len(sub(behind, mpos)) < (pr + mr) * 1.2) {
      return scale(aimDir, maxV * diff.aggression);   // strike through the puck
    }
    target = behind;
  } else {
    const restX = CFG.length * 0.86;
    target = v(restX, clamp(puck.y, CFG.width * 0.3, CFG.width * 0.7));
  }

  let vel = scale(sub(target, mpos), 8.0);
  const spd = len(vel);
  if (spd > maxV) vel = scale(vel, maxV / spd);
  return vel;
}

// ---------------------------------------------------------------- rendering
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
let PPM = 1, OX = 0, OY = 0;

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  PPM = rect.width / CFG.length;
  OX = 0; OY = 0;
}
window.addEventListener("resize", resize);

const toPx = (p) => [OX + p.x * PPM, OY + (CFG.width - p.y) * PPM];

function draw() {
  const W = CFG.length * PPM, H = CFG.width * PPM;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#16243a"; ctx.fillRect(0, 0, W, H);

  // center line + circle
  ctx.strokeStyle = "#3a5582"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
  ctx.beginPath(); ctx.arc(W / 2, H / 2, 0.18 * PPM, 0, 2 * Math.PI); ctx.stroke();

  // goals
  ctx.strokeStyle = "#ebc83c"; ctx.lineWidth = 6;
  for (const gx of [0, CFG.length]) {
    const [, y1] = toPx(v(gx, CFG.goalYmax));
    const [, y2] = toPx(v(gx, CFG.goalYmin));
    const px = gx === 0 ? 3 : W - 3;
    ctx.beginPath(); ctx.moveTo(px, y1); ctx.lineTo(px, y2); ctx.stroke();
  }

  // predicted trajectory
  if (showPred.checked) {
    ctx.strokeStyle = "rgba(120,255,160,0.55)"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= 24; i++) {
      const t = (0.8 * i) / 24;
      const x = clamp(state.puckPos.x + state.puckVel.x * t, CFG.puckRadius, CFG.length - CFG.puckRadius);
      const y = fold(state.puckPos.y + state.puckVel.y * t, CFG.puckYlo, CFG.puckYhi);
      const [px, py] = toPx(v(x, y));
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  const disc = (p, rad, fill) => {
    const [px, py] = toPx(p);
    ctx.beginPath(); ctx.arc(px, py, rad * PPM, 0, 2 * Math.PI);
    ctx.fillStyle = fill; ctx.fill();
  };
  disc(state.malletPos[0], CFG.malletRadius, "#46c8ff");
  disc(state.malletPos[1], CFG.malletRadius, "#ff5a5a");
  disc(state.puckPos, CFG.puckRadius, "#f2f2f6");
}

// ---------------------------------------------------------------- input (human = mallet 0)
let humanTarget = v(CFG.length * 0.15, CFG.width / 2);

function pointerToWorld(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const wx = (clientX - rect.left) / rect.width * CFG.length;
  const wy = CFG.width - (clientY - rect.top) / rect.height * CFG.width;
  const [xlo, xhi] = malletXBounds(0);
  humanTarget = v(clamp(wx, xlo, xhi), clamp(wy, CFG.malletRadius, CFG.width - CFG.malletRadius));
}
canvas.addEventListener("pointermove", (e) => { pointerToWorld(e.clientX, e.clientY); });
canvas.addEventListener("pointerdown", (e) => { pointerToWorld(e.clientX, e.clientY); });

// ---------------------------------------------------------------- match loop
const scoreYouEl = document.getElementById("scoreYou");
const scoreBotEl = document.getElementById("scoreBot");
const difficultyEl = document.getElementById("difficulty");
const showPred = document.getElementById("showPred");
const overlay = document.getElementById("overlay");
const overlayTitle = document.getElementById("overlayTitle");

let score = [0, 0];
let botTarget = v(0, 0);
let decisionAccum = 0;
let reactBuffer = [];      // simple reaction-delay queue of bot targets
let paused = false;
let last = performance.now();

function setScore() { scoreYouEl.textContent = score[0]; scoreBotEl.textContent = score[1]; }

function onGoal(goal) {
  if (goal === GOAL_RIGHT) { score[0]++; serve(0); }       // human scored
  else if (goal === GOAL_LEFT) { score[1]++; serve(1); }    // bot scored
  setScore();
  reactBuffer = [];
  if (score[0] >= WIN_SCORE || score[1] >= WIN_SCORE) endMatch();
}

function endMatch() {
  paused = true;
  overlayTitle.textContent = score[0] > score[1] ? "You win! 🏆" : "Bot wins 🤖";
  overlay.classList.remove("hidden");
}

function newMatch() {
  score = [0, 0]; setScore();
  overlay.classList.add("hidden");
  serve(Math.random() < 0.5 ? 0 : 1);
  reactBuffer = []; paused = false; last = performance.now();
}

function frame(now) {
  let dt = (now - last) / 1000;
  last = now;
  dt = Math.min(dt, 0.05);   // avoid huge catch-up steps after a tab switch

  if (!paused) {
    const diff = DIFFICULTY[difficultyEl.value];
    decisionAccum += dt;
    // Bot decides at 10 Hz; a small reaction delay makes lower difficulties beatable.
    if (decisionAccum >= CFG.decisionDt) {
      decisionAccum = 0;
      const reactSteps = Math.max(0, Math.round(diff.react / CFG.decisionDt));
      reactBuffer.push(diff.rl ? rlBot() : scriptedBot(diff));
      while (reactBuffer.length > reactSteps + 1) reactBuffer.shift();
      botTarget = reactBuffer[0];
    }

    // Step physics at fixed 200 Hz, holding both targets between decisions.
    let acc = dt;
    while (acc > 0) {
      const step = Math.min(CFG.physicsDt, acc);
      // human follows the pointer with a stiff controller (also accel-limited in physics)
      let humanVel = scale(sub(humanTarget, state.malletPos[0]), 18.0);
      const hs = len(humanVel);
      if (hs > CFG.malletMaxSpeed) humanVel = scale(humanVel, CFG.malletMaxSpeed / hs);
      const goal = physicsStep(humanVel, botTarget || v(0, 0));
      acc -= step;
      if (goal !== GOAL_NONE) { onGoal(goal); break; }
    }
  }

  draw();
  requestAnimationFrame(frame);
}

document.getElementById("reset").addEventListener("click", newMatch);
document.getElementById("playAgain").addEventListener("click", newMatch);

resize();
serve(Math.random() < 0.5 ? 0 : 1);
setScore();
requestAnimationFrame(frame);
