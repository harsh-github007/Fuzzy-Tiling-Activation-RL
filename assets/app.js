import { fta, tiles, tilesUsed, normals } from './fta.js';

const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const U = [0.01, 0.1, 0.5, 1, 2, 5, 20];

// ---------- 1. FTA playground ----------
function drawFTA() {
  const z0 = +$('z').value, u = U[+$('u').value], k = +$('k').value, eFrac = +$('e').value;
  const useTanh = $('tanh1').checked, z = useTanh ? Math.tanh(z0) : z0;
  const lo = -u, hi = u, { delta, c } = tiles(lo, hi, k), eta = eFrac * delta, out = fta(z, lo, hi, k, eta);
  $('zVal').textContent = z0.toFixed(2); $('uVal').textContent = u; $('kVal').textContent = k; $('eVal').textContent = eFrac.toFixed(2);
  const W = 900, pad = 40, axisY = 70, span = Math.max(u * 1.3, Math.min(3.2, Math.abs(z) * 1.15 + 0.1)), x = v => pad + ((v + span) / (2 * span)) * (W - 2 * pad);
  const blue = css('--blue'), muted = css('--muted'), line = css('--line'), ink = css('--ink'), orange = css('--orange');
  let s = `<line x1="${pad}" x2="${W - pad}" y1="${axisY}" y2="${axisY}" stroke="${line}" stroke-width="2"/>`;
  c.forEach((ci, i) => {
    const x0 = x(ci), x1 = x(ci + delta), a = out[i];
    s += `<rect x="${x0}" y="${axisY - 22}" width="${Math.max(1, x1 - x0 - 1)}" height="22" fill="${blue}" fill-opacity="${0.08 + 0.82 * a}" stroke="${line}"/>`;
  });
  s += `<text x="${x(lo)}" y="${axisY + 18}" text-anchor="middle" fill="${muted}" font-size="12">−u = ${-u}</text><text x="${x(hi)}" y="${axisY + 18}" text-anchor="middle" fill="${muted}" font-size="12">u = ${u}</text>`;
  const zx = x(Math.max(-span, Math.min(span, z)));
  s += `<line x1="${zx}" x2="${zx}" y1="${axisY - 34}" y2="${axisY + 6}" stroke="${orange}" stroke-width="3"/><text x="${zx}" y="${axisY - 40}" text-anchor="middle" fill="${orange}" font-size="13" font-weight="700">${useTanh ? `tanh(z) = ${z.toFixed(2)}` : `z = ${z.toFixed(2)}`}</text>`;
  if (useTanh) s += `<text x="${x(z0 > span ? span : z0 < -span ? -span : z0)}" y="${axisY + 34}" text-anchor="middle" fill="${muted}" font-size="11">z = ${z0.toFixed(2)}</text>`;
  // output vector
  const bw = Math.min(34, (W - 2 * pad) / k - 4), bx0 = (W - k * (bw + 4)) / 2, base = 225, hMax = 85;
  s += `<text x="${pad}" y="${base - hMax - 12}" fill="${muted}" font-size="12">Output: one number per tile (k = ${k})</text>`;
  out.forEach((a, i) => {
    const h = a * hMax, bx = bx0 + i * (bw + 4);
    s += `<rect x="${bx}" y="${base - h}" width="${bw}" height="${Math.max(h, 1)}" rx="2" fill="${a > 0 ? blue : line}"/>`;
    if (a > 0) s += `<text x="${bx + bw / 2}" y="${base - h - 4}" text-anchor="middle" font-size="10" fill="${ink}">${a.toFixed(2)}</text>`;
  });
  $('ftaViz').innerHTML = s;
  const active = out.filter(v => v > 0).length, outside = z < lo || z > hi;
  $('ftaNote').innerHTML = outside
    ? `z is outside the tiling bound, so every tile is ${out.some(v => v > 0) ? 'nearly ' : ''}zero and the network gets almost no signal or gradient. ${useTanh ? '' : 'Try switching tanh on.'}`
    : `${active} of ${k} outputs are non-zero, so the representation is sparse: ${Math.round(100 * (1 - active / k))}% of the neurons are silent for this input. ${eFrac === 0 ? 'With η = 0 the edges are hard, and the tiles pass no gradient.' : ''}`;
}

// ---------- 2. Why the bound matters ----------
const BASE = normals(64);
function drawBound() {
  const sd = 10 ** +$('s').value, u = +$('b').value, k = 20;
  $('sVal').textContent = sd < 1 ? sd.toFixed(2) : sd.toFixed(1); $('bVal').textContent = u;
  const zs = BASE.map(v => v * sd), plain = tilesUsed(zs, -u, u, k), withT = tilesUsed(zs.map(Math.tanh), -1, 1, k);
  const strip = (used, id) => { const w = 400 / k; $(id).innerHTML = used.map((b, i) => `<rect x="${i * w + 1}" y="6" width="${w - 2}" height="28" rx="3" fill="${b ? css('--blue') : css('--line')}"/>`).join(''); };
  strip(plain, 'stripPlain'); strip(withT, 'stripTanh');
  const np = plain.filter(Boolean).length, nt = withT.filter(Boolean).length;
  $('usedPlain').textContent = `${np} of ${k}`; $('usedTanh').textContent = `${nt} of ${k}`;
  const inRange = zs.filter(z => Math.abs(z) <= u).length;
  $('boundNote').innerHTML = np <= 3 && inRange > 32
    ? `The bound is far wider than the values, so almost all of them land in the middle ${np} tiles. This is the case the paper describes for u = 20: inputs between −1 and 1 fill only 2 of 20 tiles.`
    : inRange < 32
      ? `Most values (${64 - inRange} of 64) fall outside ±${u}, where FTA outputs zero and passes no gradient. The bound is too narrow.`
      : `This bound suits this spread. But change the spread and it stops fitting, and in a real network the spread changes as it learns.`;
  $('boundNote').innerHTML += ` With tanh in front, the values always lie between −1 and 1, so u = 1 fits them whatever their spread: move the spread slider and the right-hand count barely changes.`;
}

// ---------- 3. The paper's figures ----------
const FIGS = [
  ['ftavrelu', '1. Reproduced', 'DQN-FTA (black) against DQN with ReLU (red) and DQN-Large (blue) on LunarLander; dotted lines use a target network. FTA did slightly better than DQN, and DQN-Large better than both, the same pattern Pan et al. reported.'],
  ['sweepfta', '2. Bound sweep', 'DQN-FTA with tiling bounds u from 0.1 to 100, for 16, 64 and 128 tiles. It did best near u = 1 and badly when u was far bigger or smaller. More tiles helped a too-wide bound: u = 10 with 128 tiles behaves like u = 1 with 16, because both put about the same number of tiles in [−1, 1].'],
  ['bestfta', '3. Tuned FTA', 'With the right bound (u = 1, 64 tiles, green), DQN-FTA beat DQN and DQN-Large. The untuned u = 20 (black) did not. FTA wins only if you search for u.'],
  ['normalizing', '4. The fix', 'Batch Norm, Range Norm or tanh before FTA, at the best bound (u = 1) and the two worst (u = 0.01 and 100). tanh did well at every bound. Range Norm did poorly throughout, and Batch Norm rescued u = 100 but failed badly at u = 0.01.'],
  ['cartpole', '5. CartPole', 'Plain FTA on CartPole (left): here u = 100 was best and u = 1 worse, the opposite of LunarLander, so the right bound depends on the task. With tanh (right), all three bounds did about as well as the best.'],
  ['distribution', '6. Consistency', 'How returns were spread across 50 runs per agent. Near the end of training DQN (red) and untuned DQN-FTA (black) split into good and bad runs, while tanh + FTA (cyan) stayed clustered around one value. tanh made FTA more consistent as well as better.'],
];
let fig = 0;
function drawFig() {
  const [f, , note] = FIGS[fig];
  $('figChips').innerHTML = FIGS.map(([, l], i) => `<button class="chip" data-i="${i}" aria-pressed="${i === fig}">${l}</button>`).join('');
  $('figChips').querySelectorAll('.chip').forEach(b => b.onclick = () => { fig = +b.dataset.i; drawFig(); });
  const img = $('figImg'); img.onload = () => { img.style.maxWidth = img.naturalWidth < 1000 ? '620px' : '100%'; }; img.style.display = 'block'; img.style.margin = '0 auto'; img.src = `paper/figures/${f}.png`; $('figImg').alt = note; $('figNote').textContent = note;
}

// ---------- 4. Re-run results ----------
let data, chart, exp = 'cartpole', view = 'all';
const VIEWS = {
  cartpole: [['all', 'All'], ['plain', 'Plain FTA'], ['tanh', 'tanh + FTA']],
  lunarlander: [['all', 'All'], ['bound', 'Plain FTA by bound'], ['fix', 'tanh fix'], ['base', 'Against ReLU']],
};
const PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#8a5cd0', '#d6a72a', '#d23a7a', '#4a4a4a', '#16a3b8', '#8c6d3a'];
function pick(label) {
  if (exp === 'cartpole') return view === 'all' || (view === 'tanh') === label.includes('tanh');
  if (view === 'all') return true;
  if (view === 'bound') return label.startsWith('DQN-FTA (');
  if (view === 'fix') return label.includes('tanh') || /u=(0.01|100),/.test(label);
  return !label.includes('tanh') && (!label.startsWith('DQN-FTA') || label.includes('u=20,'));
}
function drawResults() {
  $('viewChips').innerHTML = VIEWS[exp].map(([k, l]) => `<button class="chip" data-v="${k}" aria-pressed="${k === view}">${l}</button>`).join('');
  $('viewChips').querySelectorAll('.chip').forEach(b => b.onclick = () => { view = b.dataset.v; drawResults(); });
  const e = data.experiments[exp], agents = e.agents.filter(a => pick(a.label));
  $('resNote').textContent = e.note;
  if (typeof Chart === 'undefined') return;
  const ds = [];
  agents.forEach(a => {
    const col = PALETTE[e.agents.indexOf(a) % PALETTE.length], dashed = a.label.includes('tanh') ? [] : a.label.startsWith('DQN-FTA') ? [6, 4] : [];
    ds.push({ label: a.label, data: a.steps.map((s, i) => ({ x: s, y: a.mean[i] })), borderColor: col, backgroundColor: col, borderWidth: 2.2, pointRadius: 0, borderDash: dashed });
    ds.push({ label: '_lo', data: a.steps.map((s, i) => ({ x: s, y: a.lo[i] })), borderWidth: 0, pointRadius: 0, fill: false });
    ds.push({ label: '_hi', data: a.steps.map((s, i) => ({ x: s, y: a.hi[i] })), borderWidth: 0, pointRadius: 0, backgroundColor: col + '22', fill: '-1' });
  });
  const opts = { responsive: true, maintainAspectRatio: false, animation: false, parsing: false, interaction: { mode: 'nearest', intersect: false },
    scales: { x: { type: 'linear', title: { display: true, text: 'Training steps' }, ticks: { callback: v => v >= 1000 ? `${v / 1000}k` : v } }, y: { title: { display: true, text: e.ylabel } } },
    plugins: { legend: { position: 'bottom', labels: { filter: i => !i.text.startsWith('_'), boxWidth: 12 } },
      tooltip: { filter: i => !i.dataset.label.startsWith('_'), callbacks: { label: c => `${c.dataset.label}: ${c.raw.y.toFixed(0)}` } } } };
  if (chart) chart.destroy();
  chart = new Chart($('resChart'), { type: 'line', data: { datasets: ds }, options: opts });
}

async function main() {
  if (typeof Chart !== 'undefined') { Chart.defaults.font.family = css('--sans'); Chart.defaults.color = css('--muted'); Chart.defaults.borderColor = css('--line'); }
  ['z', 'u', 'k', 'e', 'tanh1'].forEach(id => $(id).addEventListener('input', drawFTA)); drawFTA();
  ['s', 'b'].forEach(id => $(id).addEventListener('input', drawBound)); drawBound();
  drawFig();
  try { const r = await fetch('results/summary.json'); if (!r.ok) throw 0; data = await r.json(); } catch { $('resHead').closest('section').hidden = true; return; }
  $('resIntro').textContent = data.intro;
  $('expChips').querySelectorAll('.chip').forEach(b => b.onclick = () => { exp = b.dataset.exp; view = 'all'; $('expChips').querySelectorAll('.chip').forEach(x => x.setAttribute('aria-pressed', String(x === b))); drawResults(); });
  drawResults();
}
main();
