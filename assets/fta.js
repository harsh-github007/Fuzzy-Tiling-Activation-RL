// The fuzzy tiling activation, in plain JavaScript, matching fta/layers.py.
export function tiles(low, high, k) {
  const delta = (high - low) / k;
  return { delta, c: Array.from({ length: k }, (_, i) => low + i * delta) };
}
export function fta(z, low, high, k, eta) {
  const { delta, c } = tiles(low, high, k);
  const e = eta ?? delta;
  return c.map(ci => {
    const d = Math.max(ci - z, 0) + Math.max(z - delta - ci, 0);
    const ind = e > 0 ? (d <= e ? d : 1) : (d > 0 ? 1 : 0);
    return 1 - ind;
  });
}
// Share of the k tiles that any of the inputs activates (output > 0).
export function tilesUsed(zs, low, high, k, eta) {
  const used = new Array(k).fill(false);
  for (const z of zs) fta(z, low, high, k, eta).forEach((v, i) => { if (v > 0) used[i] = true; });
  return used;
}
// Deterministic normal samples, so the page looks the same on every visit.
export function normals(n, seed = 7) {
  let s = seed; const rnd = () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296);
  return Array.from({ length: n }, () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()));
}
