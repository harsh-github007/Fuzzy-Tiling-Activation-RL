import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fta, tilesUsed } from '../assets/fta.js';

test('paper example: u = 10, k = 4, z in (0, 1] lands in tile 3', () => {
  for (const z of [0.01, 0.5, 1]) assert.deepEqual(fta(z, -10, 10, 4, 0), [0, 0, 1, 0]);
});
test('fuzzy edge matches the Python layer', () => {
  const out = fta(0.2, -1, 1, 4);           // eta = delta = 0.5
  assert.ok(Math.abs(out[1] - 0.8) < 1e-9 && out[2] === 1 && Math.abs(out[3] - 0.7) < 1e-9);
});
test('inputs in [-1, 1] use 2 of 20 tiles when u = 20', () => {
  const zs = Array.from({ length: 200 }, (_, i) => -1 + (i * 1.999) / 199);
  assert.equal(tilesUsed(zs, -20, 20, 20, 0).filter(Boolean).length, 2);
});
