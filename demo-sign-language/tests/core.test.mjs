import test from 'node:test';
import assert from 'node:assert/strict';
import { extractFeatures, hasHand, validateDataset, emptyDataset, FRAMES, FEATURES, StabilityGate, rankPrediction, transcriptText, trainingProblem } from '../core.mjs';

const hand = (x = 0.5) => Array.from({ length: 21 }, (_, i) => ({ x: x + i * .005, y: .6 - i * .01, z: i * -.001 }));
const frame = () => extractFeatures({ landmarks: [hand()], handedness: [[{ categoryName: 'Right' }]] });

test('left/right feature slots remain stable when detection order changes', () => {
  const a = hand(.2), b = hand(.7);
  const first = extractFeatures({ landmarks: [a, b], handedness: [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]] });
  const reversed = extractFeatures({ landmarks: [b, a], handedness: [[{ categoryName: 'Right' }], [{ categoryName: 'Left' }]] });
  assert.equal(first.length, FEATURES); assert.deepEqual(first, reversed);
  assert.equal(first[0], 1); assert.equal(first[66], 1);
  assert.deepEqual(first.slice(3, 6), [0, 0, 0]);
  assert.equal(extractFeatures({}).every(v => v === 0), true);
});
test('normalization preserves wrist motion and reduces scale dependence', () => {
  const original = hand(), wrist = original[0];
  const larger = original.map(p => ({ x: wrist.x + (p.x - wrist.x) * 2, y: wrist.y + (p.y - wrist.y) * 2, z: p.z * 2 }));
  const a = extractFeatures({ landmarks: [original] }), b = extractFeatures({ landmarks: [larger] });
  a.forEach((n, i) => assert.ok(Math.abs(n - b[i]) < 1e-10));
  assert.notEqual(frame()[67], extractFeatures({ landmarks: [hand(.7)], handedness: [[{ categoryName: 'Right' }]] })[67]);
});
test('dataset rejects corrupt dimensions, missing hands, duplicate labels, and nonfinite values', () => {
  const make = () => ({ ...emptyDataset(), classes: [{ label: 'test', samples: [Array.from({ length: FRAMES }, frame)] }] });
  assert.equal(hasHand(frame()), true); assert.doesNotThrow(() => validateDataset(make()));
  let bad = make(); bad.classes[0].samples[0][0].pop(); assert.throws(() => validateDataset(bad));
  bad = make(); bad.classes[0].samples[0][0][2] = NaN; assert.throws(() => validateDataset(bad));
  bad = make(); bad.classes[0].samples[0] = Array.from({ length: FRAMES }, () => Array(FEATURES).fill(0)); assert.throws(() => validateDataset(bad));
  bad = make(); bad.classes.push({ label: 'test', samples: [] }); assert.throws(() => validateDataset(bad));
});
test('stability gate filters jitter and duplicates, rearms after release', () => {
  const gate = new StabilityGate(600, 500);
  assert.equal(gate.update('a', .9, .75, 0), false);
  assert.equal(gate.update('b', .9, .75, 200), false);
  assert.equal(gate.update('a', .9, .75, 300), false);
  assert.equal(gate.update('a', .9, .75, 901), true);
  assert.equal(gate.update('a', .9, .75, 1600), false);
  gate.update('', 0, .75, 1700); gate.update('', 0, .75, 2300);
  gate.update('a', .9, .75, 2400); assert.equal(gate.update('a', .9, .75, 3100), true);
  assert.equal(gate.update('b', .7, .75, 3200), false);
});
test('training requires two classes and at least three samples each', () => {
  assert.match(trainingProblem(emptyDataset()), /2/);
  const data = { classes: [{ samples: [1, 2, 3] }, { samples: [1, 2] }] };
  assert.match(trainingProblem(data), /3/); data.classes[1].samples.push(3); assert.equal(trainingProblem(data), '');
});
test('simulated output stays marked when exported and predictions are ranked safely', () => {
  assert.equal(transcriptText([{ label: 'a', simulated: true }, { label: 'b' }]), '[จำลอง] a b');
  assert.equal(rankPrediction([.2, .8], ['a', 'b'])[0].label, 'b');
  assert.deepEqual(rankPrediction([NaN], ['a']), []);
  assert.deepEqual(rankPrediction([.1], ['a', 'b']), []);
});
