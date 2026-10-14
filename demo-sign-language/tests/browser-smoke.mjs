// Run with an existing Chrome installation: node tests/browser-smoke.mjs
// Fixture data below is synthetic and is only used for software verification.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { writeFile, readFile } from 'node:fs/promises';
import { extractFeatures, FRAMES, emptyDataset } from '../core.mjs';

const url = process.env.DEMO_URL || 'http://localhost:5173';
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required'] });
const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(url); await page.locator('#startHero').waitFor();
  await page.screenshot({ path: 'tests/desktop-preview.png', fullPage: true });
  assert.equal(await page.locator('h1').count(), 1);
  await page.locator('#simulateButton').click();
  await page.waitForFunction(() => document.getElementById('transcript').textContent.includes('จำลอง'));
  assert.equal(await page.locator('#confidence').textContent(), '—');
  assert.equal(await page.locator('#sourceSummary').textContent(), 'โหมดซ้อมนำเสนอ');
  const download = page.waitForEvent('download'); await page.locator('#downloadText').click();
  assert.equal((await download).suggestedFilename(), 'snapsign-transcript.txt');
  await page.locator('#simulateButton').click();
  await page.locator('#labelInput').fill('<img src=x onerror=alert(1)>'); await page.locator('#addLabel').click();
  assert.equal(await page.locator('#classes img').count(), 0);
  await page.locator('[data-remove]').click();

  // Exercise the real MediaPipe WASM runtime against Chrome's fake camera stream.
  await page.locator('#cameraButton').click();
  await page.waitForFunction(() => document.getElementById('sourceSummary').textContent === 'กำลังรับภาพจากกล้อง', null, { timeout: 60000 });
  await page.waitForFunction(() => document.getElementById('overlay').width > 300);
  await page.locator('#cameraButton').click();
  assert.equal(await page.locator('#video').evaluate(v => v.srcObject), null);
  console.log('PASS: UI, simulation provenance, export, XSS label handling, actual MediaPipe initialization and camera cleanup');

  const fixture = emptyDataset();
  for (let c = 0; c < 2; c++) {
    fixture.classes.push({ label: `ท่าทดสอบ ${c + 1}`, samples: Array.from({ length: 3 }, (_, sample) => Array.from({ length: FRAMES }, (_, f) => {
      const points = Array.from({ length: 21 }, (_, p) => ({ x: .3 + c * .3 + p * .007 + Math.sin(f / 4) * .03, y: .7 - p * (c ? .008 : .015) + sample * .002, z: -.001 * p }));
      return extractFeatures({ landmarks: [points], handedness: [[{ categoryName: 'Right' }]] });
    })) });
  }
  await page.locator('#importData').setInputFiles({ name: 'fixture.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(fixture)) });
  await page.waitForFunction(() => document.getElementById('trainButton').disabled === false);
  await page.locator('#trainButton').click();
  await page.waitForFunction(() => document.getElementById('trainingStatus').textContent.startsWith('ฝึกเสร็จแล้ว'), null, { timeout: 120000 });
  assert.equal(await page.locator('#customMode').getAttribute('aria-pressed'), 'true');
  const downloadModel = page.waitForEvent('download'); await page.locator('#exportModel').click();
  const modelDownload = await downloadModel; const modelPath = await modelDownload.path();
  await page.locator('#importModel').setInputFiles(modelPath);
  await page.waitForFunction(() => document.getElementById('trainingStatus').textContent.startsWith('นำเข้าโมเดลแล้ว'), null, { timeout: 30000 });
  const roundtrip = await page.evaluate(async samples => {
    const { train, predict, serializeModel, restoreModel } = await import('./ml.mjs');
    const trained = await train(samples, null, { epochs: 1 });
    const before = await predict(trained, samples.classes[0].samples[0]);
    const saved = await serializeModel(trained); const loaded = await restoreModel(saved);
    const after = await predict(loaded, samples.classes[0].samples[0]);
    const a = trained.model.getWeights().map(w => Array.from(w.dataSync()));
    const b = loaded.model.getWeights().map(w => Array.from(w.dataSync()));
    const maxWeightDiff = Math.max(...a.map((w, i) => Math.max(...w.map((n, j) => Math.abs(n - b[i][j])))));
    trained.model.dispose(); loaded.model.dispose();
    return { before, after, maxWeightDiff };
  }, fixture);
  assert.deepEqual(roundtrip.before.map(r => r.label), roundtrip.after.map(r => r.label));
  console.log('Model roundtrip:', JSON.stringify(roundtrip));
  assert.equal(roundtrip.maxWeightDiff, 0);
  roundtrip.before.forEach((r, i) => assert.ok(Math.abs(r.score - roundtrip.after[i].score) < 0.002));
  await page.locator('#importData').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"schema":"wrong"}') });
  await page.waitForFunction(() => document.getElementById('toast').textContent.startsWith('นำเข้าไม่ได้'));
  assert.equal(await page.locator('.class-card').count(), 2);
  console.log('PASS: real TensorFlow.js LSTM training, model download/import, equivalent predictions after serialization, invalid dataset rejection');

  // A deterministic detector stub exercises sample capture, independently of ML quality.
  const capturePage = await context.newPage();
  await capturePage.route('**/vendor/vision/vision_bundle.mjs', route => route.fulfill({ contentType: 'text/javascript', body: `
    export const FilesetResolver = { forVisionTasks: async () => ({}) };
    export const GestureRecognizer = { createFromOptions: async () => ({
      recognizeForVideo: () => ({
        landmarks: [Array.from({length:21},(_,p)=>({x:.4+p*.006,y:.7-p*.01,z:-p*.001}))],
        handedness: [[{categoryName:'Right'}]], gestures: [[{categoryName:'Open_Palm',score:.95}]]
      })
    }) };
  ` }));
  await capturePage.goto(url);
  await capturePage.evaluate(() => localStorage.removeItem('snapsign.dataset.v1'));
  await capturePage.reload();
  await capturePage.locator('#labelInput').fill('ท่าจากตัวตรวจจับทดสอบ'); await capturePage.locator('#addLabel').click();
  await capturePage.locator('#cameraButton').click();
  await capturePage.waitForFunction(() => !document.querySelector('[data-record]').disabled);
  await capturePage.locator('[data-record]').click();
  await capturePage.waitForFunction(() => document.querySelector('.class-card-count').textContent.startsWith('1 /'), null, { timeout: 15000 });
  const exportCapture = capturePage.waitForEvent('download'); await capturePage.locator('#exportData').click();
  const capturedPath = await (await exportCapture).path();
  const capturedData = JSON.parse(await readFile(capturedPath, 'utf8'));
  assert.equal(capturedData.classes[0].samples[0].length, FRAMES);
  await capturePage.locator('[data-record]').click(); await capturePage.locator('#cameraButton').click();
  assert.equal(await capturePage.locator('#recordOverlay').isHidden(), true);
  assert.ok((await capturePage.locator('.class-card-count').textContent()).startsWith('1 /'));
  await capturePage.close();
  console.log('PASS: 16-frame sample capture, dataset export and interrupted recording cleanup (detector stub)');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#importData').evaluate(el => el.blur());
  await page.evaluate(() => { document.getElementById('toast').hidden = true; });
  await page.screenshot({ path: 'tests/mobile-preview.png', fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  const denied = await context.newPage();
  await denied.addInitScript(() => { Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { value: async () => { throw new DOMException('Denied', 'NotAllowedError'); } }); });
  await denied.goto(url); await denied.locator('#cameraButton').click();
  await denied.waitForFunction(() => document.getElementById('toast').textContent.includes('สิทธิ์ใช้กล้อง'), null, { timeout: 60000 });
  assert.equal(await denied.locator('#cameraButton').isEnabled(), true);
  console.log('PASS: mobile layout at 390px, denied camera permissions recovery');
  assert.deepEqual(errors, []);
  await writeFile('tests/browser-result.json', JSON.stringify({ status: 'passed', syntheticDataOnly: true, browser: browser.version(), checks: ['UI and simulation provenance', 'Transcript export', 'Label XSS prevention', 'Real MediaPipe initialization', 'Camera cleanup', 'Real LSTM training', 'Model roundtrip weights and predictions', 'Invalid dataset rejection', '16-frame capture with detector stub', 'Interrupted recording cleanup', 'Mobile layout', 'Camera permission denial'] }, null, 2));
} finally { await browser.close(); }
