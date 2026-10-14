import { FRAMES, FEATURES, SAMPLE_MS, GESTURES, extractFeatures, hasHand, emptyDataset, validateDataset, trainingProblem, StabilityGate, transcriptText } from './core.mjs';
import { train, predict, serializeModel, restoreModel } from './ml.mjs';

const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
const STORAGE = 'snapsign.dataset.v1';
const gate = new StabilityGate();
let dataset = emptyDataset(), bundle = null, recognizer = null, recognizerLoading = null;
let source = 'off', mode = 'basic', stream = null, cameraStarting = false, facing = 'user', session = 0;
let frameRequest = 0, simTimer = 0, toastTimer = 0, recording = null, training = false, trainingModel = null;
let buffer = [], lastFrameTime = -1, lastSample = 0, lastPrediction = 0, latestRanks = [], current = null;
let lastFrameAt = 0, tokens = [], resultBusy = false;
const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];

function toast(message, error = false) {
  $('toast').textContent = message; $('toast').classList.toggle('error', error); $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 8000 : 4500);
}
function persist() {
  try { localStorage.setItem(STORAGE, JSON.stringify(dataset)); }
  catch { toast('พื้นที่จัดเก็บเบราว์เซอร์เต็ม ตัวอย่างยังอยู่ในหน้านี้ กรุณาส่งออกชุดข้อมูล', true); }
}
function invalidateModelNotice() {
  if (bundle) $('trainingStatus').textContent = 'ชุดข้อมูลเปลี่ยนแล้ว โมเดลปัจจุบันยังเป็นรุ่นเดิม กดฝึกอีกครั้งเพื่ออัปเดต';
}
function busy() { return training || !!recording || cameraStarting; }
function updateControls() {
  $('cameraButton').disabled = training || cameraStarting;
  $('startHero').disabled = training || cameraStarting;
  $('simulateHero').disabled = busy(); $('simulateButton').disabled = busy();
  $('switchButton').disabled = source !== 'camera' || busy();
  $('cameraButton').textContent = cameraStarting ? 'กำลังเปิดกล้อง…' : source === 'camera' ? '■ ปิดกล้อง' : '◉ เปิดกล้อง';
  $('simulateButton').textContent = source === 'simulation' ? '■ หยุดจำลอง' : '▷ ข้อมูลจำลอง';
  $('trainButton').disabled = !!recording || cameraStarting || (!training && !!trainingProblem(dataset));
  $('trainButton').textContent = training ? '■ ยกเลิกการฝึก' : '✳ ฝึกโมเดล LSTM';
  $('exportModel').disabled = !bundle || busy();
  $('importData').disabled = busy(); $('importModel').disabled = busy();
  $('importModelLabel').classList.toggle('is-disabled', busy());
  $('addLabel').disabled = busy(); $('labelInput').disabled = busy();
  document.querySelectorAll('[data-record]').forEach(button => { button.disabled = source !== 'camera' || busy() || dataset.classes[Number(button.dataset.record)].samples.length >= 40 || totalSamples() >= 300; });
  document.querySelectorAll('[data-remove], [data-undo]').forEach(button => { button.disabled = busy() || (button.hasAttribute('data-undo') && dataset.classes[Number(button.dataset.undo)].samples.length === 0); });
  $('basicMode').disabled = training || !!recording; $('customMode').disabled = training || !!recording;
}
function totalSamples() { return dataset.classes.reduce((count, c) => count + c.samples.length, 0); }
function renderDataset() {
  const container = $('classes'); container.replaceChildren();
  if (!dataset.classes.length) {
    const p = document.createElement('p'); p.className = 'class-empty'; p.textContent = 'ยังไม่มีคำในชุดข้อมูล เพิ่มคำแรกด้านบนได้เลย'; container.append(p);
  }
  dataset.classes.forEach((item, index) => {
    const card = document.createElement('div'); card.className = 'class-card';
    const top = document.createElement('div'); top.className = 'class-card-top';
    const title = document.createElement('strong'); title.textContent = item.label;
    const remove = document.createElement('button'); remove.textContent = '×'; remove.dataset.remove = index; remove.setAttribute('aria-label', `ลบคำ ${item.label}`);
    remove.onclick = () => {
      if (item.samples.length && !confirm(`ลบคำ “${item.label}” และตัวอย่าง ${item.samples.length} คลิป? กรุณาส่งออกก่อนหากต้องการเก็บสำรอง`)) return;
      dataset.classes.splice(index, 1); persist(); renderDataset(); invalidateModelNotice();
    };
    top.append(title, remove);
    const count = document.createElement('span'); count.className = 'class-card-count'; count.textContent = `${item.samples.length} / 40 คลิป · ${item.samples.length >= 3 ? 'พร้อมทดลองฝึก' : 'เก็บอย่างน้อย 3 คลิป'}`;
    const actions = document.createElement('div'); actions.className = 'class-card-actions';
    const capture = document.createElement('button'); capture.className = 'button secondary'; capture.textContent = '◉ เก็บตัวอย่าง'; capture.dataset.record = index; capture.onclick = () => startRecording(index);
    const undo = document.createElement('button'); undo.className = 'text-button'; undo.textContent = 'ลบคลิปล่าสุด'; undo.dataset.undo = index; undo.onclick = () => { item.samples.pop(); persist(); renderDataset(); invalidateModelNotice(); };
    actions.append(capture, undo); card.append(top, count, actions); container.append(card);
  });
  if (!training) $('trainingStatus').textContent = bundle ? `มีโมเดลพร้อมใช้ ${bundle.labels.length} คำ · ชุดข้อมูล ${totalSamples()} คลิป` : trainingProblem(dataset) || `พร้อมฝึกจาก ${dataset.classes.length} คำ รวม ${totalSamples()} คลิป`;
  updateControls();
}
function setMode(next) {
  mode = next; buffer = []; latestRanks = []; gate.reset();
  $('basicMode').classList.toggle('selected', mode === 'basic'); $('basicMode').setAttribute('aria-pressed', mode === 'basic');
  $('customMode').classList.toggle('selected', mode === 'custom'); $('customMode').setAttribute('aria-pressed', mode === 'custom');
  $('modelType').textContent = mode === 'basic' ? 'MEDIAPIPE · 7 ท่าทดลอง' : bundle ? `LSTM · ${bundle.labels.length} คำที่ฝึก` : 'LSTM · ยังไม่มีโมเดล';
  showResult(null);
}
function showResult(result, ranks = []) {
  current = result;
  $('resultLabel').textContent = result?.label || (mode === 'custom' && !bundle ? 'เริ่มสร้างโมเดลของคุณ' : 'รอท่ามือของคุณ');
  $('resultIcon').textContent = result?.icon || '⌁';
  $('resultHelp').textContent = result ? result.simulated ? 'ข้อมูลจำลองสำหรับดูหน้าจอ ไม่มีการวิเคราะห์ภาพ' : 'ค้างท่าให้นิ่ง หรือกดเพิ่มผลในข้อความ' : mode === 'custom' ? (bundle ? 'ทำท่าตามคำที่ฝึก เก็บภาพประมาณ 2 วินาที' : 'เก็บตัวอย่างและฝึกโมเดลในส่วนด้านล่าง') : 'ลองกางมือ กำมือ หรือชูนิ้วโป้ง';
  $('confidence').textContent = result && !result.simulated ? `${Math.round(result.score * 100)}%` : '—';
  $('confidenceBar').style.width = result && !result.simulated ? `${result.score * 100}%` : '0%';
  $('addResult').disabled = !result;
  $('alternatives').replaceChildren();
  if (mode === 'custom' && !result?.simulated) ranks.slice(1, 3).forEach(rank => {
    const row = document.createElement('div'), label = document.createElement('span'), score = document.createElement('span');
    label.textContent = rank.label; score.textContent = `${Math.round(rank.score * 100)}%`; row.append(label, score); $('alternatives').append(row);
  });
}
function handlePrediction(result, ranks, now) {
  const threshold = Number($('threshold').value) / 100;
  const accepted = result && (result.simulated || result.score >= threshold) ? result : null;
  showResult(accepted, ranks);
  const stable = gate.update(accepted?.label || '', accepted?.score || 0, threshold, now);
  if (stable && $('autoAdd').checked && accepted) addToken(accepted);
}
function addToken(result) {
  if (!result) return;
  if (tokens.length >= 100) { toast('ข้อความครบ 100 คำแล้ว บันทึกแล้วล้างเพื่อเริ่มใหม่'); return; }
  tokens.push({ label: result.label, simulated: !!result.simulated }); renderTranscript();
}
function renderTranscript() {
  const el = $('transcript'); el.replaceChildren();
  if (!tokens.length) { const p = document.createElement('span'); p.className = 'placeholder'; p.textContent = 'คำที่เลือกจะปรากฏที่นี่…'; el.append(p); }
  tokens.forEach((token, index) => {
    const chip = document.createElement('span'); chip.className = 'token' + (token.simulated ? ' simulated' : '');
    const label = document.createElement('span'); label.textContent = token.label; chip.append(label);
    if (token.simulated) { const tag = document.createElement('small'); tag.textContent = 'จำลอง'; chip.append(tag); }
    const remove = document.createElement('button'); remove.textContent = '×'; remove.setAttribute('aria-label', `ลบ ${token.label}`); remove.onclick = () => { tokens.splice(index, 1); renderTranscript(); }; chip.append(remove); el.append(chip);
  });
  $('tokenCount').textContent = `${tokens.length} คำ · ยังไม่ได้เรียบเรียงเป็นประโยค`;
  ['speakButton', 'copyButton', 'downloadText', 'clearText'].forEach(id => { $(id).disabled = !tokens.length; });
}
function saveFile(name, value, mime = 'application/json') {
  const url = URL.createObjectURL(new Blob([typeof value === 'string' ? value : JSON.stringify(value)], { type: mime }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function loadRecognizer() {
  if (recognizer) return recognizer;
  if (!recognizerLoading) recognizerLoading = (async () => {
    const { GestureRecognizer, FilesetResolver } = await import('./vendor/vision/vision_bundle.mjs');
    const vision = await FilesetResolver.forVisionTasks('./vendor/vision/wasm');
    recognizer = await GestureRecognizer.createFromOptions(vision, { baseOptions: { modelAssetPath: './vendor/gesture_recognizer.task', delegate: 'CPU' }, runningMode: 'VIDEO', numHands: 2 });
    return recognizer;
  })().catch(() => { recognizerLoading = null; throw new Error('โหลดโมเดลไม่สำเร็จ ลองรัน python scripts/download_assets.py แล้วเปิดหน้านี้ใหม่'); });
  return recognizerLoading;
}
function stopSource() {
  session++; cancelAnimationFrame(frameRequest); clearInterval(simTimer);
  stream?.getTracks().forEach(track => track.stop()); stream = null; video.srcObject = null;
  if (recording) { recording = null; $('recordOverlay').hidden = true; toast('หยุดเก็บตัวอย่าง ยังไม่บันทึกคลิปที่ไม่ครบ'); }
  source = 'off'; buffer = []; latestRanks = []; lastFrameTime = -1; gate.reset();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $('cameraCard').classList.remove('is-live'); $('cameraEmpty').hidden = false;
  $('sourcePill').classList.remove('sim-label'); $('sourcePill').replaceChildren(document.createTextNode('ยังไม่เปิดกล้อง'));
  $('sourceSummary').textContent = 'พร้อมเริ่มต้น'; $('handCount').textContent = '0 มือ'; $('fps').textContent = '— FPS'; $('liveLabel').textContent = 'รอรับภาพ';
  $('cameraHint').textContent = 'ภาพจากกล้องจะไม่ถูกส่งไปยังเซิร์ฟเวอร์'; showResult(null); updateControls();
}
async function startCamera() {
  if (training || cameraStarting) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { toast('กล้องต้องเปิดผ่าน http://localhost หรือ HTTPS กรุณารัน npm start', true); return; }
  stopSource(); cameraStarting = true; updateControls();
  $('sourceSummary').textContent = 'กำลังโหลดโมเดลและขอใช้กล้อง…';
  let pendingStream;
  const mySession = session;
  try {
    await loadRecognizer();
    pendingStream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: facing }, width: { ideal: 960 }, height: { ideal: 720 } } });
    if (mySession !== session) { pendingStream.getTracks().forEach(t => t.stop()); return; }
    video.srcObject = pendingStream; await video.play();
    stream = pendingStream; source = 'camera'; lastSample = 0; lastPrediction = 0; lastFrameAt = 0;
    $('cameraCard').classList.add('is-live'); $('cameraEmpty').hidden = true;
    $('sourcePill').textContent = '● กล้องสด'; $('sourceSummary').textContent = 'กำลังรับภาพจากกล้อง'; $('liveLabel').textContent = '● LIVE';
    $('cameraHint').textContent = 'หันฝ่ามือเข้ากล้อง เห็นมือและนิ้วครบทั้งมือ';
    frameRequest = requestAnimationFrame(loop);
  } catch (error) {
    pendingStream?.getTracks().forEach(t => t.stop()); stopSource();
    const messages = { NotAllowedError: 'ยังไม่ได้รับสิทธิ์ใช้กล้อง กดรูปกล้องข้างแถบที่อยู่เพื่ออนุญาต แล้วลองใหม่', NotFoundError: 'ไม่พบกล้อง ลองเชื่อมต่อกล้องหรือใช้ข้อมูลจำลอง', NotReadableError: 'กล้องอาจถูกใช้งานโดยโปรแกรมอื่น ปิดโปรแกรมนั้นแล้วลองใหม่' };
    toast(messages[error.name] || error.message, true);
  } finally { cameraStarting = false; updateControls(); }
}
function fitCanvas(width, height) { if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; } }
function drawHands(hands, simulated = false) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!$('landmarksToggle').checked && !simulated) return;
  const lineWidth = Math.max(2, canvas.width / 300);
  hands.forEach((points, i) => {
    const color = i === 0 ? '#daf39a' : '#8cdbd4'; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lineWidth;
    for (const [a, b] of connections) { ctx.beginPath(); ctx.moveTo(points[a].x * canvas.width, points[a].y * canvas.height); ctx.lineTo(points[b].x * canvas.width, points[b].y * canvas.height); ctx.stroke(); }
    points.forEach(point => { ctx.beginPath(); ctx.arc(point.x * canvas.width, point.y * canvas.height, lineWidth * 1.5, 0, Math.PI * 2); ctx.fill(); });
  });
}
async function loop(now) {
  if (source !== 'camera') return;
  const mySession = session;
  try {
    if (video.readyState >= 2 && video.currentTime !== lastFrameTime && now - lastSample >= SAMPLE_MS) {
      lastFrameTime = video.currentTime; lastSample = now;
      const result = recognizer.recognizeForVideo(video, now);
      fitCanvas(video.videoWidth, video.videoHeight); drawHands(result.landmarks);
      $('handCount').textContent = `${result.landmarks.length} มือ`;
      if (lastFrameAt) $('fps').textContent = `${Math.min(8, Math.round(1000 / (now - lastFrameAt)))} FPS`;
      lastFrameAt = now;
      const features = extractFeatures(result);
      if (buffer.length === FRAMES) buffer.shift(); buffer.push(features);
      collectFrame(features, now);
      if (!recording) {
        if (mode === 'basic') {
          const top = result.gestures.map(g => g[0]).filter(g => GESTURES[g?.categoryName]).sort((a, b) => b.score - a.score)[0];
          const label = top && GESTURES[top.categoryName];
          handlePrediction(label ? { label: label.label, icon: label.icon, score: top.score } : null, [], now);
        } else if (bundle && buffer.length === FRAMES && now - lastPrediction >= 400 && !resultBusy) {
          lastPrediction = now;
          if (buffer.filter(hasHand).length >= FRAMES * 0.75 && hasHand(features)) {
            resultBusy = true;
            try { latestRanks = await predict(bundle, buffer); } finally { resultBusy = false; }
            if (session !== mySession || source !== 'camera') return;
            const top = latestRanks[0]; handlePrediction(top ? { ...top, icon: '✳' } : null, latestRanks, now);
          } else { latestRanks = []; handlePrediction(null, [], now); }
        } else if (!hasHand(features) || !bundle || buffer.length < FRAMES) handlePrediction(null, [], now);
      }
    }
  } catch (error) { stopSource(); toast(`การประมวลผลหยุด: ${error.message}`, true); return; }
  if (source === 'camera' && session === mySession) frameRequest = requestAnimationFrame(loop);
}
function startRecording(index) {
  if (source !== 'camera' || busy() || dataset.classes[index].samples.length >= 40 || totalSamples() >= 300) return;
  recording = { index, start: performance.now() + 1500, frames: [] }; gate.reset(); showResult(null);
  $('recordOverlay').hidden = false; $('recordText').textContent = 'เตรียมตัว…'; $('recordProgress').value = 0;
  $('cameraCard').scrollIntoView({ behavior: 'smooth', block: 'center' }); updateControls();
}
function collectFrame(features, now) {
  if (!recording) return;
  if (now < recording.start) { $('recordText').textContent = `เตรียมตัว ${Math.ceil((recording.start - now) / 1000)}`; return; }
  recording.frames.push(features); $('recordText').textContent = `กำลังเก็บ “${dataset.classes[recording.index].label}”`; $('recordProgress').value = recording.frames.length;
  if (recording.frames.length === FRAMES) {
    const item = dataset.classes[recording.index], frames = recording.frames;
    recording = null; $('recordOverlay').hidden = true; buffer = []; latestRanks = []; gate.reset();
    if (frames.filter(hasHand).length < FRAMES * 0.75) toast('คลิปนี้เห็นมือไม่ครบ ลองใหม่โดยให้มืออยู่ในภาพตลอดท่า', true);
    else { item.samples.push(frames); persist(); toast(`บันทึก “${item.label}” แล้ว ${item.samples.length} คลิป`); }
    renderDataset(); invalidateModelNotice();
  }
}

function simulatedHand(kind, phase) {
  const coords = [[.5,.83],[.41,.74],[.34,.64],[.28,.54],[.23,.45],[.43,.58],[.42,.43],[.42,.30],[.42,.19],[.51,.56],[.52,.39],[.52,.25],[.52,.13],[.59,.58],[.62,.43],[.63,.31],[.64,.22],[.66,.64],[.71,.53],[.74,.44],[.76,.35]];
  return coords.map(([x,y], i) => {
    if ((kind === 'Thumb_Up' && i >= 5) || (kind === 'Victory' && i >= 13)) y = .64 + (i % 4) * .025;
    if (kind === 'Thumb_Up' && i > 0 && i <= 4) { x = .37; y = .7 - i * .12; }
    return { x: x + Math.sin(phase) * .035, y, z: 0 };
  });
}
function startSimulation() {
  if (busy()) return;
  if (source === 'simulation') { stopSource(); return; }
  stopSource(); setMode('basic'); source = 'simulation'; $('cameraEmpty').hidden = true;
  $('sourcePill').textContent = '● ข้อมูลจำลอง · ไม่ใช่ผลจากกล้อง'; $('sourcePill').classList.add('sim-label');
  $('sourceSummary').textContent = 'โหมดซ้อมนำเสนอ'; $('liveLabel').textContent = 'SIMULATED'; $('cameraHint').textContent = 'ภาพจุดมือจำลอง · ไม่มีการประเมินความมั่นใจ'; $('handCount').textContent = '1 มือจำลอง'; $('fps').textContent = 'DEMO';
  const started = performance.now(); const kinds = ['Open_Palm', 'Thumb_Up', 'Victory'];
  const tick = () => {
    const now = performance.now(), elapsed = now - started, kind = kinds[Math.floor(elapsed / 3200) % kinds.length];
    fitCanvas(960, 720); drawHands([simulatedHand(kind, elapsed / 1000)], true);
    handlePrediction({ ...GESTURES[kind], score: 1, simulated: true }, [], now);
  };
  tick(); simTimer = setInterval(tick, 120); updateControls();
}

async function trainModel() {
  if (training) { if (trainingModel) trainingModel.stopTraining = true; return; }
  if (trainingProblem(dataset)) { toast(trainingProblem(dataset), true); return; }
  stopSource(); training = true; trainingModel = null; updateControls(); $('trainingProgress').hidden = false; $('trainingProgress').value = 0;
  $('trainingStatus').textContent = 'กำลังเตรียมโมเดล LSTM…';
  try {
    const trained = await train(dataset, (epoch, loss) => {
      $('trainingProgress').value = epoch; $('trainingStatus').textContent = `กำลังฝึก ${epoch} / 25 รอบ · loss บนชุดฝึก ${loss.toFixed(4)} (ไม่ใช่คะแนนชุดทดสอบ)`;
    }, { onModel: model => { trainingModel = model; } });
    bundle?.model.dispose(); bundle = trained; setMode('custom');
    $('trainingStatus').textContent = `ฝึกเสร็จแล้ว ${bundle.labels.length} คำ · เปิดกล้องเพื่อทดสอบกับตัวอย่างใหม่`;
    toast('ฝึกโมเดลเสร็จแล้ว เปิดกล้องแล้วลองท่าที่ฝึกได้เลย');
  } catch (error) { $('trainingStatus').textContent = error.message; toast(error.message, true); }
  finally { training = false; trainingModel = null; $('trainingProgress').hidden = true; updateControls(); }
}
async function readJSON(input) {
  const file = input.files?.[0]; if (!file) return null;
  if (file.size > 20_000_000) throw new Error('ไฟล์ใหญ่เกิน 20 MB');
  return JSON.parse(await file.text());
}

$('cameraButton').onclick = () => source === 'camera' ? stopSource() : startCamera();
$('startHero').onclick = startCamera; $('simulateHero').onclick = startSimulation; $('simulateButton').onclick = startSimulation;
$('switchButton').onclick = () => { facing = facing === 'user' ? 'environment' : 'user'; startCamera(); };
$('basicMode').onclick = () => setMode('basic');
$('customMode').onclick = () => { if (source === 'simulation') stopSource(); setMode('custom'); };
$('threshold').oninput = () => { $('thresholdValue').value = `${$('threshold').value}%`; gate.reset(); showResult(null); };
$('autoAdd').onchange = () => gate.reset();
$('addResult').onclick = () => addToken(current);
$('clearText').onclick = () => { tokens = []; renderTranscript(); };
$('copyButton').onclick = async () => { try { await navigator.clipboard.writeText(transcriptText(tokens)); toast('คัดลอกข้อความแล้ว'); } catch { toast('คัดลอกไม่ได้ ใช้ปุ่มบันทึก .txt แทนได้', true); } };
$('downloadText').onclick = () => saveFile('snapsign-transcript.txt', transcriptText(tokens), 'text/plain;charset=utf-8');
$('speakButton').onclick = () => {
  if (!('speechSynthesis' in window)) { toast('เบราว์เซอร์นี้ไม่รองรับเสียงอ่าน', true); return; }
  speechSynthesis.cancel(); const utterance = new SpeechSynthesisUtterance(transcriptText(tokens)); utterance.lang = 'th-TH';
  const voice = speechSynthesis.getVoices().find(v => v.lang.startsWith('th')); if (voice) utterance.voice = voice;
  utterance.onerror = () => toast('อ่านเสียงไม่ได้ กรุณาตรวจสอบเสียงภาษาไทยในระบบ', true); speechSynthesis.speak(utterance);
};
$('labelForm').onsubmit = event => {
  event.preventDefault(); if (busy()) return;
  const label = $('labelInput').value.trim();
  if (!label) return;
  if (dataset.classes.length >= 12 || dataset.classes.some(c => c.label === label)) { toast('ชื่อคำซ้ำ หรือเพิ่มครบ 12 คำแล้ว', true); return; }
  dataset.classes.push({ label, samples: [] }); $('labelInput').value = ''; persist(); renderDataset(); invalidateModelNotice();
};
$('trainButton').onclick = trainModel;
$('exportData').onclick = () => saveFile('snapsign-dataset.json', dataset);
$('importData').onchange = async () => {
  try {
    const data = await readJSON($('importData')); if (!data) return; validateDataset(data);
    if (totalSamples() && !confirm('แทนที่ชุดข้อมูลปัจจุบันด้วยไฟล์นี้? ควรส่งออกชุดเดิมก่อน')) return;
    dataset = data; persist(); renderDataset(); invalidateModelNotice(); toast('นำเข้าชุดข้อมูลแล้ว');
  } catch (error) { toast(`นำเข้าไม่ได้: ${error.message}`, true); } finally { $('importData').value = ''; }
};
$('exportModel').onclick = async () => { try { saveFile('snapsign-model.json', await serializeModel(bundle)); toast('บันทึกโมเดลพร้อมชื่อคำแล้ว'); } catch (error) { toast(error.message, true); } };
$('importModel').onchange = async () => {
  if (busy()) return;
  stopSource(); training = true; updateControls();
  try {
    const data = await readJSON($('importModel')); if (!data) return;
    const restored = await restoreModel(data); bundle?.model.dispose(); bundle = restored; setMode('custom');
    $('trainingStatus').textContent = `นำเข้าโมเดลแล้ว ${bundle.labels.length} คำ (ไม่รวมชุดข้อมูลฝึก)`; toast('นำเข้าโมเดลแล้ว เปิดกล้องเพื่อทดลอง');
  } catch (error) { toast(`นำเข้าโมเดลไม่ได้: ${error.message}`, true); }
  finally { training = false; $('importModel').value = ''; updateControls(); }
};
Object.values(GESTURES).forEach(gesture => {
  const pill = document.createElement('span'); pill.className = 'gesture-pill'; pill.title = gesture.hint;
  const icon = document.createElement('span'); icon.textContent = gesture.icon; pill.append(icon, document.createTextNode(gesture.label)); $('gestureList').append(pill);
});
try { const saved = localStorage.getItem(STORAGE); if (saved) dataset = validateDataset(JSON.parse(saved)); }
catch { toast('ชุดข้อมูลเดิมอ่านไม่ได้ จึงเริ่มชุดว่าง คุณยังนำเข้าไฟล์สำรองได้', true); }
document.addEventListener('visibilitychange', () => { if (document.hidden && source !== 'off') stopSource(); });
window.addEventListener('pagehide', () => { stopSource(); if ('speechSynthesis' in window) speechSynthesis.cancel(); });
renderDataset(); renderTranscript();
