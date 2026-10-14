export const FRAMES = 16;
export const FEATURES = 132;
export const SCHEMA = 'snapsign-sequences-v1';
export const SAMPLE_MS = 125;
export const GESTURES = {
  Open_Palm: { label: 'ฝ่ามือเปิด', icon: '✋', hint: 'กางนิ้วทั้งห้า หันฝ่ามือเข้ากล้อง' },
  Closed_Fist: { label: 'กำมือ', icon: '✊', hint: 'กำมือโดยให้นิ้วโป้งอยู่ด้านนอก' },
  Thumb_Up: { label: 'ยกนิ้วโป้ง', icon: '👍', hint: 'กำมือแล้วชูนิ้วโป้งขึ้น' },
  Thumb_Down: { label: 'คว่ำนิ้วโป้ง', icon: '👎', hint: 'กำมือแล้วชี้นิ้วโป้งลง' },
  Victory: { label: 'ชูสองนิ้ว', icon: '✌️', hint: 'ชูนิ้วชี้และนิ้วกลางเป็นรูปตัว V' },
  Pointing_Up: { label: 'ชูนิ้วชี้', icon: '☝️', hint: 'ชูนิ้วชี้ขึ้นเพียงนิ้วเดียว' },
  ILoveYou: { label: 'ท่า I Love You (ASL)', icon: '🤟', hint: 'ชูนิ้วโป้ง นิ้วชี้ และนิ้วก้อย' }
};

// Stable left/right slots: presence + wrist location + 21 relative xyz points.
// Wrist coordinates preserve movement; palm scaling reduces distance sensitivity.
export function extractFeatures(result) {
  const slots = [Array(66).fill(0), Array(66).fill(0)];
  const used = new Set();
  (result.landmarks || []).slice(0, 2).forEach((points, index) => {
    if (points.length !== 21 || points.some(p => ![p.x, p.y, p.z].every(Number.isFinite))) return;
    const side = result.handedness?.[index]?.[0]?.categoryName;
    const slot = side === 'Left' ? 0 : side === 'Right' ? 1 : index;
    if (used.has(slot)) return;
    used.add(slot);
    const wrist = points[0];
    const scale = Math.max(0.03, Math.hypot(points[9].x - wrist.x, points[9].y - wrist.y));
    slots[slot] = [1, (wrist.x - 0.5) * 2, (wrist.y - 0.5) * 2,
      ...points.flatMap(p => [(p.x - wrist.x) / scale, (p.y - wrist.y) / scale, (p.z - wrist.z) / scale])];
  });
  return slots.flat();
}

export function hasHand(features) { return features[0] === 1 || features[66] === 1; }

export function validateDataset(data) {
  if (data?.schema !== SCHEMA || data.frames !== FRAMES || data.features !== FEATURES) throw new Error('รูปแบบชุดข้อมูลไม่ตรงกับเดโม่นี้');
  if (!Array.isArray(data.classes) || data.classes.length > 12) throw new Error('รองรับได้ไม่เกิน 12 คำ');
  const labels = new Set();
  let count = 0;
  for (const item of data.classes) {
    if (typeof item.label !== 'string' || !item.label.trim() || item.label.length > 40 || labels.has(item.label)) throw new Error('ชื่อคำต้องไม่ว่าง ไม่ซ้ำ และยาวไม่เกิน 40 ตัวอักษร');
    labels.add(item.label);
    if (!Array.isArray(item.samples) || item.samples.length > 40) throw new Error('รองรับไม่เกิน 40 คลิปต่อคำ');
    count += item.samples.length;
    for (const sample of item.samples) {
      if (!Array.isArray(sample) || sample.length !== FRAMES || sample.some(frame => !Array.isArray(frame) || frame.length !== FEATURES || frame.some(n => !Number.isFinite(n) || Math.abs(n) > 100))) throw new Error('ข้อมูลจุดสำคัญของมือไม่ถูกต้อง');
      if (sample.filter(hasHand).length < Math.ceil(FRAMES * 0.75)) throw new Error('คลิปต้องตรวจพบมืออย่างน้อย 75% ของเฟรม');
    }
  }
  if (count > 300) throw new Error('รองรับชุดข้อมูลได้ไม่เกิน 300 คลิป');
  return data;
}

export function emptyDataset() { return { schema: SCHEMA, frames: FRAMES, features: FEATURES, classes: [] }; }

export function trainingProblem(data) {
  if (data.classes.length < 2) return 'เพิ่มอย่างน้อย 2 คำก่อนฝึกโมเดล';
  if (data.classes.some(item => item.samples.length < 3)) return 'เก็บอย่างน้อยคำละ 3 คลิป (แนะนำ 10 คลิปขึ้นไป)';
  return '';
}

export class StabilityGate {
  constructor(holdMs = 650, releaseMs = 500) {
    this.holdMs = holdMs; this.releaseMs = releaseMs; this.reset();
  }
  reset() { this.candidate = ''; this.since = 0; this.emitted = ''; this.absentSince = null; }
  update(label, score, threshold, now) {
    if (!label || score < threshold) {
      this.candidate = ''; this.since = now;
      this.absentSince ??= now;
      if (now - this.absentSince >= this.releaseMs) this.emitted = '';
      return false;
    }
    this.absentSince = null;
    if (this.candidate !== label) { this.candidate = label; this.since = now; return false; }
    if (now - this.since >= this.holdMs && this.emitted !== label) { this.emitted = label; return true; }
    return false;
  }
}

export function transcriptText(tokens) { return tokens.map(t => (t.simulated ? '[จำลอง] ' : '') + t.label).join(' '); }

export function rankPrediction(scores, labels) {
  if (scores.length !== labels.length || !scores.length || [...scores].some(n => !Number.isFinite(n))) return [];
  return labels.map((label, i) => ({ label, score: scores[i] })).sort((a, b) => b.score - a.score);
}
