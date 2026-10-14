import { FRAMES, FEATURES, trainingProblem, rankPrediction } from './core.mjs';

let loading;
export async function tensorflow() {
  if (!loading) loading = (async () => {
    if (!globalThis.tf) await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = './vendor/tf.min.js';
      script.onload = resolve;
      script.onerror = () => reject(new Error('ไม่พบ TensorFlow.js ให้รัน python scripts/download_assets.py'));
      document.head.append(script);
    });
    await globalThis.tf.ready();
    return globalThis.tf;
  })().catch(error => { loading = null; throw error; });
  return loading;
}

export async function train(data, onEpoch, { epochs = 25, onModel } = {}) {
  const problem = trainingProblem(data);
  if (problem) throw new Error(problem);
  const tf = await tensorflow();
  const labels = data.classes.map(c => c.label);
  const samples = data.classes.flatMap(c => c.samples);
  const targets = data.classes.flatMap((c, i) => c.samples.map(() => i));
  const model = tf.sequential();
  let xs, ys;
  try {
    model.add(tf.layers.lstm({ units: 24, inputShape: [FRAMES, FEATURES] }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.15 }));
    model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(0.003), loss: 'categoricalCrossentropy' });
    xs = tf.tensor3d(samples);
    ys = tf.tidy(() => tf.oneHot(tf.tensor1d(targets, 'int32'), labels.length));
    onModel?.(model);
    await model.fit(xs, ys, { epochs, batchSize: Math.min(8, samples.length), shuffle: true, callbacks: { onEpochEnd: async (epoch, logs) => { await onEpoch?.(epoch + 1, logs.loss); await tf.nextFrame(); } } });
    if (model.stopTraining) throw new Error('ยกเลิกการฝึกแล้ว โมเดลเดิมยังใช้งานได้');
    return { model, labels };
  } catch (error) { model.dispose(); throw error; }
  finally { xs?.dispose(); ys?.dispose(); }
}

export async function predict(bundle, frames) {
  const tf = await tensorflow();
  const scores = tf.tidy(() => {
    const input = tf.tensor3d([frames], [1, FRAMES, FEATURES]);
    return Array.from(bundle.model.predict(input).dataSync());
  });
  return rankPrediction(scores, bundle.labels);
}

function base64(buffer) {
  const bytes = new Uint8Array(buffer);
  let value = '';
  for (let i = 0; i < bytes.length; i += 8192) value += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(value);
}

export async function serializeModel(bundle) {
  let artifacts;
  await bundle.model.save({ save: async value => { artifacts = value; return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } }; } });
  return { schema: 'snapsign-lstm-v1', frames: FRAMES, features: FEATURES, labels: bundle.labels, modelTopology: artifacts.modelTopology, weightSpecs: artifacts.weightSpecs, weights: base64(artifacts.weightData) };
}

export async function restoreModel(data) {
  if (data?.schema !== 'snapsign-lstm-v1' || data.frames !== FRAMES || data.features !== FEATURES || !Array.isArray(data.labels) || data.labels.length < 2 || data.labels.length > 12 || data.labels.some(l => typeof l !== 'string' || !l.trim() || l.length > 40) || new Set(data.labels).size !== data.labels.length || typeof data.weights !== 'string' || data.weights.length > 20_000_000) throw new Error('รูปแบบโมเดลไม่ถูกต้อง');
  // Accept only the small architecture emitted by this demo, before allocating tensors.
  const layers = data.modelTopology?.config?.layers;
  if (!Array.isArray(layers) || layers.length !== 4 || layers.map(l => l.class_name).join(',') !== 'LSTM,Dense,Dropout,Dense' || layers[0].config.units !== 24 || layers[1].config.units !== 24 || layers[3].config.units !== data.labels.length) throw new Error('รองรับเฉพาะโมเดล LSTM ที่บันทึกจากเดโม่นี้');
  const inputShape = layers[0].config.batch_input_shape;
  if (!Array.isArray(inputShape) || inputShape.length !== 3 || inputShape[0] !== null || inputShape[1] !== FRAMES || inputShape[2] !== FEATURES) throw new Error('ขนาดข้อมูลของโมเดลไม่ตรงกัน');
  if (!Array.isArray(data.weightSpecs) || data.weightSpecs.length > 12 || data.weightSpecs.some(s => s.dtype !== 'float32' || !Array.isArray(s.shape) || s.shape.length > 2 || s.shape.some(n => !Number.isInteger(n) || n < 1 || n > 200))) throw new Error('ขนาดน้ำหนักโมเดลไม่ถูกต้อง');
  const binary = atob(data.weights);
  const expectedBytes = data.weightSpecs.reduce((sum, spec) => sum + spec.shape.reduce((a, b) => a * b, 1) * 4, 0);
  if (binary.length !== expectedBytes) throw new Error('ไฟล์น้ำหนักโมเดลไม่ครบ');
  const tf = await tensorflow();
  const weightData = Uint8Array.from(binary, c => c.charCodeAt(0)).buffer;
  const model = await tf.loadLayersModel({ load: async () => ({ modelTopology: data.modelTopology, weightSpecs: data.weightSpecs, weightData }) });
  if (model.outputs[0].shape[1] !== data.labels.length) { model.dispose(); throw new Error('จำนวนคำไม่ตรงกับโมเดล'); }
  return { model, labels: data.labels };
}
