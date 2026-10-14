"""Download pinned browser runtime assets. No Python dependencies required."""
from pathlib import Path
from urllib.request import urlopen, Request
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21'
FILES = {
    'vision/vision_bundle.mjs': f'{VISION}/vision_bundle.mjs',
    'vision/wasm/vision_wasm_internal.js': f'{VISION}/wasm/vision_wasm_internal.js',
    'vision/wasm/vision_wasm_internal.wasm': f'{VISION}/wasm/vision_wasm_internal.wasm',
    'vision/wasm/vision_wasm_nosimd_internal.js': f'{VISION}/wasm/vision_wasm_nosimd_internal.js',
    'vision/wasm/vision_wasm_nosimd_internal.wasm': f'{VISION}/wasm/vision_wasm_nosimd_internal.wasm',
    'tf.min.js': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    'gesture_recognizer.task': 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
}

def main():
    manifest = {}
    for name, url in FILES.items():
        dest = ROOT / 'vendor' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f'Downloading {name} ...', flush=True)
            with urlopen(Request(url, headers={'User-Agent': 'SNAPSIGN-Demo/0.1'}), timeout=120) as response:
                body = response.read()
            tmp = dest.with_suffix(dest.suffix + '.part')
            tmp.write_bytes(body)
            tmp.replace(dest)
        body = dest.read_bytes()
        manifest[name] = {'url': url, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
        print(f'OK {name} ({len(body):,} bytes)', flush=True)
    (ROOT / 'vendor' / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Ready. Run: npm start')

if __name__ == '__main__':
    main()
