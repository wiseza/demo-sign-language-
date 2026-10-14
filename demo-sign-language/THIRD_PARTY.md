**Third-party assets**

Runtime assets are downloaded by `scripts/download_assets.py` into `vendor/`. The manifest records original URLs, file sizes and SHA-256 hashes. No external font, analytics or image service is used.

- MediaPipe Tasks Vision 0.10.21: [source and Apache-2.0 license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE), [npm package](https://www.npmjs.com/package/@mediapipe/tasks-vision/v/0.10.21).
- MediaPipe Gesture Recognizer model, float16/version 1: [model documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer), [Google-hosted model](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task). The built-in categories are general gesture categories; this is not a pretrained Thai Sign Language model.
- TensorFlow.js 4.22.0: [source and Apache-2.0 license](https://github.com/tensorflow/tfjs/blob/master/LICENSE).
- Playwright 1.56.1 is a development dependency used for browser verification: [Apache-2.0 license](https://github.com/microsoft/playwright/blob/main/LICENSE).

Emoji rendering depends on the operating system. The hand illustration in the interface is drawn as SVG in this project.
