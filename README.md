<video src="./demo.mp4" controls loop muted playsinline width="100%"></video>

[Download/open demo.mp4](./demo.mp4)

# MNIST 3D Inference Visualizer

Browser app where you draw a handwritten digit, click **Predict**, and watch a 3D neural network render the inference: input image, hidden-layer activations, output probabilities, and strongest weights between neurons.

## What it does

- 28×28 drawing canvas with clear button.
- Fixed, pre-trained dense MNIST classifier.
- One-click prediction.
- 3D network topology visualization with Three.js.
- Live activation coloring on neurons.
- Top-K strongest weight lines between layers.
- Output panel with predicted digit and probability bars.

Out of scope: custom architectures, in-browser training, backend inference, and CNN conv-layer visualization.

## Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Framework | Astro | Static-site output, neat organization, no backend |
| NN runtime | TensorFlow.js | Loads pre-trained dense model, exposes per-layer activations |
| 3D rendering | Three.js | Standard WebGL library for topology visualization |
| Language | TypeScript | Type-safe model parsing and scene logic |
| Styling | Plain CSS / scoped Astro styles | No heavy UI framework needed |

## Model

Uses the pre-trained dense MNIST model **`dacorvo/mnist-mlp`** from Hugging Face, converted to TensorFlow.js format.

- Input: `28 × 28` grayscale, flattened to `784` floats.
- Normalization: `(pixel / 255 - 0.1307) / 0.3081`.
- Hidden layers: `784 → ReLU(256) → ReLU(256) → 10`.
- Output: `10` neurons with softmax for digits `0–9`.

The visualizer reads layer shapes from the loaded model, so neuron counts adapt if a different dense MNIST model is used.

## Components

| Component | Responsibility |
| --- | --- |
| `DrawingCanvas.astro` | 28×28 HTML5 canvas, mouse/touch drawing, clear button, emits bitmap |
| `PredictionPanel.astro` | Shows predicted digit and 10 probability bars |
| `NetworkVisualization.ts` | Three.js scene manager: input plane, hidden neuron grids, output neurons, activation colors/scale, top-K weight lines |
| `modelLoader.ts` | Loads TensorFlow.js model from `public/model/`, runs forward pass, captures activations |
| `preprocess.ts` | Inverts canvas pixels, normalizes to `[0, 1]`, reshapes to `[1, 784]` |
| `predict.ts` | Orchestrates preprocess → model forward → extract activations → update UI and 3D scene |

## Data flow

```txt
User draws on canvas
        ↓
[DrawingCanvas] emits 28×28 ImageData
        ↓
[preprocess] → float32 tensor [1, 784], white stroke = 1.0
        ↓
[modelLoader] runs TensorFlow.js forward pass
        ↓
Captures layer outputs: input, hidden1, hidden2, output
        ↓
[NetworkVisualization] updates neuron colors/scale and top-K weight lines
        ↓
[PredictionPanel] shows predicted digit + probability bars
```

Inference runs only when **Predict** is clicked, not on every stroke, to keep the UI calm and avoid expensive model calls while drawing.

## 3D visualization

### Scene objects

| Layer | Representation |
| --- | --- |
| Input | 28×28 grid plane textured with the drawn digit |
| Hidden layers | Spheres in regular 3D grids sized to model layer dimensions |
| Output | 10 spheres in a vertical column, labeled 0–9 |

### Activation encoding

- Color interpolates from dark gray/black (inactive) through dark yellow to green (highly active).
- Sphere radius grows slightly with activation magnitude.
- Output layer highlights the predicted digit; other outputs stay dimmer.

### Weight lines

Dense MNIST models have ~109k connections, so the app draws only the **Top-K strongest weights** per layer transition.

- Default: `K = 100` per transition, configurable.
- Line opacity/width scale with absolute weight magnitude.
- Color shows sign: blue negative, orange positive.
- Use `THREE.InstancedMesh` or merged line geometry for performance.

### Camera and interaction

- OrbitControls: rotate, zoom, pan.
- Initial camera angle shows all layers clearly.
- Optional reset camera button.
