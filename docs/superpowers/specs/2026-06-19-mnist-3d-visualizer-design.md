# MNIST 3D Inference Visualizer — Design Spec

**Date:** 2026-06-19  
**Status:** Draft (pending review)

## 1. Goal

Build a browser-based visualization app where a user draws a handwritten digit on a canvas, clicks **Predict**, and sees a 3D neural network render the inference process: input image, hidden-layer activations, output probabilities, and the strongest weights connecting neurons.

## 2. Scope

**In scope:**
- 28×28 drawing canvas with clear button.
- Fixed, pre-trained dense MNIST classifier.
- One-click prediction.
- 3D network topology visualization using Three.js.
- Live activation coloring on neurons.
- Top-K strongest weight lines rendered between layers.
- Output panel showing predicted digit + probability bars.

**Out of scope:**
- User-customizable network architecture (removed after gap discussion).
- Training inside the browser.
- Backend/server inference.
- CNN-style conv-layer visualization (dense model only).

## 3. Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Framework | Astro | Neat code organization, static-site output, no backend. |
| NN runtime | TensorFlow.js | Loads pre-trained dense model and exposes per-layer activations. |
| 3D rendering | Three.js | Standard, well-documented WebGL library for topology visualization. |
| Language | TypeScript | Type safety for model parsing and scene logic. |
| Styling | Plain CSS / scoped Astro styles | No heavy UI framework needed. |

## 4. Fixed Model Architecture

Use a **pre-trained dense MNIST model** matching the input/output shapes:

- Input: `28 × 28` grayscale → flattened to `784` floats in `[0, 1]`.
- Hidden layers: dense with ReLU (target `784 → 128 → 64`).
- Output: `10` neurons with softmax → probabilities for digits `0–9`.

The exact hidden-layer sizes depend on the pre-trained model we obtain. The visualization must read layer shapes from the model at load time rather than hard-coding them.

## 5. Components

| Component | Responsibility |
|-----------|----------------|
| `DrawingCanvas.astro` | 28×28 HTML5 canvas; mouse/touch drawing; clear button; emits `onDrawComplete` bitmap. |
| `PredictionPanel.astro` | Displays predicted digit and 10 probability bars. |
| `NetworkVisualization.ts` | Three.js scene manager: creates input plane, hidden-layer neuron grids, output neurons; updates colors/scale from activations; renders top-K weight lines. |
| `modelLoader.ts` | Loads the pre-trained TensorFlow.js model from `public/model/`. Runs forward pass and captures intermediate activations. |
| `preprocess.ts` | Inverts canvas pixels (white stroke on black background), normalizes to `[0, 1]`, reshapes to `[1, 784]`. |
| `predict.ts` | Orchestrates: preprocess → model forward → extract activations → update UI and 3D scene. |

## 6. Data Flow

```
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

Inference is triggered by a **Predict** button, not on every stroke, to keep the UI calm and avoid expensive model calls while drawing.

## 7. 3D Visualization Design

### 7.1 Scene Objects

| Layer | Representation |
|-------|----------------|
| Input | A 28×28 grid plane textured with the drawn digit. |
| Hidden layers | Spheres arranged in regular 3D grids sized to match the loaded model layer dimensions. |
| Output | 10 spheres in a vertical column, labeled 0–9. |

For the target model (`784 → 128 → 64 → 10`), this means 128 hidden-1 spheres and 64 hidden-2 spheres. The implementation reads actual layer sizes from the model so it adapts if a different pre-trained dense model is used.

### 7.2 Activation Encoding

- **Color:** interpolate from dark gray/black (inactive) through green/yellow to red (highly active).
- **Scale:** sphere radius grows slightly with activation magnitude.
- **Output layer:** the predicted digit sphere is largest/highlighted; others dimmer.

### 7.3 Weight Lines

A dense model has roughly 109,000 connections, so we do **not** draw all of them.

- Strategy: **Top-K strongest weights** per layer transition.
- Default `K = 100` per transition (configurable).
- Line opacity/width proportional to absolute weight magnitude.
- Color indicates sign: blue for negative, orange for positive.

Use `THREE.InstancedMesh` or merged line geometry for performance.

### 7.4 Camera & Interaction

- OrbitControls: rotate, zoom, pan.
- Initial camera angle shows all layers clearly.
- Optional: reset camera button.

## 8. Model Loading

- Store the pre-trained model in `public/model/`.
- Preferred format: TensorFlow.js layers model (`model.json` + weight shards).
- If the available model is Keras HDF5, convert it with the `@tensorflow/tfjs-converter` CLI before build.
- `modelLoader.ts` reads the model, inspects layer shapes, and exposes a `predictActivations(inputTensor)` function returning an array of layer outputs.

## 9. Build & Deployment

```bash
# Model is already present in public/model/
npm install
npm run dev     # local development
npm run build   # static export to dist/
```

- No backend required.
- Deploy `dist/` to any static host: Vercel, Netlify, GitHub Pages.
- Add `.superpowers/` to `.gitignore` (visual companion artifacts).

## 10. Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| Fixed architecture | Pre-training every possible architecture is infeasible; gap discussion removed user customization. |
| Pre-trained model | User explicitly chose not to train a custom model. |
| Dense model only | CNN feature maps are harder to map to the "neuron sphere" mental model. |
| 28×28 canvas | Downscaled from the original 64×64 idea to match standard MNIST input and keep visualization tractable. |
| Predict button | Chosen over live inference to reduce model calls while drawing. |
| Top-K weight lines | Balances visual clarity and performance vs. drawing all ~109k connections. |

## 11. Success Criteria

- [ ] User can draw a digit and click Predict.
- [ ] Predicted digit matches a reasonable MNIST model (≥90% accuracy on clear digits).
- [ ] 3D scene renders without frame drops on a modern laptop.
- [ ] Neuron colors update to reflect activations after each prediction.
- [ ] Top-K weight lines are visible and update on each prediction.
- [ ] App builds as a static site and runs without a backend.

## 12. Open Questions

1. Which exact pre-trained dense MNIST model will we use? (Need to locate or prepare one in TensorFlow.js format.)
2. What `K` value for top-K weight lines feels right visually?
3. Do we need touch support for drawing on mobile devices?
