<video src="./demo.mp4" controls loop muted playsinline width="100%"></video>

[Open demo.mp4](./demo.mp4)

# MNIST 3D Inference Visualizer

Draw a digit, click **Predict**, and watch a small neural network classify it in 3D.

This is a static browser app built with Astro, TensorFlow.js, and Three.js. It loads a pre-trained dense MNIST model, runs inference locally, and turns the forward pass into an interactive network view: input pixels, hidden activations, output probabilities, and the strongest learned weights.

## Features

- 28×28 drawing canvas with clear/reset.
- Pre-trained MLP classifier from `dacorvo/mnist-mlp`, converted for TensorFlow.js.
- One-click prediction, fully client-side.
- 3D layer visualization with activation-driven color and scale.
- Top-K weight lines so the network stays readable instead of drawing ~109k connections.
- Probability bars for digits 0–9.

## How it works

1. You draw on a 28×28 canvas.
2. The bitmap is inverted, MNIST-normalized, and flattened to `[1, 784]`.
3. TensorFlow.js runs the model and captures each layer output.
4. Three.js updates neuron colors/sizes and the strongest weight lines.
5. The prediction panel shows the predicted digit and class probabilities.

Layer sizes are read from the loaded model, so the scene can adapt to another dense MNIST network.

## Run

```bash
bun install
bun run dev     # local dev
bun run build   # static export to dist/
```

Deploy `dist/` to any static host. No backend required.

## Layout

```txt
public/model/              TensorFlow.js model files
src/components/            canvas + prediction UI
src/lib/                   preprocessing, model loading, prediction orchestration
src/visualization/         Three.js scene and activation rendering
```

Design details: [`docs/superpowers/specs/2026-06-19-mnist-3d-visualizer-design.md`](docs/superpowers/specs/2026-06-19-mnist-3d-visualizer-design.md)
