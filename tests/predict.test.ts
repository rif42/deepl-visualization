import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { runPrediction } from '../src/lib/predict';

describe('runPrediction', () => {
  it('returns probabilities and activations for all dense layers', async () => {
    const input = tf.input({ shape: [28, 28] });
    const flat = tf.layers.flatten().apply(input);
    const hidden = tf.layers.dense({ units: 4, activation: 'relu' }).apply(flat);
    const output = tf.layers.dense({ units: 2, activation: 'softmax' }).apply(hidden);
    const model = tf.model({ inputs: input, outputs: output as tf.SymbolicTensor });

    const pixels = new Float32Array(28 * 28).fill(0.5);
    const result = await runPrediction(model, pixels);

    expect(result.probabilities).toHaveLength(2);
    expect(result.activations).toHaveLength(2); // hidden + output
    expect(result.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    result.activations.forEach((t) => t.dispose());
  });

  it('can predict multiple times on the same model', async () => {
    const input = tf.input({ shape: [28, 28] });
    const flat = tf.layers.flatten().apply(input);
    const hidden = tf.layers.dense({ units: 4, activation: 'relu' }).apply(flat);
    const output = tf.layers.dense({ units: 2, activation: 'softmax' }).apply(hidden);
    const model = tf.model({ inputs: input, outputs: output as tf.SymbolicTensor });

    const pixels = new Float32Array(28 * 28).fill(0.5);
    const result1 = await runPrediction(model, pixels);
    result1.activations.forEach((t) => t.dispose());

    const result2 = await runPrediction(model, pixels);
    expect(result2.probabilities).toHaveLength(2);
    expect(result2.activations).toHaveLength(2);
    result2.activations.forEach((t) => t.dispose());
  });
});
