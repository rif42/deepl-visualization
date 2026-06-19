import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { runPrediction } from '../src/lib/predict';

describe('runPrediction', () => {
  it('returns probabilities and activations', async () => {
    const input = tf.input({ shape: [28, 28] });
    const flat = tf.layers.flatten().apply(input);
    const output = tf.layers.dense({ units: 2, activation: 'softmax' }).apply(flat);
    const model = tf.model({ inputs: input, outputs: output as tf.SymbolicTensor });

    const pixels = new Float32Array(28 * 28).fill(0.5);
    const result = await runPrediction(model, pixels);

    expect(result.probabilities).toHaveLength(2);
    expect(result.activations).toHaveLength(1);
    expect(result.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
  });
});
