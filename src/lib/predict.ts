import * as tf from '@tensorflow/tfjs';
import type { LayersModel } from '@tensorflow/tfjs';

export interface PredictionResult {
  probabilities: number[];
  activations: tf.Tensor[];
}

export async function runPrediction(
  model: LayersModel,
  normalizedPixels: Float32Array
): Promise<PredictionResult> {
  if (!tf.getBackend()) {
    await tf.setBackend('cpu');
  }

  const inputTensor = tf.tensor(normalizedPixels, [1, 28, 28]);

  const outputs = model.outputs.map((output) => output as tf.SymbolicTensor);
  const activationModel = tf.model({ inputs: model.input, outputs });
  const rawActivations = activationModel.predict(inputTensor);
  const activations = Array.isArray(rawActivations)
    ? rawActivations
    : [rawActivations];

  const probabilities = Array.from(
    (await activations[activations.length - 1].data()) as Float32Array
  );

  inputTensor.dispose();
  activationModel.dispose();

  return { probabilities, activations };
}
