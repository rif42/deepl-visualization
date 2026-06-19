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
  const inputTensor = tf.tensor(normalizedPixels, [1, 28, 28]);

  const outputs = model.outputs.map((output) => output as tf.SymbolicTensor);
  const activationModel = tf.model({ inputs: model.input, outputs });
  const predictResult = await activationModel.predict(inputTensor);
  const activations = Array.isArray(predictResult) ? predictResult : [predictResult];

  const probabilities = Array.from(await activations[activations.length - 1].data() as Float32Array);

  inputTensor.dispose();
  // Do NOT dispose activationModel — it shares layers with the original model.

  return { probabilities, activations };
}
