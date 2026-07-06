import { tf } from './tf';
import type { LayersModel, Tensor, SymbolicTensor } from '@tensorflow/tfjs';

export interface PredictionResult {
  probabilities: number[];
  activations: Tensor[];
}

export async function runPrediction(
  model: LayersModel,
  normalizedPixels: Float32Array
): Promise<PredictionResult> {
  const inputTensor = tf.tensor(normalizedPixels, [1, 28, 28]);

  // Capture outputs from all Dense layers: hidden1, hidden2, output
  const outputs = model.layers
    .filter((layer) => layer.getClassName() === 'Dense')
    .map((layer) => layer.output as SymbolicTensor);

  const activationModel = tf.model({ inputs: model.input, outputs });
  const predictResult = await activationModel.predict(inputTensor);
  const activations = Array.isArray(predictResult) ? predictResult : [predictResult];

  const probabilities = Array.from(await activations[activations.length - 1].data() as Float32Array);

  inputTensor.dispose();
  // Do NOT dispose activationModel — it shares layers with the original model.

  return { probabilities, activations };
}
