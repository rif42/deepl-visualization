import * as tf from '@tensorflow/tfjs';

interface TensorInfo {
  file: string;
  shape: number[];
}

interface LayerManifest {
  weight: TensorInfo;
  bias: TensorInfo;
}

interface Manifest {
  input: LayerManifest;
  mid: LayerManifest;
  output: LayerManifest;
}

async function fetchBinary(baseUrl: string, filename: string, shape: number[]): Promise<tf.Tensor> {
  const response = await fetch(`${baseUrl}/${filename}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${filename}: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  const values = new Float32Array(buffer);
  return tf.tensor(values, shape);
}

export async function loadModel(baseUrl: string): Promise<tf.LayersModel> {
  const manifestResponse = await fetch(`${baseUrl}/manifest.json`);
  if (!manifestResponse.ok) {
    throw new Error(`Failed to load manifest: ${manifestResponse.status}`);
  }
  const manifest: Manifest = await manifestResponse.json();

  const inputFeatures = manifest.input.weight.shape[1];
  const side = Math.sqrt(inputFeatures);
  const inputShape = Number.isInteger(side) ? [side, side] : [inputFeatures];

  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape }));
  model.add(tf.layers.dense({ units: manifest.input.weight.shape[0], activation: 'relu', name: 'hidden1' }));
  model.add(tf.layers.dense({ units: manifest.mid.weight.shape[0], activation: 'relu', name: 'hidden2' }));
  model.add(tf.layers.dense({ units: manifest.output.weight.shape[0], activation: 'softmax', name: 'output' }));

  const hidden1WeightsRaw = await fetchBinary(baseUrl, manifest.input.weight.file, manifest.input.weight.shape);
  const hidden1Weights = hidden1WeightsRaw.transpose();
  hidden1WeightsRaw.dispose();
  const hidden1Bias = await fetchBinary(baseUrl, manifest.input.bias.file, manifest.input.bias.shape);
  model.getLayer('hidden1').setWeights([hidden1Weights, hidden1Bias]);
  hidden1Weights.dispose();
  hidden1Bias.dispose();

  const hidden2WeightsRaw = await fetchBinary(baseUrl, manifest.mid.weight.file, manifest.mid.weight.shape);
  const hidden2Weights = hidden2WeightsRaw.transpose();
  hidden2WeightsRaw.dispose();
  const hidden2Bias = await fetchBinary(baseUrl, manifest.mid.bias.file, manifest.mid.bias.shape);
  model.getLayer('hidden2').setWeights([hidden2Weights, hidden2Bias]);
  hidden2Weights.dispose();
  hidden2Bias.dispose();

  const outputWeightsRaw = await fetchBinary(baseUrl, manifest.output.weight.file, manifest.output.weight.shape);
  const outputWeights = outputWeightsRaw.transpose();
  outputWeightsRaw.dispose();
  const outputBias = await fetchBinary(baseUrl, manifest.output.bias.file, manifest.output.bias.shape);
  model.getLayer('output').setWeights([outputWeights, outputBias]);
  outputWeights.dispose();
  outputBias.dispose();

  return model;
}
