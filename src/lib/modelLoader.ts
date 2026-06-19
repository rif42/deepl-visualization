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

function validateManifest(manifest: Manifest): void {
  const sections: (keyof Manifest)[] = ['input', 'mid', 'output'];
  for (const section of sections) {
    const layer = manifest[section];
    if (!layer || typeof layer !== 'object') {
      throw new Error(`Missing manifest section: ${section}`);
    }
    if (
      !layer.weight ||
      typeof layer.weight.file !== 'string' ||
      !Array.isArray(layer.weight.shape) ||
      layer.weight.shape.length < 2 ||
      layer.weight.shape.some((dim) => !Number.isInteger(dim) || dim <= 0)
    ) {
      throw new Error(`Invalid weight manifest for section: ${section}`);
    }
    if (
      !layer.bias ||
      typeof layer.bias.file !== 'string' ||
      !Array.isArray(layer.bias.shape) ||
      layer.bias.shape.length < 1 ||
      layer.bias.shape.some((dim) => !Number.isInteger(dim) || dim <= 0)
    ) {
      throw new Error(`Invalid bias manifest for section: ${section}`);
    }
  }
}

export async function loadModel(baseUrl: string): Promise<tf.LayersModel> {
  const manifestResponse = await fetch(`${baseUrl}/manifest.json`);
  if (!manifestResponse.ok) {
    throw new Error(`Failed to load manifest: ${manifestResponse.status}`);
  }
  const manifest: Manifest = await manifestResponse.json();
  validateManifest(manifest);

  const inputFeatures = manifest.input.weight.shape[1];
  const side = Math.sqrt(inputFeatures);
  const inputShape = Number.isInteger(side) ? [side, side] : [inputFeatures];

  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape }));
  model.add(tf.layers.dense({ units: manifest.input.weight.shape[0], activation: 'relu', name: 'hidden1' }));
  model.add(tf.layers.dense({ units: manifest.mid.weight.shape[0], activation: 'relu', name: 'hidden2' }));
  model.add(tf.layers.dense({ units: manifest.output.weight.shape[0], activation: 'softmax', name: 'output' }));

  let hidden1Weights: tf.Tensor | null = null;
  let hidden1Bias: tf.Tensor | null = null;
  try {
    const hidden1WeightsRaw = await fetchBinary(baseUrl, manifest.input.weight.file, manifest.input.weight.shape);
    hidden1Weights = hidden1WeightsRaw.transpose();
    hidden1WeightsRaw.dispose();
    hidden1Bias = await fetchBinary(baseUrl, manifest.input.bias.file, manifest.input.bias.shape);
    model.getLayer('hidden1').setWeights([hidden1Weights, hidden1Bias]);
  } finally {
    hidden1Weights?.dispose();
    hidden1Bias?.dispose();
  }

  let hidden2Weights: tf.Tensor | null = null;
  let hidden2Bias: tf.Tensor | null = null;
  try {
    const hidden2WeightsRaw = await fetchBinary(baseUrl, manifest.mid.weight.file, manifest.mid.weight.shape);
    hidden2Weights = hidden2WeightsRaw.transpose();
    hidden2WeightsRaw.dispose();
    hidden2Bias = await fetchBinary(baseUrl, manifest.mid.bias.file, manifest.mid.bias.shape);
    model.getLayer('hidden2').setWeights([hidden2Weights, hidden2Bias]);
  } finally {
    hidden2Weights?.dispose();
    hidden2Bias?.dispose();
  }

  let outputWeights: tf.Tensor | null = null;
  let outputBias: tf.Tensor | null = null;
  try {
    const outputWeightsRaw = await fetchBinary(baseUrl, manifest.output.weight.file, manifest.output.weight.shape);
    outputWeights = outputWeightsRaw.transpose();
    outputWeightsRaw.dispose();
    outputBias = await fetchBinary(baseUrl, manifest.output.bias.file, manifest.output.bias.shape);
    model.getLayer('output').setWeights([outputWeights, outputBias]);
  } finally {
    outputWeights?.dispose();
    outputBias?.dispose();
  }

  return model;
}
