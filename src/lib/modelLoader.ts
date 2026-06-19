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

async function loadLayerWeights(
  baseUrl: string,
  layer: tf.layers.Layer,
  layerManifest: LayerManifest
): Promise<void> {
  let rawWeights: tf.Tensor | null = null;
  let weights: tf.Tensor | null = null;
  let bias: tf.Tensor | null = null;
  try {
    rawWeights = await fetchBinary(baseUrl, layerManifest.weight.file, layerManifest.weight.shape);
    weights = rawWeights.transpose();
    rawWeights.dispose();
    rawWeights = null;
    bias = await fetchBinary(baseUrl, layerManifest.bias.file, layerManifest.bias.shape);
    layer.setWeights([weights, bias]);
  } finally {
    rawWeights?.dispose();
    weights?.dispose();
    bias?.dispose();
  }
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

  await loadLayerWeights(baseUrl, model.getLayer('hidden1'), manifest.input);
  await loadLayerWeights(baseUrl, model.getLayer('hidden2'), manifest.mid);
  await loadLayerWeights(baseUrl, model.getLayer('output'), manifest.output);

  return model;
}
