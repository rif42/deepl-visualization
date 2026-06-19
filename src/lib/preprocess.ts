const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

export function normalizeCanvasPixels(
  imageData: Uint8ClampedArray,
  size: number
): Float32Array {
  const pixelCount = size * size;
  const output = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    const r = imageData[i * 4];
    const gray = r / 255.0;
    output[i] = (gray - MNIST_MEAN) / MNIST_STD;
  }

  return output;
}
