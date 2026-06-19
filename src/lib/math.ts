export interface WeightEntry {
  index: number;
  value: number;
}

export function topKWeights(weights: Float32Array, k: number): WeightEntry[] {
  const entries: WeightEntry[] = Array.from(weights).map((value, index) => ({
    index,
    value,
  }));

  entries.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return entries.slice(0, k);
}

/**
 * Returns the indices of the top-k largest absolute values in `scores`.
 * Faster than `topKWeights` for repeated dynamic scoring because it avoids
 * creating an array of objects.
 */
export function topKIndices(scores: Float32Array, k: number): number[] {
  const indices = new Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    indices[i] = i;
  }

  indices.sort((a, b) => Math.abs(scores[b]) - Math.abs(scores[a]));
  return indices.slice(0, k);
}
