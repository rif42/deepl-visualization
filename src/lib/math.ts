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
