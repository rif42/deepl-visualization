import { describe, it, expect } from 'vitest';
import { topKWeights } from '../src/lib/math';

describe('topKWeights', () => {
  it('returns the K largest-magnitude weights', () => {
    const weights = new Float32Array([0.1, -0.9, 0.5, -0.2, 0.8]);
    const result = topKWeights(weights, 2);
    expect(result).toHaveLength(2);
    expect(result[0].index).toBe(1); // -0.9
    expect(result[1].index).toBe(4); // 0.8
  });

  it('returns all weights when k is larger than array', () => {
    const weights = new Float32Array([0.1, -0.2]);
    const result = topKWeights(weights, 10);
    expect(result).toHaveLength(2);
  });
});
