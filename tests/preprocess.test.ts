import { describe, it, expect } from 'vitest';
import { normalizeCanvasPixels } from '../src/lib/preprocess';

describe('normalizeCanvasPixels', () => {
  it('normalizes white-on-black pixels to MNIST mean/std', () => {
    const size = 28;
    const data = new Uint8ClampedArray(size * size * 4);
    data[0] = 255; data[1] = 255; data[2] = 255; data[3] = 255;

    const result = normalizeCanvasPixels(data, size);

    expect(result[0]).toBeCloseTo((1.0 - 0.1307) / 0.3081, 5);
    expect(result[1]).toBeCloseTo((0.0 - 0.1307) / 0.3081, 5);
    expect(result).toHaveLength(size * size);
  });
});
