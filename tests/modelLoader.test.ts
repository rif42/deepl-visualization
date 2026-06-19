import { describe, it, expect, beforeAll, vi } from 'vitest';
import { loadModel } from '../src/lib/modelLoader';

describe('loadModel', () => {
  beforeAll(() => {
    global.fetch = vi.fn(async (url: string) => {
      if (url.endsWith('/model/manifest.json')) {
        return new Response(
          JSON.stringify({
            input: { weight: { file: 'input_weight.bin', shape: [2, 4] }, bias: { file: 'input_bias.bin', shape: [2] } },
            mid: { weight: { file: 'mid_weight.bin', shape: [2, 2] }, bias: { file: 'mid_bias.bin', shape: [2] } },
            output: { weight: { file: 'output_weight.bin', shape: [2, 2] }, bias: { file: 'output_bias.bin', shape: [2] } },
          }),
          { status: 200 }
        );
      }
      const sizeMap: Record<string, number> = {
        'input_weight.bin': 2 * 4,
        'input_bias.bin': 2,
        'mid_weight.bin': 2 * 2,
        'mid_bias.bin': 2,
        'output_weight.bin': 2 * 2,
        'output_bias.bin': 2,
      };
      const file = url.split('/').pop() ?? '';
      const count = sizeMap[file] ?? 1;
      return new Response(new Float32Array(count).buffer, { status: 200 });
    }) as unknown as typeof fetch;
  });

  it('builds a model from a manifest', async () => {
    const model = await loadModel('/model');
    expect(model.layers).toHaveLength(4);
  });
});
