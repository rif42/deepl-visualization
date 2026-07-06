import type * as tfTypes from '@tensorflow/tfjs';

function getGlobalTf(): typeof tfTypes {
  if (typeof window === 'undefined') {
    throw new Error('TensorFlow.js must be loaded in the browser before this module runs.');
  }
  const globalTf = (window as any).tf as typeof tfTypes | undefined;
  if (!globalTf) {
    throw new Error('TensorFlow.js not found on window. Ensure the CDN script loaded before this module.');
  }
  return globalTf;
}

export const tf = getGlobalTf();
