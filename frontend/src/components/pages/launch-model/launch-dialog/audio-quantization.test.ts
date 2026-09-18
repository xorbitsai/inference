import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ModelSpec } from '../types';
import { audioSpecsForEngine } from './audio-quantization';
import { syncLinkedField } from '../utils';
import { createForm } from '@/hooks/use-form';

function spec(engine: string | undefined, quantization: string): ModelSpec {
  return {
    model_engine: engine,
    quantization,
    model_format: engine === 'MLX' ? 'mlx' : 'pytorch',
    model_uri: '',
    model_size_in_billions: '',
  };
}

const pytorch = [
  'none',
  'INT8-Weight-Only',
  'INT8-Dynamic',
  'INT4-Weight-Only',
  'Float8-Weight-Only',
  'Float8-Dynamic',
].map((q) => spec('PyTorch', q));
const mlx = spec('MLX', 'none');
const specs = [...pytorch, mlx];

test('audio quantizations are scoped to the selected engine, case-insensitively', () => {
  assert.deepEqual(audioSpecsForEngine(specs, 'mlx'), [mlx]);
  assert.deepEqual(audioSpecsForEngine(specs, 'PyTorch'), pytorch);
  assert.deepEqual(audioSpecsForEngine(specs, ''), specs);
  assert.deepEqual(audioSpecsForEngine(specs, 'unknown'), []);
});

test('legacy specs remain selectable without leaking into explicit engine specs', () => {
  const legacy = spec(undefined, 'INT8');
  assert.deepEqual(audioSpecsForEngine([legacy], 'PyTorch'), [legacy]);
  assert.deepEqual(audioSpecsForEngine([legacy, mlx], 'MLX'), [mlx]);
});

test('switching to MLX resets a PyTorch-only quantization and preserves valid values', () => {
  const form = createForm();
  form.setFieldValue('quantization', 'INT4-Weight-Only');
  const options = audioSpecsForEngine(specs, 'MLX').map((item) => ({ value: item.quantization }));
  syncLinkedField(form, 'quantization', form.getFieldValue('quantization'), options, true);
  assert.equal(form.getFieldValue('quantization'), 'none');
  syncLinkedField(form, 'quantization', 'none', options, true);
  assert.equal(form.getFieldValue('quantization'), 'none');
  syncLinkedField(form, 'quantization', 'none', [], true);
  assert.equal(form.getFieldValue('quantization'), undefined);
});
