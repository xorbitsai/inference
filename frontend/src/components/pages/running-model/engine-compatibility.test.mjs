import assert from 'node:assert/strict';
import test from 'node:test';
import { hasCompatibleEngineSpec } from './engine-compatibility.mjs';

const currentPytorchSpec = {
  modelFormat: 'pytorch',
  modelSizeInBillions: 7,
  quantization: 'none',
};

test('accepts an engine that supports the running model exact spec', () => {
  assert.equal(
    hasCompatibleEngineSpec(
      [
        {
          model_format: 'pytorch',
          model_size_in_billions: 7,
          quantizations: ['none', '4-bit'],
        },
      ],
      currentPytorchSpec
    ),
    true
  );
});

test('rejects an engine that only supports another model format', () => {
  assert.equal(
    hasCompatibleEngineSpec(
      [
        {
          model_format: 'ggufv2',
          model_size_in_billions: 7,
          quantizations: ['q4_0'],
        },
      ],
      currentPytorchSpec
    ),
    false
  );
});

test('rejects engines with a different size or quantization', () => {
  assert.equal(
    hasCompatibleEngineSpec(
      [
        {
          model_format: 'pytorch',
          model_size_in_billions: 14,
          quantization: 'none',
        },
      ],
      currentPytorchSpec
    ),
    false
  );
  assert.equal(
    hasCompatibleEngineSpec(
      [
        {
          model_format: 'pytorch',
          model_size_in_billions: 7,
          quantization: '4-bit',
        },
      ],
      currentPytorchSpec
    ),
    false
  );
});

test('normalizes model sizes and quantization casing', () => {
  assert.equal(
    hasCompatibleEngineSpec(
      [
        {
          model_format: 'pytorch',
          model_size_in_billions: '1_8',
          quantization: 'Int4',
        },
      ],
      {
        modelFormat: 'PyTorch',
        modelSizeInBillions: 1.8,
        quantization: 'int4',
      }
    ),
    true
  );
});

test('does not require size or quantization when they do not apply', () => {
  assert.equal(
    hasCompatibleEngineSpec([{ model_format: 'pytorch' }], { modelFormat: 'pytorch' }),
    true
  );
});

test('rejects unavailable engine entries', () => {
  assert.equal(
    hasCompatibleEngineSpec('Engine dependency is not installed', currentPytorchSpec),
    false
  );
});
