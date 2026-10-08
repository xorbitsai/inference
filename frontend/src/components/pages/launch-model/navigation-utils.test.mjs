import assert from 'node:assert/strict';
import test from 'node:test';
import {
  clearPendingLaunchModelTarget,
  createLatestRequestGuard,
  findModelRegistration,
  getLaunchModelHref,
  getSuccessfulRegistrationGroups,
  peekPendingLaunchModelTarget,
  prioritizeModelByName,
  setPendingLaunchModelTarget,
} from './navigation-utils.mjs';

test('allows only the latest overlapping request to apply', async () => {
  const guard = createLatestRequestGuard();
  const applied = [];
  let finishFirst = () => {};
  const firstCompletion = new Promise((resolve) => {
    finishFirst = resolve;
  });
  const applyAfter = async (name, completion) => {
    const requestId = guard.start();
    await completion;
    if (guard.isLatest(requestId)) applied.push(name);
  };

  const first = applyAfter('first', firstCompletion);
  const second = applyAfter('second', Promise.resolve());
  await second;
  finishFirst();
  await first;

  assert.deepEqual(applied, ['second']);
});

test('keeps registration groups from successful requests', async () => {
  const successfulGroup = { modelType: 'LLM', registrations: [] };
  const results = await Promise.allSettled([
    Promise.resolve(successfulGroup),
    Promise.reject(new Error('unsupported model type')),
  ]);

  assert.deepEqual(getSuccessfulRegistrationGroups(results), [successfulGroup]);
});

test('finds a model type from existing registration lists', () => {
  assert.deepEqual(
    findModelRegistration(
      [
        { modelType: 'LLM', registrations: [{ model_name: 'qwen', is_builtin: true }] },
        {
          modelType: 'embedding',
          registrations: [{ model_name: 'bge-m3', is_builtin: false }],
        },
      ],
      'bge-m3'
    ),
    { modelType: 'embedding', isBuiltin: false }
  );
  assert.equal(findModelRegistration([], 'missing'), null);
});

test('skips malformed registration lists and entries', () => {
  assert.deepEqual(
    findModelRegistration(
      [
        { modelType: 'LLM', registrations: null },
        { modelType: 'embedding', registrations: { model_name: 'bge-m3' } },
        {
          modelType: 'rerank',
          registrations: [null, { model_name: 'bge-reranker', is_builtin: true }],
        },
      ],
      'bge-reranker'
    ),
    { modelType: 'rerank', isBuiltin: true }
  );
});

test('builds a launch deep link for a built-in model type', () => {
  assert.equal(getLaunchModelHref('LLM', 'qwen 2.5', true), '/launch-model/llm');
});

test('routes flexible models through the custom model tab', () => {
  assert.equal(getLaunchModelHref('flexible', 'custom/model', false), '/launch-model/custom');
});

test('does not build a deep link without a supported model type', () => {
  assert.equal(getLaunchModelHref(undefined, 'qwen', true), null);
  assert.equal(getLaunchModelHref('unknown', 'qwen', true), null);
});

test('routes a custom model through its inner model type tab', () => {
  assert.equal(getLaunchModelHref('LLM', 'custom-llm', false), '/launch-model/custom');
});

test('clears the one-time launch target after the destination reads it', () => {
  setPendingLaunchModelTarget('embedding', 'bge-m3');
  const target = peekPendingLaunchModelTarget();

  assert.deepEqual(target, { modelType: 'embedding', modelName: 'bge-m3' });
  clearPendingLaunchModelTarget(target);
  assert.equal(peekPendingLaunchModelTarget(), null);
});

test('moves the target model first without changing remaining order', () => {
  const models = [{ model_name: 'featured' }, { model_name: 'target' }, { model_name: 'other' }];

  assert.deepEqual(prioritizeModelByName(models, 'target'), [models[1], models[0], models[2]]);
  assert.deepEqual(
    models.map((model) => model.model_name),
    ['featured', 'target', 'other']
  );
});
