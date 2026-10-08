import assert from 'node:assert/strict';
import test from 'node:test';

import { shouldApplyPreferredDownloadSource } from './download-source-utils.mjs';

const options = [{ value: 'auto' }, { value: 'huggingface' }, { value: 'modelscope' }];

test('preserves a download hub selected before settings response resolves', async () => {
  let resolvePreferredSource;
  const preferredSource = new Promise((resolve) => {
    resolvePreferredSource = resolve;
  });
  let userChangedDownloadHub = false;

  userChangedDownloadHub = true;
  resolvePreferredSource('modelscope');

  assert.equal(
    shouldApplyPreferredDownloadSource(
      await preferredSource,
      options,
      userChangedDownloadHub
    ),
    false
  );
});

test('applies a matching preferred source before user interaction', () => {
  assert.equal(shouldApplyPreferredDownloadSource('modelscope', options, false), true);
});

test('does not apply an unavailable preferred source', () => {
  assert.equal(shouldApplyPreferredDownloadSource('csghub', options, false), false);
});
