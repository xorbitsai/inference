import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const source = readFileSync(new URL('./file-upload.tsx', import.meta.url), 'utf8');
const tree = ts.createSourceFile(
  'file-upload.tsx',
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
);
let updateFiles;
function visit(node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === 'updateFiles') {
    updateFiles = node.initializer.getText(tree);
  }
  ts.forEachChild(node, visit);
}
visit(tree);
assert.ok(updateFiles, 'FileUpload must define updateFiles');
const { outputText } = ts.transpileModule(`globalThis.updateFiles = ${updateFiles};`, {
  compilerOptions: { target: ts.ScriptTarget.ES2020 },
});

test('invalid replacement clears the previously selected file before conversion', async () => {
  const oldFile = { name: 'old.pdf' };
  let selected = [oldFile];
  let error;
  let conversions = 0;
  const context = {
    Error,
    disabled: false,
    validateFile: () => {
      throw new Error('PDF or image required');
    },
    setValidationError: (value) => {
      error = value;
    },
    onChange: (value) => {
      selected = value;
    },
    toUploadValue: () => {
      conversions += 1;
    },
  };
  runInNewContext(outputText, context);
  await context.updateFiles([{ name: 'invalid.txt' }]);
  assert.equal(error, 'PDF or image required');
  assert.equal(selected.length, 0);
  assert.equal(conversions, 0);
});

test('valid replacement clears the validation error and updates the file', async () => {
  const file = { name: 'new.pdf' };
  const uploaded = { file };
  let selected;
  let error = 'previous error';
  const context = {
    Error,
    disabled: false,
    validateFile: () => {},
    setValidationError: (value) => {
      error = value;
    },
    onChange: (value) => {
      selected = value;
    },
    toUploadValue: async (value) => {
      assert.equal(value, file);
      return uploaded;
    },
  };
  runInNewContext(outputText, context);
  await context.updateFiles([file]);
  assert.equal(error, '');
  assert.equal(selected[0], uploaded);
});

test('empty and disabled selections do not change the current file', async () => {
  let changes = 0;
  const context = {
    disabled: false,
    onChange: () => {
      changes += 1;
    },
    validateFile: () => {
      throw new Error('must not validate');
    },
  };
  runInNewContext(outputText, context);
  await context.updateFiles([]);
  context.disabled = true;
  await context.updateFiles([{ name: 'new.pdf' }]);
  assert.equal(changes, 0);
});
