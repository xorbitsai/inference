import assert from 'node:assert/strict';
import { File } from 'node:buffer';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import { runInThisContext } from 'node:vm';
import ts from 'typescript';

const require = createRequire(import.meta.url);
function loadSource(file, mocks = {}) {
  const source = readFileSync(new URL(file, import.meta.url), 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  });
  const compiledModule = { exports: {} };
  runInThisContext(`(function(require, module, exports) { ${outputText}\n})`)(
    (name) => mocks[name] ?? require(name),
    compiledModule,
    compiledModule.exports
  );
  return compiledModule.exports;
}

const uploadUtils = loadSource('./document-upload-utils.ts');
const locales = ['zh', 'zh-TW', 'en', 'ja', 'ko'];
const abilities = new Proxy({}, { get: (_, key) => String(key).toLowerCase() });
const config = loadSource('./capability-config.tsx', {
  'lucide-react': {},
  '@/constants': { ModelAbility: abilities },
  '@/lib/is': { isEmpty: (value) => Object.keys(value).length === 0 },
  './panels/form-panels': {},
  './panels/result-panels': { ResultPanels: {} },
  './emotion-vector-utils': { EMPTY_INDEX_TTS_EMOTION_VECTOR: [] },
  './seed-utils': {},
  './document-upload-utils': uploadUtils,
  './utils': { firstUpload: (values, key) => values[key]?.[0] },
}).CAPABILITY_CONFIGS.docanalyze;

test('PDF and backend-supported image extensions are accepted', () => {
  for (const extension of ['pdf', 'png', 'jpeg', 'jp2', 'webp', 'gif', 'bmp', 'jpg', 'PDF']) {
    assert.doesNotThrow(() =>
      uploadUtils.validateDocumentFile({
        name: `document.${extension}`,
        type: '',
        size: 1,
      })
    );
  }
});

test('other files, conflicting MIME types and empty files are rejected', () => {
  for (const file of [
    { name: 'document.docx', type: 'application/pdf', size: 1 },
    { name: 'image.svg', type: 'image/svg+xml', size: 1 },
    { name: 'document.pdf', type: 'text/plain', size: 1 },
    { name: 'image.png', type: 'application/pdf', size: 1 },
    { name: 'document.pdf', type: 'application/pdf', size: 0 },
  ])
    assert.throws(() => uploadUtils.validateDocumentFile(file));
});

test('docanalyze uses its own endpoint and sends file plus request ID', () => {
  assert.equal(config.requestApi, '/v1/images/docanalyze');
  assert.equal(config.codeExample.fields.find((field) => field.type === 'file').key, 'file');
  const body = config.transformValues({
    modelUid: 'MinerU2.5',
    requestId: 'request-1',
    values: { file: [{ file: new File(['pdf'], 'document.pdf', { type: 'application/pdf' }) }] },
  });
  assert.equal(body.get('model'), 'MinerU2.5');
  assert.equal(body.get('file').name, 'document.pdf');
  assert.equal(body.has('image'), false);
  assert.deepEqual(JSON.parse(body.get('kwargs')), { request_id: 'request-1' });
});

test('docanalyze refuses missing or invalid uploads before sending', () => {
  assert.throws(() => config.transformValues({ modelUid: 'MinerU2.5', values: { file: [] } }));
  assert.throws(() =>
    config.transformValues({
      modelUid: 'MinerU2.5',
      values: { file: [{ file: new File(['text'], 'document.txt') }] },
    })
  );
});

test('all supported locales define docanalyze labels, descriptions and upload messages', () => {
  for (const locale of locales) {
    const dictionary = loadSource(`../../../i18n/locales/${locale}.ts`).default;
    for (const key of ['docanalyze', 'docanalyzeDescription']) {
      assert.ok(dictionary.launchModel[key]?.trim(), `${locale}: ${key}`);
    }
    for (const key of [
      'uploadLabel',
      'uploadDescription',
      'engineHint',
      'submit',
      'uploadRequired',
      'invalidFile',
      'emptyFile',
    ]) {
      assert.ok(dictionary.documentParsing[key]?.trim(), `${locale}: ${key}`);
    }
    const t = (key) => key.split('.').reduce((value, part) => value[part], dictionary);
    assert.equal(t(config.labelKey), dictionary.launchModel.docanalyze);
    assert.equal(t(config.descriptionKey), dictionary.launchModel.docanalyzeDescription);
    assert.equal(t(config.submitLabelKey), dictionary.documentParsing.submit);
    assert.throws(
      () => uploadUtils.validateDocumentFile({ name: 'file.txt', type: 'text/plain', size: 1 }, t),
      { message: dictionary.documentParsing.invalidFile }
    );
    assert.throws(
      () =>
        uploadUtils.validateDocumentFile({ name: 'file.pdf', type: 'application/pdf', size: 0 }, t),
      { message: dictionary.documentParsing.emptyFile }
    );
    assert.throws(
      () => config.transformValues({ modelUid: 'MinerU2.5', values: { file: [] }, t }),
      { message: dictionary.documentParsing.uploadRequired }
    );
  }
});
