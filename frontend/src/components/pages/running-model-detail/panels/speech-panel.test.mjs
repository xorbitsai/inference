import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import { runInThisContext } from 'node:vm';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const source = readFileSync(new URL('./form-panels.tsx', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    jsx: ts.JsxEmit.ReactJSX,
    esModuleInterop: true,
  },
});
const ModelAbility = {
  Text2music: 'text2music',
  Text2audioVoiceDesign: 'text2audio_voice_design',
  Text2audioVoiceCloning: 'text2audio_voice_cloning',
  Text2audioEmotionControl: 'text2audio_emotion_control',
};

function fieldsFor(abilities, model = {}, values = {}) {
  const mocks = {
    react: { ...require('react'), useEffect: () => {} },
    '@/constants': { ModelAbility },
    '@/contexts/i18n-context': {
      useI18n: () => ({
        t: (key) => (key === 'runningModels.detail.voiceInstruction' ? 'Voice Instruction' : key),
      }),
    },
    '@/hooks/use-form': {
      useWatch: (name) => values[name],
    },
    '../emotion-vector-utils': { isIndexTTSEmotionModel: () => false },
    '../speech-options': require('../speech-options.ts'),
  };
  const compiledModule = { exports: {} };
  runInThisContext(`(function(require, module, exports) { ${outputText}\n})`)(
    (name) => {
      if (mocks[name]) return mocks[name];
      if (name.startsWith('@/components/ui/')) {
        return new Proxy({}, { get: (_, key) => String(key) });
      }
      if (name.startsWith('@/') || name.startsWith('../')) return {};
      return require(name);
    },
    compiledModule,
    compiledModule.exports
  );
  const tree = compiledModule.exports.SpeechPanel({
    form: {},
    model: {
      model_name: 'generic-tts',
      model_family: 'generic',
      model_ability: abilities,
      ...model,
    },
  });
  const fields = {};
  function visit(node) {
    if (Array.isArray(node)) return node.forEach(visit);
    if (!node?.props) return;
    if (node.type === 'FormField') fields[node.props.name] = node.props;
    visit(node.props.children);
  }
  visit(tree);
  return fields;
}

test('voice design exposes an optional instruction, including without reference audio', () => {
  for (const abilities of [
    [ModelAbility.Text2audioVoiceDesign],
    [ModelAbility.Text2audioVoiceDesign, ModelAbility.Text2audioVoiceCloning],
  ]) {
    const fields = fieldsFor(abilities);
    assert.equal(fields.instruct.label, 'Voice Instruction');
    assert.equal(
      fields.instruct.rules.some((rule) => rule.required),
      false
    );
    assert.equal(
      fields.input.rules.some((rule) => rule.required),
      true
    );
  }
});

test('music description remains required', () => {
  const fields = fieldsFor([ModelAbility.Text2music]);
  assert.equal(
    fields.instruct.rules.some((rule) => rule.required),
    true
  );
});

test('ordinary TTS does not gain an instruction field', () => {
  assert.equal(fieldsFor(['text2audio']).instruct, undefined);
});

test('ZONOS3-Mini offers only engine-compatible speech controls', (t) => {
  const previousWindow = globalThis.window;
  t.after(() => {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  });
  globalThis.window = {
    AudioContext: class {
      createMediaStreamDestination() {}
    },
    HTMLMediaElement: class {
      get srcObject() {
        return null;
      }
    },
  };

  for (const engine of ['transformers', 'SGLang-Omni', 'MLX', 'GGML']) {
    const model = {
      model_family: 'zonos3_mini',
      model_engine: engine,
      model_lang: ['en', 'zh'],
    };
    const controls = (response_format) =>
      fieldsFor(['text2audio', ModelAbility.Text2audioVoiceCloning], model, { response_format });
    const pcm = controls('pcm');
    assert.equal(Boolean(pcm.stream), engine !== 'transformers');
    assert.equal(Boolean(pcm.language), engine !== 'SGLang-Omni');
    assert.equal(pcm.prompt_text, undefined);
    assert.equal(pcm.speed, undefined);
    assert.ok(pcm.prompt_speech);
    assert.equal(
      pcm.response_format.children.props.options.some(({ value }) => value === 'pcm'),
      engine !== 'transformers'
    );
    assert.equal(controls('mp3').stream, undefined);
  }

  delete globalThis.window.AudioContext;
  const fallback = fieldsFor(
    ['text2audio'],
    { model_name: 'ZONOS3-Mini', model_engine: 'GGML' },
    { response_format: 'pcm' }
  );
  assert.equal(fallback.stream, undefined);
  assert.equal(
    fallback.response_format.children.props.options.some(({ value }) => value === 'pcm'),
    false
  );
});
