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

function fieldsFor(abilities, promptSpeech = []) {
  const mocks = {
    '@/constants': { ModelAbility },
    '@/contexts/i18n-context': {
      useI18n: () => ({
        t: (key) => (key === 'runningModels.detail.voiceInstruction' ? 'Voice Instruction' : key),
      }),
    },
    '@/hooks/use-form': {
      useWatch: (name) => (name === 'prompt_speech' ? promptSpeech : undefined),
    },
    '../emotion-vector-utils': { isIndexTTSEmotionModel: () => false },
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
    model: { model_name: 'generic-tts', model_family: 'generic', model_ability: abilities },
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
