'use client';

import { useCallback, useEffect, useState } from 'react';
import { ArrowLeftRight, Plus, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { FileUpload } from '@/components/ui/file-upload';
import { FormField } from '@/components/ui/form-field';
import { FormList } from '@/components/ui/form-list';
import { Input } from '@/components/ui/input';
import { Segmented } from '@/components/ui/segmented';
import { Select } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { useWatch } from '@/hooks/use-form';
import { useI18n } from '@/contexts/i18n-context';
import { ModelAbility } from '@/constants';
import {
  SAMPLING_METHOD_OPTIONS,
  OCR_TYPE_OPTIONS,
  OCR_MODEL_SIZE_OPTIONS,
} from '@/constants/running';
import type { FileUploadValue } from '@/types/common';
import type { BaseFormFieldProps } from '@/types/form';

import { AudioRecorderUpload } from '../components/audio-recorder-upload';
import { ImageEditorCreateMask } from '../components/image-editor-create-mask';
import {
  INDEX_TTS_EMOTION_DIMENSIONS,
  INDEX_TTS_EMOTION_MAX_TOTAL,
  isIndexTTSEmotionModel,
  parseIndexTTSEmotionVector,
} from '../emotion-vector-utils';
import {
  formatImageSeeds,
  generateRandomImageSeeds,
  MAX_IMAGE_SEED,
  parseImageSeeds,
} from '../image-seed-utils';
import { createRandomSeed, MAX_SEED, parseScalarSeed } from '../seed-utils';
import type { CapabilityFormProps } from '../types';
import { isJsonObject } from '../utils';
import { DOCUMENT_UPLOAD_ACCEPT, validateDocumentFile } from '../document-upload-utils';

const SPEECH_RESPONSE_FORMAT_OPTIONS = ['mp3', 'wav', 'flac', 'ogg'].map((value) => ({
  label: value.toUpperCase(),
  value,
}));

const FISH_SPEECH_RESPONSE_FORMAT_OPTIONS = ['mp3', 'wav', 'pcm'].map((value) => ({
  label: value.toUpperCase(),
  value,
}));

const MUSIC_RESPONSE_FORMAT_OPTIONS = ['mp3', 'wav', 'flac', 'ogg'].map((value) => ({
  label: value.toUpperCase(),
  value,
}));

const ASTRA_CAMERA_MOTIONS = [
  { key: 'moveForward', symbol: '↑', value: 1 },
  { key: 'rotateLeft', symbol: '↺', value: 2 },
  { key: 'rotateRight', symbol: '↻', value: 3 },
  { key: 'forwardLeft', symbol: '↖', value: 4 },
  { key: 'forwardRight', symbol: '↗', value: 5 },
  { key: 'sCurve', symbol: '∿', value: 6 },
  { key: 'leftThenRight', symbol: '⇆', value: 7 },
];

function normalizeNumberInput(value: unknown) {
  return value === '' ? '' : Number(value);
}

function PromptFields() {
  const { t } = useI18n();
  return (
    <>
      <FormField
        name="prompt"
        label={t('runningModels.detail.prompt')}
        rules={[{ required: true }]}
      >
        <Textarea className="min-h-24" placeholder={t('runningModels.detail.promptPlaceholder')} />
      </FormField>
      <FormField name="negative_prompt" label={t('runningModels.detail.negativePrompt')}>
        <Textarea
          className="min-h-20"
          placeholder={t('runningModels.detail.negativePromptPlaceholder')}
        />
      </FormField>
    </>
  );
}

function ScalarSeedField({ form }: Pick<CapabilityFormProps, 'form'>) {
  const { t } = useI18n();
  return (
    <div className="flex items-end gap-2">
      <FormField
        className="flex-1"
        name="seed"
        label={t('runningModels.detail.seed')}
        placeholder={t('runningModels.detail.randomSeedPlaceholder')}
        normalize={normalizeNumberInput}
        rules={[
          {
            validator: (value) => {
              try {
                parseScalarSeed(value);
                return true;
              } catch {
                return false;
              }
            },
            message: t('runningModels.detail.seedValidation', { max: MAX_SEED }),
          },
        ]}
      >
        <Input type="number" min={-1} max={MAX_SEED} step={1} />
      </FormField>
      <Button
        variant="outline"
        size="icon"
        aria-label={t('runningModels.detail.generateRandomSeed')}
        title={t('runningModels.detail.generateRandomSeed')}
        onClick={() => form.setFieldValue('seed', createRandomSeed())}
      >
        <span aria-hidden="true">🎲</span>
      </Button>
    </div>
  );
}

function ImageGenerationFields({
  form,
  includeImageParams = false,
}: Pick<CapabilityFormProps, 'form'> & { includeImageParams?: boolean }) {
  const { t } = useI18n();
  const imageCount = Math.max(1, Math.round(Number(useWatch('n', form)) || 1));
  const samplingMethodOptions = SAMPLING_METHOD_OPTIONS.map((option) => ({
    ...option,
    label: option.value === 'default' ? t('runningModels.detail.defaultOption') : option.label,
  }));

  const generateRandomSeeds = () => {
    form.setFieldValue('seed', formatImageSeeds(generateRandomImageSeeds(imageCount)));
  };

  const swapDimensions = () => {
    const width = form.getFieldValue('width');
    const height = form.getFieldValue('height');
    form.setFieldsValue({ width: height, height: width });
  };

  return (
    <>
      <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-end gap-2">
        <FormField
          name="width"
          label={t('runningModels.detail.width')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        <Button
          aria-label={t('runningModels.detail.swapDimensions')}
          title={t('runningModels.detail.swapDimensions')}
          variant="outline"
          size="icon"
          onClick={swapDimensions}
        >
          <ArrowLeftRight />
        </Button>
        <FormField
          name="height"
          label={t('runningModels.detail.height')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <FormField
          name="n"
          label={t('runningModels.detail.imageCount')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" min={1} max={10} />
        </FormField>
        <FormField
          name="guidance_scale"
          label={t('runningModels.detail.guidanceScale')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" step={0.1} />
        </FormField>
        <FormField
          name="num_inference_steps"
          label={t('runningModels.detail.inferenceSteps')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        {includeImageParams && (
          <>
            <FormField
              name="padding_image_to_multiple"
              label={t('runningModels.detail.paddingMultiple')}
              normalize={normalizeNumberInput}
            >
              <Input type="number" />
            </FormField>
            <FormField
              name="strength"
              label={t('runningModels.detail.strength')}
              normalize={normalizeNumberInput}
            >
              <Input type="number" min={0} max={1} step={0.1} />
            </FormField>
          </>
        )}
      </div>
      <div className="flex items-end gap-2">
        <FormField
          className="flex-1"
          name="seed"
          label={t('runningModels.detail.seeds')}
          placeholder={t('runningModels.detail.imageSeedsPlaceholder')}
          rules={[
            {
              validator: (value) => {
                try {
                  parseImageSeeds(value, imageCount);
                  return true;
                } catch {
                  return false;
                }
              },
              message: t('runningModels.detail.imageSeedsValidation', {
                count: imageCount,
                max: MAX_IMAGE_SEED,
              }),
            },
          ]}
        >
          <Input />
        </FormField>
        <Button
          aria-label={t('runningModels.detail.generateRandomImageSeeds')}
          title={t('runningModels.detail.generateRandomImageSeeds')}
          variant="outline"
          size="icon"
          onClick={generateRandomSeeds}
        >
          <span aria-hidden="true">🎲</span>
        </Button>
      </div>
      <FormField name="sampler_name" label={t('runningModels.detail.samplingMethod')}>
        <Select options={samplingMethodOptions} allowClear={false} showSearch />
      </FormField>
    </>
  );
}

function VideoFields({ form }: Pick<CapabilityFormProps, 'form'>) {
  const { t } = useI18n();
  const swapDimensions = () => {
    const width = form.getFieldValue('width');
    const height = form.getFieldValue('height');
    form.setFieldsValue({ width: height, height: width });
  };

  return (
    <>
      <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-end gap-2">
        <FormField
          name="width"
          label={t('runningModels.detail.width')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        <Button
          aria-label={t('runningModels.detail.swapDimensions')}
          title={t('runningModels.detail.swapDimensions')}
          variant="outline"
          size="icon"
          onClick={swapDimensions}
        >
          <ArrowLeftRight />
        </Button>
        <FormField
          name="height"
          label={t('runningModels.detail.height')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <FormField
          name="num_frames"
          label={t('runningModels.detail.frames')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        <FormField
          name="fps"
          label={t('runningModels.detail.fps')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        <FormField
          name="num_inference_steps"
          label={t('runningModels.detail.inferenceSteps')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        <FormField
          name="guidance_scale"
          label={t('runningModels.detail.guidanceScale')}
          normalize={normalizeNumberInput}
        >
          <Input type="number" min={1} max={20} step={0.1} />
        </FormField>
        <ScalarSeedField form={form} />
      </div>
    </>
  );
}

export function TextPromptPanel({ actions }: Pick<CapabilityFormProps, 'actions'>) {
  const { t } = useI18n();
  return (
    <>
      <FormField
        name="prompt"
        label={t('runningModels.detail.prompt')}
        rules={[{ required: true }]}
      >
        <Textarea className="min-h-40" placeholder={t('runningModels.detail.enterPrompt')} />
      </FormField>
      {actions}
      <FormField
        name="max_tokens"
        label={t('runningModels.detail.maxTokens')}
        normalize={normalizeNumberInput}
      >
        <Input type="number" min={0} />
      </FormField>
      <FormField
        name="temperature"
        label={t('runningModels.detail.temperature')}
        normalize={normalizeNumberInput}
      >
        <Input type="number" min={0} max={2} step={0.01} />
      </FormField>
    </>
  );
}

export function EmbedPanel({ form, model }: CapabilityFormProps) {
  const { t } = useI18n();
  const abilities = model.model_ability || [];
  const watchedAbility = useWatch('model_ability', form);
  const selectedAbility = abilities.includes(watchedAbility as ModelAbility)
    ? (watchedAbility as ModelAbility)
    : abilities[0] || ModelAbility.Embed;
  const showAbilitySelector = !(abilities.length === 1 && abilities[0] === ModelAbility.Embed);

  useEffect(() => {
    if (watchedAbility !== selectedAbility) {
      form.setFieldValue('model_ability', selectedAbility);
    }
  }, [form, selectedAbility, watchedAbility]);

  return (
    <>
      {showAbilitySelector && (
        <FormField name="model_ability" label={t('runningModels.modelAbility')}>
          <Segmented
            options={abilities.map((ability) => ({
              label: t(`launchModel.${ability}`),
              value: ability,
            }))}
          />
        </FormField>
      )}

      {selectedAbility === ModelAbility.Embed && (
        <FormField
          name="input"
          label={t('runningModels.detail.textInput')}
          rules={[{ required: true }]}
          placeholder={t('runningModels.detail.embeddingTextPlaceholder')}
        >
          <Textarea className="min-h-24" />
        </FormField>
      )}

      {selectedAbility === ModelAbility.EmbedVision && (
        <FormField name="image" rules={[{ required: true }]}>
          <FileUpload
            accept="image/*"
            label={t('runningModels.detail.uploadImage')}
            description={t('runningModels.detail.embeddingImageDescription')}
          />
        </FormField>
      )}

      {selectedAbility === ModelAbility.EmbedVideo && (
        <FormField
          name="video"
          label={t('runningModels.detail.videoInput')}
          rules={[{ required: true }]}
          placeholder={t('runningModels.detail.videoInputPlaceholder')}
        >
          <Textarea className="min-h-24" />
        </FormField>
      )}

      {selectedAbility === ModelAbility.EmbedAudio && (
        <FormField
          name="audio"
          label={t('runningModels.detail.audioInput')}
          rules={[{ required: true }]}
          placeholder={t('runningModels.detail.audioInputPlaceholder')}
        >
          <Textarea className="min-h-24" />
        </FormField>
      )}
    </>
  );
}

export function RerankPanel({ form, model }: CapabilityFormProps) {
  const { t } = useI18n();
  const abilities = model.model_ability || [];
  const watchedAbility = useWatch('model_ability', form);
  const selectedAbility = abilities.includes(watchedAbility as ModelAbility)
    ? (watchedAbility as ModelAbility)
    : abilities[0] || ModelAbility.Rerank;
  const showAbilitySelector = !(abilities.length === 1 && abilities[0] === ModelAbility.Rerank);
  const documentPlaceholder = (index: number) => {
    if (selectedAbility === ModelAbility.RerankVideo) {
      return t('runningModels.detail.rerankVideoPlaceholder');
    }
    if (selectedAbility === ModelAbility.RerankVision) {
      return t('runningModels.detail.rerankImagePlaceholder');
    }
    if (selectedAbility === ModelAbility.RerankAudio) {
      return t('runningModels.detail.rerankAudioPlaceholder');
    }
    return t('runningModels.detail.candidateParagraphPlaceholder', { index: index + 1 });
  };

  useEffect(() => {
    if (watchedAbility !== selectedAbility) {
      form.setFieldValue('model_ability', selectedAbility);
    }
  }, [form, selectedAbility, watchedAbility]);

  return (
    <>
      {showAbilitySelector && (
        <FormField name="model_ability" label={t('runningModels.modelAbility')}>
          <Segmented
            block
            options={abilities.map((ability) => ({
              label: t(`launchModel.${ability}`),
              value: ability,
            }))}
          />
        </FormField>
      )}
      <FormField
        name="query"
        label={t('runningModels.detail.query')}
        placeholder={t('runningModels.detail.queryPlaceholder')}
        rules={[{ required: true }]}
      >
        <Textarea className="min-h-24" />
      </FormField>
      <FormList
        name="documents"
        label={t('runningModels.detail.candidateContent')}
        layout="horizontal"
        renderAction={({ add }) => (
          <Button size="sm" type="button" variant="outline" onClick={() => add('')}>
            <Plus />
            {t('runningModels.detail.add')}
          </Button>
        )}
      >
        {({ fields, remove }) => (
          <div className="space-y-3">
            {fields.map((field, index) => (
              <div className="flex gap-2" key={field.name}>
                <FormField
                  className="flex-1"
                  name={['documents', field.name]}
                  rules={[{ required: true }]}
                  placeholder={documentPlaceholder(index)}
                >
                  <Input />
                </FormField>
                {fields.length > 1 && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="shrink-0 rounded-full text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                    aria-label={t('runningModels.detail.removeCandidate', {
                      index: index + 1,
                    })}
                    disabled={fields.length <= 1}
                    onClick={() => remove(field.name)}
                  >
                    <Trash2 />
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </FormList>
    </>
  );
}

export function OcrPanel() {
  const { t } = useI18n();
  const ocrTypeOptions = OCR_TYPE_OPTIONS.map((option) => ({
    ...option,
    label: t(`runningModels.detail.ocrType.${option.value}`),
  }));
  const ocrModelSizeOptions = OCR_MODEL_SIZE_OPTIONS.map((option) => ({
    ...option,
    label: t(`runningModels.detail.ocrModelSize.${option.value}`),
  }));
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload
          accept="image/*,application/pdf"
          label={t('runningModels.detail.uploadImageOrPdf')}
          description={t('runningModels.detail.ocrUploadDescription')}
        />
      </FormField>
      <div className="grid grid-cols-2 gap-3">
        <FormField
          name="ocr_type"
          label={t('runningModels.detail.outputFormat')}
          tooltip={t('runningModels.detail.ocrOutputFormatTooltip')}
        >
          <Select options={ocrTypeOptions} allowClear={false} />
        </FormField>
        <FormField
          name="model_size"
          label={t('runningModels.modelSize')}
          tooltip={t('runningModels.detail.modelSizeTooltip')}
        >
          <Select options={ocrModelSizeOptions} allowClear={false} />
        </FormField>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <FormField
          name="test_compress"
          label={t('runningModels.detail.testCompress')}
          valuePropName="checked"
          layout="horizontal"
          tooltip={t('runningModels.detail.testCompressTooltip')}
        >
          <Switch />
        </FormField>
        <FormField
          name="save_results"
          label={t('runningModels.detail.saveResults')}
          valuePropName="checked"
          layout="horizontal"
          tooltip={t('runningModels.detail.saveResultsTooltip')}
        >
          <Switch />
        </FormField>
      </div>
    </>
  );
}

export function TextToImagePanel({ form }: CapabilityFormProps) {
  return (
    <>
      <PromptFields />
      <ImageGenerationFields form={form} />
    </>
  );
}

export function ImageToImagePanel({ form }: CapabilityFormProps) {
  const { t } = useI18n();
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload
          accept="image/*"
          label={t('runningModels.detail.uploadReferenceImage')}
          description={t('runningModels.detail.uploadReferenceImageDescription')}
        />
      </FormField>
      <PromptFields />
      <ImageGenerationFields form={form} includeImageParams />
    </>
  );
}

export function InpaintingPanel({ form }: CapabilityFormProps) {
  const updateMask = useCallback(
    (value: FileUploadValue[]) => {
      form.setFieldValue('mask_image', value);
    },
    [form]
  );

  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <ImageEditorCreateMask updateMask={updateMask} />
      </FormField>
      <FormField name="mask_image" hidden />
      <PromptFields />
      <ImageGenerationFields form={form} includeImageParams />
    </>
  );
}

export function TextToVideoPanel({ form }: CapabilityFormProps) {
  return (
    <>
      <PromptFields />
      <VideoFields form={form} />
    </>
  );
}

export function ImageToVideoPanel({ form }: CapabilityFormProps) {
  const { t } = useI18n();
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload accept="image/*" label={t('runningModels.detail.uploadFirstFrame')} />
      </FormField>
      <PromptFields />
      <VideoFields form={form} />
    </>
  );
}

export function FirstLastFrameVideoPanel({ form }: CapabilityFormProps) {
  const { t } = useI18n();
  return (
    <>
      <div className="grid gap-3 md:grid-cols-2">
        <FormField name="first_frame" rules={[{ required: true }]}>
          <FileUpload accept="image/*" label={t('runningModels.detail.firstFrame')} />
        </FormField>
        <FormField name="last_frame" rules={[{ required: true }]}>
          <FileUpload accept="image/*" label={t('runningModels.detail.lastFrame')} />
        </FormField>
      </div>
      <PromptFields />
      <VideoFields form={form} />
    </>
  );
}

const jsonObjectRule = (message: string) => ({
  validator: (value: unknown) => isJsonObject(value),
  message,
});

function isAstraWorldModel(model: CapabilityFormProps['model']) {
  return model.model_family === 'Astra' || model.model_name === 'Astra';
}

function WorldGenerationFields({ model }: Pick<CapabilityFormProps, 'model'>) {
  const { t } = useI18n();
  const cameraMotionOptions = ASTRA_CAMERA_MOTIONS.map(({ key, symbol, value }) => ({
    label: `${symbol} ${t(`runningModels.detail.cameraMotion.${key}`)}`,
    value,
  }));
  return (
    <>
      <FormField
        name="prompt"
        label={t('runningModels.detail.prompt')}
        rules={[{ required: true }]}
      >
        <Textarea
          className="min-h-24"
          placeholder={t('runningModels.detail.worldPromptPlaceholder')}
        />
      </FormField>
      {isAstraWorldModel(model) && (
        <FormField
          name="astra_camera_motion"
          label={t('runningModels.detail.cameraMotionLabel')}
          extra={t('runningModels.detail.cameraMotionDescription')}
        >
          <Select options={cameraMotionOptions} allowClear={false} />
        </FormField>
      )}
      <FormField
        name="generation_config"
        label={t('runningModels.detail.generationConfig')}
        extra={t('runningModels.detail.generationConfigDescription')}
        rules={[jsonObjectRule(t('runningModels.detail.jsonObjectValidation'))]}
      >
        <Textarea className="min-h-28 font-mono text-xs" spellCheck={false} />
      </FormField>
      <FormField
        name="model_kwargs"
        label={t('runningModels.detail.advancedModelKwargs')}
        extra={t('runningModels.detail.advancedModelKwargsDescription')}
        rules={[jsonObjectRule(t('runningModels.detail.jsonObjectValidation'))]}
      >
        <Textarea className="min-h-28 font-mono text-xs" spellCheck={false} />
      </FormField>
    </>
  );
}

export function TextToWorldPanel({ model }: CapabilityFormProps) {
  return <WorldGenerationFields model={model} />;
}

export function ImageToWorldPanel({ model }: CapabilityFormProps) {
  const { t } = useI18n();
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload
          accept="image/*"
          label={t('runningModels.detail.uploadWorldImage')}
          description={t('runningModels.detail.uploadWorldImageDescription')}
        />
      </FormField>
      <WorldGenerationFields model={model} />
    </>
  );
}

export function VideoToWorldPanel({ model }: CapabilityFormProps) {
  const { t } = useI18n();
  return (
    <>
      <FormField name="video" rules={[{ required: true }]}>
        <FileUpload
          accept="video/*"
          label={t('runningModels.detail.uploadWorldVideo')}
          description={t('runningModels.detail.uploadWorldVideoDescription')}
        />
      </FormField>
      <WorldGenerationFields model={model} />
    </>
  );
}

export function AudioToTextPanel({ form }: CapabilityFormProps) {
  const { t } = useI18n();
  return (
    <>
      <FormField name="file" rules={[{ required: true }]}>
        <AudioRecorderUpload form={form} />
      </FormField>
      <FormField
        name="language"
        label={t('runningModels.detail.language')}
        placeholder={t('runningModels.detail.languagePlaceholder')}
      >
        <Input />
      </FormField>
      <FormField
        name="prompt"
        label={t('runningModels.detail.prompt')}
        placeholder={t('runningModels.detail.audioPromptPlaceholder')}
      >
        <Textarea />
      </FormField>
      <FormField
        name="temperature"
        label={t('runningModels.detail.temperature')}
        normalize={normalizeNumberInput}
      >
        <Input type="number" min={0} max={1} step={0.1} />
      </FormField>
    </>
  );
}

export function SpeakerEmbeddingPanel() {
  const { t } = useI18n();
  return (
    <FormField name="file" rules={[{ required: true }]}>
      <FileUpload
        accept="audio/*"
        label={t('runningModels.detail.uploadSpeechSample')}
        description={t('runningModels.detail.speechSampleDescription')}
      />
    </FormField>
  );
}

function EmotionVectorInput({ value, onChange, disabled, error }: BaseFormFieldProps<number[]>) {
  const { t } = useI18n();
  const vector = INDEX_TTS_EMOTION_DIMENSIONS.map((_, index) => {
    const item = value?.[index];
    return typeof item === 'number' && Number.isFinite(item) ? item : 0;
  });
  const [localValues, setLocalValues] = useState<string[]>(() =>
    vector.map((item) => String(item))
  );

  useEffect(() => {
    setLocalValues((previousValues) =>
      INDEX_TTS_EMOTION_DIMENSIONS.map((_, index) => {
        const item = value?.[index];
        const nextValue = typeof item === 'number' && Number.isFinite(item) ? item : 0;
        const previousValue = previousValues[index];

        return previousValue !== undefined && Number(previousValue) === nextValue
          ? previousValue
          : String(nextValue);
      })
    );
  }, [value]);

  const total = vector.reduce((sum, item) => sum + item, 0);
  const totalExceeded = parseIndexTTSEmotionVector(vector) === undefined;

  const updateDimension = (index: number, nextValue: number | string) => {
    setLocalValues((previousValues) => {
      const nextValues = [...previousValues];
      nextValues[index] = String(nextValue);
      return nextValues;
    });

    const parsedValue = Number(nextValue);
    if (!Number.isFinite(parsedValue)) return;

    const nextVector = [...vector];
    nextVector[index] =
      Math.round(Math.min(INDEX_TTS_EMOTION_MAX_TOTAL, Math.max(0, parsedValue)) * 100) / 100;
    onChange?.(nextVector);
  };

  const normalizeDimension = (index: number) => {
    setLocalValues((previousValues) => {
      const nextValues = [...previousValues];
      nextValues[index] = String(vector[index]);
      return nextValues;
    });
  };

  return (
    <div
      className={`space-y-3 rounded-lg border p-3 ${
        error ? 'border-destructive' : 'border-border'
      } ${disabled ? 'opacity-60' : ''}`}
    >
      {INDEX_TTS_EMOTION_DIMENSIONS.map(({ key }, index) => {
        const label = t(`runningModels.detail.emotion.${key}`);
        return (
          <div key={key} className="grid grid-cols-[88px_minmax(0,1fr)_64px] items-center gap-3">
            <span className="truncate text-xs font-medium text-muted-foreground">{label}</span>
            <Slider
              aria-label={`${label} emotion weight`}
              disabled={disabled}
              min={0}
              max={INDEX_TTS_EMOTION_MAX_TOTAL}
              step={0.01}
              value={[vector[index]]}
              onValueChange={([nextValue]) => updateDimension(index, nextValue)}
            />
            <Input
              aria-label={`${label} emotion weight value`}
              className="h-8 px-2 text-right font-mono text-xs"
              disabled={disabled}
              error={error}
              type="number"
              min={0}
              max={INDEX_TTS_EMOTION_MAX_TOTAL}
              step={0.01}
              value={localValues[index] ?? ''}
              onChange={(event) => updateDimension(index, event.target.value)}
              onBlur={() => normalizeDimension(index)}
            />
          </div>
        );
      })}
      <div
        className={`flex justify-end text-xs font-medium ${
          totalExceeded ? 'text-destructive' : 'text-muted-foreground'
        }`}
      >
        {t('runningModels.detail.total')}: {total.toFixed(2)} /{' '}
        {INDEX_TTS_EMOTION_MAX_TOTAL.toFixed(2)}
      </div>
    </div>
  );
}

export function SpeechPanel({ form, model }: CapabilityFormProps) {
  const { t } = useI18n();
  const isMusicGeneration = model.model_ability.includes(ModelAbility.Text2music);
  const speechResponseFormatOptions =
    model.model_family === 'FishAudio'
      ? FISH_SPEECH_RESPONSE_FORMAT_OPTIONS
      : SPEECH_RESPONSE_FORMAT_OPTIONS;
  const supportsVoiceCloning = model.model_ability.includes(ModelAbility.Text2audioVoiceCloning);
  const supportsVoiceDesign = model.model_ability.includes(ModelAbility.Text2audioVoiceDesign);
  const supportedLanguages = model.model_lang ?? [];
  const languageOptions = supportedLanguages.map((value) => ({ label: value, value }));
  const promptSpeech = useWatch('prompt_speech', form);
  const hasPromptSpeech =
    supportsVoiceCloning && Array.isArray(promptSpeech) && promptSpeech.length > 0;
  const showPromptSpeech = supportsVoiceCloning;
  const showPromptText = supportsVoiceCloning && (!supportsVoiceDesign || hasPromptSpeech);
  const showVoiceInstruction = supportsVoiceDesign && !hasPromptSpeech;
  const showInstruct = showVoiceInstruction || isMusicGeneration;
  const instructLabel = isMusicGeneration
    ? t('runningModels.detail.musicDescription')
    : t('runningModels.detail.voiceInstruction');
  const instructPlaceholder = isMusicGeneration
    ? t('runningModels.detail.musicDescriptionPlaceholder')
    : t('runningModels.detail.voiceInstructionPlaceholder');
  const supportsEmotionVector =
    isIndexTTSEmotionModel(model.model_family, model.model_name) &&
    model.model_ability.includes(ModelAbility.Text2audioEmotionControl);
  const emotionVectorEnabled = Boolean(useWatch('use_emo_vector', form));

  return (
    <>
      <FormField
        name="input"
        label={
          isMusicGeneration ? t('runningModels.detail.lyrics') : t('runningModels.detail.textInput')
        }
        rules={[{ required: true }]}
      >
        <Textarea
          className="min-h-32"
          placeholder={t('runningModels.detail.speechTextPlaceholder')}
        />
      </FormField>
      {!isMusicGeneration && (
        <div className="grid grid-cols-2 gap-3">
          <FormField
            name="voice"
            label={t('runningModels.detail.voice')}
            placeholder={t('runningModels.detail.optionalVoiceId')}
          >
            <Input />
          </FormField>
          <FormField
            name="speed"
            label={t('runningModels.detail.speed')}
            normalize={normalizeNumberInput}
          >
            <Input type="number" min={0.5} max={2} step={0.1} />
          </FormField>
          <FormField
            name="stream"
            label={t('runningModels.detail.streaming')}
            valuePropName="checked"
            layout="horizontal"
            className="flex h-full items-center"
            tooltip={t('runningModels.detail.streamingTooltip')}
          >
            <Switch />
          </FormField>
          {supportedLanguages.length > 0 && (
            <FormField name="language" label={t('runningModels.detail.language')}>
              <Select options={languageOptions} placeholder={t('runningModels.detail.optional')} />
            </FormField>
          )}
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <ScalarSeedField form={form} />
        {isMusicGeneration ? (
          <FormField
            name="duration"
            label={t('runningModels.detail.durationSeconds')}
            normalize={normalizeNumberInput}
          >
            <Input type="number" />
          </FormField>
        ) : (
          <FormField name="response_format" label={t('runningModels.detail.outputFormat')}>
            <Select options={speechResponseFormatOptions} allowClear={false} />
          </FormField>
        )}
      </div>
      {isMusicGeneration && (
        <FormField name="response_format" label={t('runningModels.detail.outputFormat')}>
          <Select options={MUSIC_RESPONSE_FORMAT_OPTIONS} allowClear={false} />
        </FormField>
      )}
      {showPromptSpeech && (
        <FormField name="prompt_speech">
          <FileUpload
            accept="audio/*"
            label={t('runningModels.detail.promptSpeech')}
            description={t('runningModels.detail.promptSpeechDescription')}
          />
        </FormField>
      )}
      {showPromptText && (
        <FormField name="prompt_text" label={t('runningModels.detail.promptText')}>
          <Textarea placeholder={t('runningModels.detail.promptTextPlaceholder')} />
        </FormField>
      )}
      {showInstruct && (
        <FormField name="instruct" label={instructLabel} rules={[{ required: isMusicGeneration }]}>
          <Textarea placeholder={instructPlaceholder} />
        </FormField>
      )}
      {supportsEmotionVector && (
        <div className="space-y-3 rounded-lg bg-muted/30 p-3">
          <FormField
            name="use_emo_vector"
            label={t('runningModels.detail.emotionControl')}
            valuePropName="checked"
            layout="horizontal"
          >
            <Switch />
          </FormField>
          <FormField
            name="emo_vector"
            disabled={!emotionVectorEnabled}
            rules={[
              {
                validator: (value) =>
                  !emotionVectorEnabled || parseIndexTTSEmotionVector(value) !== undefined,
                message: t('runningModels.detail.emotionValidation'),
              },
            ]}
          >
            <EmotionVectorInput />
          </FormField>
        </div>
      )}
    </>
  );
}

export function DocumentParsingPanel() {
  const { t } = useI18n();
  return (
    <>
      <FormField name="file" rules={[{ required: true }]}>
        <FileUpload
          accept={DOCUMENT_UPLOAD_ACCEPT}
          validateFile={(file) => validateDocumentFile(file, t)}
          label={t('documentParsing.uploadLabel')}
          description={t('documentParsing.uploadDescription')}
        />
      </FormField>
      <p className="text-xs text-muted-foreground">{t('documentParsing.engineHint')}</p>
    </>
  );
}
