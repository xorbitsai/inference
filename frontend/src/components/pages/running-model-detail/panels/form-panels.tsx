'use client';

import { useCallback, useEffect, useState } from 'react';
import { ArrowLeftRight, Plus, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { FileUpload } from '@/components/ui/file-upload';
import { FormField } from '@/components/ui/form-field';
import { FormList } from '@/components/ui/form-list';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { useWatch } from '@/hooks/use-form';
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

const DOCUMENT_BACKEND_OPTIONS = ['pipeline', 'vlm-auto-engine', 'hybrid-auto-engine'].map(
  (value) => ({ label: value, value })
);

const DOCUMENT_PARSE_METHOD_OPTIONS = ['auto', 'txt', 'ocr'].map((value) => ({
  label: value,
  value,
}));

const DOCUMENT_LANGUAGE_OPTIONS = [
  { label: 'Chinese', value: 'ch' },
  { label: 'English', value: 'en' },
  { label: 'Traditional Chinese', value: 'chinese_cht' },
];

const DOCUMENT_OUTPUT_OPTIONS = ['markdown', 'json'].map((value) => ({ label: value, value }));

function normalizeNumberInput(value: unknown) {
  return value === '' ? '' : Number(value);
}

function PromptFields() {
  return (
    <>
      <FormField name="prompt" label="Prompt" rules={[{ required: true }]}>
        <Textarea className="min-h-24" placeholder="Describe what you want..." />
      </FormField>
      <FormField name="negative_prompt" label="Negative Prompt">
        <Textarea className="min-h-20" placeholder="Things to avoid..." />
      </FormField>
    </>
  );
}

function ScalarSeedField({ form }: Pick<CapabilityFormProps, 'form'>) {
  return (
    <div className="flex items-end gap-2">
      <FormField
        className="flex-1"
        name="seed"
        label="Seed"
        placeholder="-1 = random"
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
            message: `Seed must be -1 or an integer from 0 to ${MAX_SEED}.`,
          },
        ]}
      >
        <Input type="number" min={-1} max={MAX_SEED} step={1} />
      </FormField>
      <Button
        variant="outline"
        size="icon"
        aria-label="Generate random seed"
        title="Generate random seed"
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
  const imageCount = Math.max(1, Math.round(Number(useWatch('n', form)) || 1));

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
        <FormField name="width" label="Width" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
        <Button
          aria-label="Swap width and height"
          title="Swap width and height"
          variant="outline"
          size="icon"
          onClick={swapDimensions}
        >
          <ArrowLeftRight />
        </Button>
        <FormField name="height" label="Height" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <FormField name="n" label="Number of Images" normalize={normalizeNumberInput}>
          <Input type="number" min={1} max={10} />
        </FormField>
        <FormField name="guidance_scale" label="Guidance Scale" normalize={normalizeNumberInput}>
          <Input type="number" step={0.1} />
        </FormField>
        <FormField
          name="num_inference_steps"
          label="Inference Step Number"
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        {includeImageParams && (
          <>
            <FormField
              name="padding_image_to_multiple"
              label="Padding Multiple"
              normalize={normalizeNumberInput}
            >
              <Input type="number" />
            </FormField>
            <FormField name="strength" label="Strength" normalize={normalizeNumberInput}>
              <Input type="number" min={0} max={1} step={0.1} />
            </FormField>
          </>
        )}
      </div>
      <div className="flex items-end gap-2">
        <FormField
          className="flex-1"
          name="seed"
          label="Seed(s)"
          placeholder="Comma-separated; missing or -1 = random. For 4 images: 11, 22 = 11, 22, -1, -1"
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
              message: `Use up to ${imageCount} comma-separated seeds; each must be -1 or 0-${MAX_IMAGE_SEED}.`,
            },
          ]}
        >
          <Input />
        </FormField>
        <Button
          aria-label="Generate new random image seeds"
          title="Generate a new random seed for every image"
          variant="outline"
          size="icon"
          onClick={generateRandomSeeds}
        >
          <span aria-hidden="true">🎲</span>
        </Button>
      </div>
      <FormField name="sampler_name" label="Sampling Method">
        <Select options={SAMPLING_METHOD_OPTIONS} allowClear={false} showSearch />
      </FormField>
    </>
  );
}

function VideoFields({ form }: Pick<CapabilityFormProps, 'form'>) {
  const swapDimensions = () => {
    const width = form.getFieldValue('width');
    const height = form.getFieldValue('height');
    form.setFieldsValue({ width: height, height: width });
  };

  return (
    <>
      <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-end gap-2">
        <FormField name="width" label="Width" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
        <Button
          aria-label="Swap width and height"
          title="Swap width and height"
          variant="outline"
          size="icon"
          onClick={swapDimensions}
        >
          <ArrowLeftRight />
        </Button>
        <FormField name="height" label="Height" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <FormField name="num_frames" label="Frames" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
        <FormField name="fps" label="FPS" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
        <FormField
          name="num_inference_steps"
          label="Inference Steps"
          normalize={normalizeNumberInput}
        >
          <Input type="number" />
        </FormField>
        <FormField name="guidance_scale" label="Guidance Scale" normalize={normalizeNumberInput}>
          <Input type="number" min={1} max={20} step={0.1} />
        </FormField>
        <ScalarSeedField form={form} />
      </div>
    </>
  );
}

export function TextPromptPanel() {
  return (
    <>
      <FormField name="prompt" label="Prompt" rules={[{ required: true }]}>
        <Textarea className="min-h-40" placeholder="Enter prompt..." />
      </FormField>
      <FormField name="max_tokens" label="Max Tokens" normalize={normalizeNumberInput}>
        <Input type="number" min={0} />
      </FormField>
      <FormField name="temperature" label="Temperature" normalize={normalizeNumberInput}>
        <Input type="number" min={0} max={2} step={0.01} />
      </FormField>
    </>
  );
}

export function EmbedPanel() {
  return (
    <FormField
      name="input"
      rules={[{ required: true }]}
      placeholder="Enter text to be vectorized..."
    >
      <Textarea className="min-h-24" />
    </FormField>
  );
}

export function RerankPanel() {
  return (
    <>
      <FormField
        name="query"
        label="Query"
        placeholder="Enter query..."
        rules={[{ required: true }]}
      >
        <Textarea className="min-h-24" />
      </FormField>
      <FormList
        name="documents"
        label="Documents"
        layout="horizontal"
        renderAction={({ add }) => (
          <Button size="sm" type="button" variant="outline" onClick={() => add('')}>
            <Plus />
            Add
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
                  placeholder={`Document ${index + 1}`}
                >
                  <Input />
                </FormField>
                {fields.length > 1 && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="shrink-0 rounded-full text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
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
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload
          accept="image/*,application/pdf"
          label="Upload image or PDF"
          description="PNG, JPG, WebP, scanned page or PDF document"
        />
      </FormField>
      <div className="grid grid-cols-2 gap-3">
        <FormField
          name="ocr_type"
          label="Output Format"
          tooltip="Ocr: Plain text extraction \n Format: Structured document (with annotations) \n Markdown: Standard Markdown format"
        >
          <Select options={OCR_TYPE_OPTIONS} allowClear={false} />
        </FormField>
        <FormField name="model_size" label="Model Size" tooltip="Choose model size configuration">
          <Select options={OCR_MODEL_SIZE_OPTIONS} allowClear={false} />
        </FormField>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <FormField
          name="test_compress"
          label="Test Compress"
          valuePropName="checked"
          layout="horizontal"
          tooltip="Analyze image compression performance"
        >
          <Switch />
        </FormField>
        <FormField
          name="save_results"
          label="Save Results"
          valuePropName="checked"
          layout="horizontal"
          tooltip="Save OCR results to files (if supported)"
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
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload
          accept="image/*"
          label="Upload reference image"
          description="Upload a reference image"
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
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload accept="image/*" label="Upload first frame image" />
      </FormField>
      <PromptFields />
      <VideoFields form={form} />
    </>
  );
}

export function FirstLastFrameVideoPanel({ form }: CapabilityFormProps) {
  return (
    <>
      <div className="grid gap-3 md:grid-cols-2">
        <FormField name="first_frame" rules={[{ required: true }]}>
          <FileUpload accept="image/*" label="First frame" />
        </FormField>
        <FormField name="last_frame" rules={[{ required: true }]}>
          <FileUpload accept="image/*" label="Last frame" />
        </FormField>
      </div>
      <PromptFields />
      <VideoFields form={form} />
    </>
  );
}

const jsonObjectRule = {
  validator: (value: unknown) => isJsonObject(value),
  message: 'Enter a valid JSON object.',
};

function WorldGenerationFields() {
  return (
    <>
      <FormField name="prompt" label="Prompt" rules={[{ required: true }]}>
        <Textarea className="min-h-24" placeholder="Describe the scene or action to generate..." />
      </FormField>
      <FormField
        name="generation_config"
        label="Generation config (JSON)"
        extra="Common generation settings shared by the world API."
        rules={[jsonObjectRule]}
      >
        <Textarea className="min-h-28 font-mono text-xs" spellCheck={false} />
      </FormField>
      <FormField
        name="model_kwargs"
        label="Model kwargs (JSON)"
        extra="Model-specific controls; sent as extra_body."
        rules={[jsonObjectRule]}
      >
        <Textarea className="min-h-28 font-mono text-xs" spellCheck={false} />
      </FormField>
    </>
  );
}

export function TextToWorldPanel() {
  return <WorldGenerationFields />;
}

export function ImageToWorldPanel() {
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload
          accept="image/*"
          label="Upload initial world image"
          description="This image becomes the first frame of the generated world."
        />
      </FormField>
      <WorldGenerationFields />
    </>
  );
}

export function VideoToWorldPanel() {
  return (
    <>
      <FormField name="video" rules={[{ required: true }]}>
        <FileUpload
          accept="video/*"
          label="Upload initial world video"
          description="Use a short source video supported by the selected model."
        />
      </FormField>
      <WorldGenerationFields />
    </>
  );
}

export function AudioToTextPanel({ form }: CapabilityFormProps) {
  return (
    <>
      <FormField name="file" rules={[{ required: true }]}>
        <AudioRecorderUpload form={form} />
      </FormField>
      <FormField name="language" label="Language" placeholder="e.g. en, zh">
        <Input />
      </FormField>
      <FormField name="prompt" label="Prompt" placeholder="Optional context or vocabulary">
        <Textarea />
      </FormField>
      <FormField name="temperature" label="Temperature" normalize={normalizeNumberInput}>
        <Input type="number" min={0} max={1} step={0.1} />
      </FormField>
    </>
  );
}

export function SpeakerEmbeddingPanel() {
  return (
    <FormField name="file" rules={[{ required: true }]}>
      <FileUpload
        accept="audio/*"
        label="Upload a speech sample"
        description="Use clear speech; audio is converted to mono and resampled to 16 kHz."
      />
    </FormField>
  );
}

function EmotionVectorInput({ value, onChange, disabled, error }: BaseFormFieldProps<number[]>) {
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
      {INDEX_TTS_EMOTION_DIMENSIONS.map(({ key, label }, index) => (
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
      ))}
      <div
        className={`flex justify-end text-xs font-medium ${
          totalExceeded ? 'text-destructive' : 'text-muted-foreground'
        }`}
      >
        Total: {total.toFixed(2)} / {INDEX_TTS_EMOTION_MAX_TOTAL.toFixed(2)}
      </div>
    </div>
  );
}

export function SpeechPanel({ form, model }: CapabilityFormProps) {
  const isMusicGeneration = model.model_ability.includes(ModelAbility.Text2music);
  const supportsVoiceCloning = model.model_ability.includes(ModelAbility.Text2audioVoiceCloning);
  const supportsVoiceDesign = model.model_ability.includes(ModelAbility.Text2audioVoiceDesign);
  const promptSpeech = useWatch('prompt_speech', form);
  const hasPromptSpeech =
    supportsVoiceCloning && Array.isArray(promptSpeech) && promptSpeech.length > 0;
  const showPromptSpeech = supportsVoiceCloning;
  const showPromptText = supportsVoiceCloning && (!supportsVoiceDesign || hasPromptSpeech);
  const showVoiceInstruction = supportsVoiceDesign && !hasPromptSpeech;
  const showInstruct = showVoiceInstruction || isMusicGeneration;
  const instructLabel = isMusicGeneration ? 'Music description' : 'Voice Instruction';
  const instructPlaceholder = isMusicGeneration
    ? 'Describe the music to generate'
    : 'Describe the voice to generate';
  const supportsEmotionVector =
    isIndexTTSEmotionModel(model.model_family, model.model_name) &&
    model.model_ability.includes(ModelAbility.Text2audioEmotionControl);
  const emotionVectorEnabled = Boolean(useWatch('use_emo_vector', form));

  return (
    <>
      <FormField
        name="input"
        label={isMusicGeneration ? 'Lyrics' : 'Text'}
        rules={[{ required: true }]}
      >
        <Textarea className="min-h-32" placeholder="Enter text to synthesize..." />
      </FormField>
      {!isMusicGeneration && (
        <div className="grid grid-cols-2 gap-3">
          <FormField name="voice" label="Voice" placeholder="Optional voice ID">
            <Input />
          </FormField>
          <FormField name="speed" label="Speed" normalize={normalizeNumberInput}>
            <Input type="number" min={0.5} max={2} step={0.1} />
          </FormField>
        </div>
      )}
      <div className={isMusicGeneration ? 'grid grid-cols-2 gap-3' : undefined}>
        <ScalarSeedField form={form} />
        {isMusicGeneration && (
          <FormField name="duration" label="Duration (seconds)" normalize={normalizeNumberInput}>
            <Input type="number" />
          </FormField>
        )}
      </div>
      {showPromptSpeech && (
        <FormField name="prompt_speech">
          <FileUpload
            accept="audio/*"
            label="Prompt speech"
            description="Reference audio for cloning"
          />
        </FormField>
      )}
      {showPromptText && (
        <FormField name="prompt_text" label="Prompt Text">
          <Textarea placeholder="Text spoken in the prompt audio" />
        </FormField>
      )}
      {showInstruct && (
        <FormField
          name="instruct"
          label={instructLabel}
          rules={[{ required: isMusicGeneration || showVoiceInstruction }]}
        >
          <Textarea placeholder={instructPlaceholder} />
        </FormField>
      )}
      {supportsEmotionVector && (
        <div className="space-y-3 rounded-lg bg-muted/30 p-3">
          <FormField
            name="use_emo_vector"
            label="Emotion control"
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
                message: 'Use non-negative emotion values with a total no greater than 0.8.',
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
  return (
    <>
      <FormField name="file" rules={[{ required: true }]}>
        <FileUpload
          accept=".pdf,image/*"
          label="Upload document"
          description="PDF, PNG, JPG, WebP, BMP or GIF"
        />
      </FormField>
      <div className="grid grid-cols-2 gap-3">
        <FormField name="backend" label="Backend">
          <Select options={DOCUMENT_BACKEND_OPTIONS} allowClear={false} />
        </FormField>
        <FormField name="parse_method" label="Parse Method">
          <Select options={DOCUMENT_PARSE_METHOD_OPTIONS} allowClear={false} />
        </FormField>
        <FormField name="language" label="Language">
          <Select options={DOCUMENT_LANGUAGE_OPTIONS} allowClear={false} />
        </FormField>
        <FormField name="output_format" label="Output">
          <Select options={DOCUMENT_OUTPUT_OPTIONS} allowClear={false} />
        </FormField>
      </div>
    </>
  );
}
