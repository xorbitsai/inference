'use client';

import { useCallback } from 'react';
import { Plus, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { FileUpload } from '@/components/ui/file-upload';
import { FormField } from '@/components/ui/form-field';
import { FormList } from '@/components/ui/form-list';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { ModelAbility } from '@/constants';
import {
  SAMPLING_METHOD_OPTIONS,
  OCR_TYPE_OPTIONS,
  OCR_MODEL_SIZE_OPTIONS,
} from '@/constants/running';
import type { FileUploadValue } from '@/types/common';

import { ImageEditorCreateMask } from '../components/image-editor-create-mask';
import type { CapabilityFormProps } from '../types';

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

function ImageGenerationFields({ includeImageParams = false }: { includeImageParams?: boolean }) {
  return (
    <>
      <div className="grid grid-cols-2 gap-3">
        <FormField name="width" label="Width" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
        <FormField name="height" label="Height" normalize={normalizeNumberInput}>
          <Input type="number" />
        </FormField>
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
      <FormField name="sampler_name" label="Sampling Method">
        <Select options={SAMPLING_METHOD_OPTIONS} allowClear={false} showSearch />
      </FormField>
    </>
  );
}

function VideoFields() {
  return (
    <div className="grid grid-cols-2 gap-3">
      <FormField name="width" label="Width" normalize={normalizeNumberInput}>
        <Input type="number" />
      </FormField>
      <FormField name="height" label="Height" normalize={normalizeNumberInput}>
        <Input type="number" />
      </FormField>
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
    </div>
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
          accept="image/*"
          label="Upload image"
          description="PNG, JPG, WebP or scanned page"
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

export function TextToImagePanel() {
  return (
    <>
      <PromptFields />
      <ImageGenerationFields />
    </>
  );
}

export function ImageToImagePanel() {
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
      <ImageGenerationFields includeImageParams />
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
      <ImageGenerationFields includeImageParams />
    </>
  );
}

export function TextToVideoPanel() {
  return (
    <>
      <PromptFields />
      <VideoFields />
    </>
  );
}

export function ImageToVideoPanel() {
  return (
    <>
      <FormField name="image" rules={[{ required: true }]}>
        <FileUpload accept="image/*" label="Upload first frame image" />
      </FormField>
      <PromptFields />
      <VideoFields />
    </>
  );
}

export function FirstLastFrameVideoPanel() {
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
      <VideoFields />
    </>
  );
}

export function AudioToTextPanel() {
  return (
    <>
      <FormField name="file" rules={[{ required: true }]}>
        <FileUpload
          accept="audio/*,video/*"
          label="Upload or drop audio"
          description="MP3, WAV, M4A, WebM..."
        />
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

export function SpeechPanel({ model }: CapabilityFormProps) {
  const supportsVoiceCloning = model.model_ability.includes(ModelAbility.Text2audioVoiceCloning);

  return (
    <>
      <FormField name="input" label="Text" rules={[{ required: true }]}>
        <Textarea className="min-h-32" placeholder="Enter text to synthesize..." />
      </FormField>
      <div className="grid grid-cols-2 gap-3">
        <FormField name="voice" label="Voice" placeholder="Optional voice ID">
          <Input />
        </FormField>
        <FormField name="speed" label="Speed" normalize={normalizeNumberInput}>
          <Input type="number" min={0.5} max={2} step={0.1} />
        </FormField>
      </div>
      {supportsVoiceCloning && (
        <>
          <FormField name="prompt_speech">
            <FileUpload
              accept="audio/*"
              label="Prompt speech"
              description="Reference audio for cloning"
            />
          </FormField>
          <FormField name="prompt_text" label="Prompt Text">
            <Textarea placeholder="Text spoken in the prompt audio" />
          </FormField>
        </>
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
