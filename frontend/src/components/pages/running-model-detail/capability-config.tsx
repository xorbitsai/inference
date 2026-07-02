import {
  AudioLines,
  Binary,
  FileText,
  FileSearch,
  ImagePlus,
  ImageUp,
  ListFilter,
  Mic,
  Paintbrush,
  ScanText,
  Video,
} from 'lucide-react';

import { ModelAbility } from '@/constants';
import { isEmpty } from '@/lib/is';

import {
  AudioToTextPanel,
  DocumentParsingPanel,
  FirstLastFrameVideoPanel,
  ImageToImagePanel,
  ImageToVideoPanel,
  InpaintingPanel,
  OcrPanel,
  EmbedPanel,
  RerankPanel,
  SpeechPanel,
  TextPromptPanel,
  TextToImagePanel,
  TextToVideoPanel,
} from './panels/form-panels';
import { ResultPanels } from './panels/result-panels';
import type { CapabilityConfig, TransformContext } from './types';
import {
  appendIfPresent,
  booleanValue,
  buildGenerationKwargs,
  firstUpload,
  formatOCRPrompt,
  numberValue,
  sizeFromValues,
  stringValue,
  uploadList,
} from './utils';

function appendKwargsIfPresent(formData: FormData, kwargs: Record<string, unknown>) {
  if (isEmpty(kwargs)) {
    return;
  }

  formData.append('kwargs', JSON.stringify(kwargs));
}

function withKwargsIfPresent<T extends Record<string, unknown>>(
  body: T,
  kwargs: Record<string, unknown>
) {
  if (isEmpty(kwargs)) {
    return body;
  }

  return {
    ...body,
    kwargs: JSON.stringify(kwargs),
  };
}

const imageDefaults = {
  prompt: '',
  negative_prompt: '',
  n: 1,
  width: 1024,
  height: 1024,
  guidance_scale: -1,
  num_inference_steps: -1,
  padding_image_to_multiple: -1,
  strength: 0.6,
  sampler_name: 'default',
};

const videoDefaults = {
  prompt: '',
  negative_prompt: '',
  n: 1,
  width: 512,
  height: 512,
  num_frames: 16,
  fps: 8,
  num_inference_steps: 25,
  guidance_scale: 7.5,
};

const imageKwargsExample = {
  guidance_scale: 7.5,
  num_inference_steps: 25,
};

const videoKwargsExample = {
  width: 512,
  height: 512,
  num_frames: 16,
  fps: 8,
  num_inference_steps: 25,
  guidance_scale: 7.5,
};

function commonImageBody({ modelUid, values, requestId }: TransformContext, fallbackSize?: string) {
  const kwargs = buildGenerationKwargs(values, requestId, { excludeKeys: ['width', 'height'] });

  return withKwargsIfPresent(
    {
      model: modelUid,
      prompt: stringValue(values.prompt),
      negative_prompt: stringValue(values.negative_prompt) || undefined,
      n: Math.max(1, Math.round(numberValue(values.n, 1))),
      size: sizeFromValues(values, fallbackSize),
      response_format: 'b64_json',
    },
    kwargs
  );
}

function appendCommonMediaFormData(formData: FormData, context: TransformContext) {
  const { modelUid, values } = context;
  const size = sizeFromValues(values);
  const kwargs = buildGenerationKwargs(values, context.requestId, {
    excludeKeys: ['width', 'height'],
  });

  formData.append('model', modelUid);
  appendIfPresent(formData, 'prompt', values.prompt);
  appendIfPresent(formData, 'negative_prompt', values.negative_prompt);
  appendIfPresent(formData, 'n', Math.max(1, Math.round(numberValue(values.n, 1))));
  appendIfPresent(formData, 'size', size);
  formData.append('response_format', 'b64_json');
  appendKwargsIfPresent(formData, kwargs);
}

function commonVideoBody({ modelUid, values, requestId }: TransformContext) {
  const kwargs = buildGenerationKwargs(values, requestId);

  return withKwargsIfPresent(
    {
      model: modelUid,
      prompt: stringValue(values.prompt),
      negative_prompt: stringValue(values.negative_prompt) || undefined,
    },
    kwargs
  );
}

function appendCommonVideoFormData(formData: FormData, context: TransformContext) {
  const { modelUid, values } = context;
  const kwargs = buildGenerationKwargs(values, context.requestId);

  formData.append('model', modelUid);
  appendIfPresent(formData, 'prompt', values.prompt);
  appendIfPresent(formData, 'negative_prompt', values.negative_prompt);
  appendKwargsIfPresent(formData, kwargs);
}

function audioTextFormData(context: TransformContext) {
  const { modelUid, values } = context;
  const audio = firstUpload(values, 'file');
  const formData = new FormData();

  formData.append('model', modelUid);
  if (audio) formData.append('file', audio.file);
  appendIfPresent(formData, 'language', values.language);
  appendIfPresent(formData, 'prompt', values.prompt);
  appendIfPresent(formData, 'response_format', values.response_format || 'json');
  appendIfPresent(formData, 'temperature', numberValue(values.temperature, 0));

  return formData;
}

export const CAPABILITY_CONFIGS: Partial<Record<ModelAbility, CapabilityConfig>> = {
  [ModelAbility.Generate]: {
    ability: ModelAbility.Generate,
    label: 'Generate',
    icon: FileText,
    requestApi: '/v1/completions',
    codeExample: {
      method: 'POST',
      contentType: 'json',
      fields: [
        { key: 'model', required: true },
        { key: 'prompt', required: true, value: 'Hello, what can you do?' },
        { key: 'max_tokens', value: 256, comment: 'Optional' },
        { key: 'temperature', value: 1, comment: 'Optional' },
      ],
    },
    initialValues: {
      prompt: '',
      max_tokens: 0,
      temperature: 1,
    },
    formPanel: TextPromptPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: ({ modelUid, values }) => ({
      model: modelUid,
      prompt: stringValue(values.prompt),
      stream: false,
      temperature: numberValue(values.temperature, 1),
      ...(numberValue(values.max_tokens, 0) > 0
        ? { max_tokens: Math.round(numberValue(values.max_tokens, 0)) }
        : {}),
    }),
  },
  [ModelAbility.Embed]: {
    ability: ModelAbility.Embed,
    label: 'Embedding',
    icon: Binary,
    requestApi: '/v1/embeddings',
    codeExample: {
      method: 'POST',
      contentType: 'json',
      fields: [
        { key: 'model', required: true },
        {
          key: 'input',
          required: true,
          value: 'Xinference helps serve models through OpenAI-compatible APIs.',
        },
      ],
    },
    initialValues: {
      input: '',
    },
    formPanel: EmbedPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: ({ modelUid, values }) => ({
      model: modelUid,
      input: stringValue(values.input),
    }),
  },
  [ModelAbility.Rerank]: {
    ability: ModelAbility.Rerank,
    label: 'Rerank',
    icon: ListFilter,
    requestApi: '/v1/rerank',
    codeExample: {
      method: 'POST',
      contentType: 'json',
      fields: [
        { key: 'model', required: true },
        { key: 'query', required: true, value: 'What is Xinference?' },
        {
          key: 'documents',
          required: true,
          value: [
            'Xinference is a model-serving platform.',
            'The weather is sunny today.',
            'It provides OpenAI-compatible APIs for models.',
          ],
        },
      ],
    },
    initialValues: {
      query: '',
      documents: ['', ''],
    },
    formPanel: RerankPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: ({ modelUid, values }) => ({
      model: modelUid,
      query: stringValue(values.query),
      documents: Array.isArray(values.documents)
        ? values.documents.map((item) => stringValue(item).trim()).filter(Boolean)
        : [],
    }),
  },

  [ModelAbility.Ocr]: {
    ability: ModelAbility.Ocr,
    label: 'OCR',
    icon: ScanText,
    requestApi: '/v1/images/ocr',
    codeExample: {
      method: 'POST',
      contentType: 'form',
      fields: [
        { key: 'model', required: true },
        { key: 'image', required: true, type: 'file', value: '/path/to/file.png' },
        {
          key: 'kwargs',
          value: { prompt: 'OCR', model_size: 'gundam', eval_mode: true },
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: {
      image: [],
      ocr_type: 'ocr',
      model_size: 'gundam',
      test_compress: false,
      save_results: false,
    },
    formPanel: OcrPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: ({ modelUid, values }) => {
      const image = firstUpload(values, 'image');
      const ocrType = stringValue(values.ocr_type, 'ocr');
      const formData = new FormData();

      formData.append('model', modelUid);
      if (image) formData.append('image', image.file);
      formData.append(
        'kwargs',
        JSON.stringify({
          prompt: formatOCRPrompt(ocrType),
          model_size: stringValue(values.model_size, 'gundam'),
          test_compress: booleanValue(values.test_compress),
          save_results: booleanValue(values.save_results),
          eval_mode: true,
        })
      );

      return formData;
    },
  },
  [ModelAbility.Docanalyze]: {
    ability: ModelAbility.Docanalyze,
    label: 'Document Parsing',
    icon: FileSearch,
    requestApi: '/v1/images/ocr',
    codeExample: {
      method: 'POST',
      contentType: 'form',
      fields: [
        { key: 'model', required: true },
        { key: 'image', required: true, type: 'file', value: './document.pdf' },
        {
          key: 'kwargs',
          value: {
            backend: 'hybrid-auto-engine',
            parse_method: 'auto',
            language: 'ch',
            output_format: 'markdown',
            return_dict: true,
          },
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: {
      file: [],
      backend: 'hybrid-auto-engine',
      parse_method: 'auto',
      language: 'ch',
      output_format: 'markdown',
    },
    formPanel: DocumentParsingPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: ({ modelUid, values }) => {
      const file = firstUpload(values, 'file');
      const formData = new FormData();

      formData.append('model', modelUid);
      if (file) formData.append('image', file.file);
      formData.append(
        'kwargs',
        JSON.stringify({
          backend: stringValue(values.backend, 'hybrid-auto-engine'),
          parse_method: stringValue(values.parse_method, 'auto'),
          language: stringValue(values.language, 'ch'),
          output_format: stringValue(values.output_format, 'markdown'),
          return_dict: true,
        })
      );

      return formData;
    },
  },
  [ModelAbility.Text2image]: {
    ability: ModelAbility.Text2image,
    label: 'Text to Image',
    icon: ImagePlus,
    requestApi: '/v1/images/generations',
    codeExample: {
      method: 'POST',
      contentType: 'json',
      fields: [
        { key: 'model', required: true },
        {
          key: 'prompt',
          required: true,
          value: 'A cute little cat wearing a tiny astronaut helmet',
        },
        { key: 'negative_prompt', value: 'low quality, blurry', comment: 'Optional' },
        { key: 'n', value: 1, required: true },
        { key: 'size', value: '1024*1024', required: true },
        { key: 'response_format', value: 'b64_json', required: true },
        {
          key: 'kwargs',
          value: imageKwargsExample,
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: imageDefaults,
    showProgress: true,
    formPanel: TextToImagePanel,
    resultPanel: ResultPanels.Universal,
    transformValues: (context) => commonImageBody(context, '1024*1024'),
  },
  [ModelAbility.Image2image]: {
    ability: ModelAbility.Image2image,
    label: 'Image to Image',
    icon: ImageUp,
    requestApi: '/v1/images/variations',
    codeExample: {
      method: 'POST',
      contentType: 'form',
      fields: [
        { key: 'model', required: true },
        { key: 'image', required: true, type: 'file', value: '/path/to/file.png' },
        {
          key: 'prompt',
          required: true,
          value: 'Turn this image into a watercolor illustration',
        },
        { key: 'negative_prompt', value: 'low quality, blurry', comment: 'Optional' },
        { key: 'n', value: 1, required: true },
        { key: 'size', value: '1024*1024', required: true },
        { key: 'response_format', value: 'b64_json', required: true },
        {
          key: 'kwargs',
          value: {
            ...imageKwargsExample,
            strength: 0.6
          },
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: { ...imageDefaults, width: -1, height: -1, image: [] },
    showProgress: true,
    formPanel: ImageToImagePanel,
    resultPanel: ResultPanels.Universal,
    transformValues: (context) => {
      const formData = new FormData();
      appendCommonMediaFormData(formData, context);
      uploadList(context.values, 'image').forEach((item) => formData.append('image', item.file));
      return formData;
    },
  },
  [ModelAbility.Inpainting]: {
    ability: ModelAbility.Inpainting,
    label: 'Inpainting',
    icon: Paintbrush,
    requestApi: '/v1/images/inpainting',
    codeExample: {
      method: 'POST',
      contentType: 'form',
      fields: [
        { key: 'model', required: true },
        { key: 'image', required: true, type: 'file', value: '/path/to/file.png' },
        { key: 'mask_image', required: true, type: 'file', value: '/path/to/mask.png' },
        { key: 'prompt', required: true, value: 'Replace the masked area with a blooming garden' },
        { key: 'negative_prompt', value: 'low quality, blurry', comment: 'Optional' },
        { key: 'n', value: 1, required: true },
        { key: 'size', value: '1024*1024', required: true },
        { key: 'response_format', value: 'b64_json', required: true },
        {
          key: 'kwargs',
          value: {
            ...imageKwargsExample,
            strength: 0.6
          },
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: { ...imageDefaults, width: -1, height: -1, image: [], mask_image: [] },
    showProgress: true,
    formPanel: InpaintingPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: (context) => {
      const image = firstUpload(context.values, 'image');
      const mask = firstUpload(context.values, 'mask_image');
      const formData = new FormData();
      appendCommonMediaFormData(formData, context);
      if (image) formData.append('image', image.file);
      if (mask) formData.append('mask_image', mask.file);
      return formData;
    },
  },
  [ModelAbility.Text2video]: {
    ability: ModelAbility.Text2video,
    label: 'Text to Video',
    icon: Video,
    requestApi: '/v1/video/generations',
    codeExample: {
      method: 'POST',
      contentType: 'json',
      fields: [
        { key: 'model', required: true },
        {
          key: 'prompt',
          required: true,
          value: 'A cute little cat walking through a sunny garden',
        },
        { key: 'negative_prompt', value: 'low quality, blurry', comment: 'Optional' },
        {
          key: 'kwargs',
          value: videoKwargsExample,
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: videoDefaults,
    showProgress: true,
    formPanel: TextToVideoPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: commonVideoBody,
  },
  [ModelAbility.Image2video]: {
    ability: ModelAbility.Image2video,
    label: 'Image to Video',
    icon: Video,
    requestApi: '/v1/video/generations/image',
    codeExample: {
      method: 'POST',
      contentType: 'form',
      fields: [
        { key: 'model', required: true },
        { key: 'image', required: true, type: 'file', value: '/path/to/file.png' },
        { key: 'prompt', required: true, value: 'Animate the scene with gentle camera movement' },
        { key: 'negative_prompt', value: 'low quality, blurry', comment: 'Optional' },
        {
          key: 'kwargs',
          value: videoKwargsExample,
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: { ...videoDefaults, image: [] },
    showProgress: true,
    formPanel: ImageToVideoPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: (context) => {
      const image = firstUpload(context.values, 'image');
      const formData = new FormData();
      appendCommonVideoFormData(formData, context);
      if (image) formData.append('image', image.file);
      return formData;
    },
  },
  [ModelAbility.Firstlastframe2video]: {
    ability: ModelAbility.Firstlastframe2video,
    label: 'First/Last Frame Video',
    icon: Video,
    requestApi: '/v1/video/generations/flf',
    codeExample: {
      method: 'POST',
      contentType: 'form',
      fields: [
        { key: 'model', required: true },
        { key: 'first_frame', required: true, type: 'file', value: '/path/to/first-frame.png' },
        { key: 'last_frame', required: true, type: 'file', value: '/path/to/last-frame.png' },
        { key: 'prompt', required: true, value: 'Create a smooth transition between these frames' },
        { key: 'negative_prompt', value: 'low quality, blurry', comment: 'Optional' },
        {
          key: 'kwargs',
          value: videoKwargsExample,
          stringify: true,
          comment: 'Optional(other key/value)',
        },
      ],
    },
    initialValues: { ...videoDefaults, first_frame: [], last_frame: [] },
    showProgress: true,
    formPanel: FirstLastFrameVideoPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: (context) => {
      const firstFrame = firstUpload(context.values, 'first_frame');
      const lastFrame = firstUpload(context.values, 'last_frame');
      const formData = new FormData();
      appendCommonVideoFormData(formData, context);
      if (firstFrame) formData.append('first_frame', firstFrame.file);
      if (lastFrame) formData.append('last_frame', lastFrame.file);
      return formData;
    },
  },
  [ModelAbility.Audio2text]: {
    ability: ModelAbility.Audio2text,
    label: 'Audio to Text',
    icon: Mic,
    requestApi: '/v1/audio/transcriptions',
    codeExample: {
      method: 'POST',
      contentType: 'form',
      fields: [
        { key: 'model', required: true },
        { key: 'file', required: true, type: 'file', value: '/path/to/audio.map3' },
        { key: 'prompt', value: 'Meeting notes about the product roadmap', comment: 'Optional' },
        { key: 'language', value: 'en', comment: 'Optional' },
        { key: 'temperature', value: 0, comment: 'Optional' },
      ],
    },
    initialValues: {
      file: [],
      language: '',
      prompt: '',
      response_format: 'json',
      temperature: 0,
    },
    formPanel: AudioToTextPanel,
    resultPanel: ResultPanels.Universal,
    transformValues: audioTextFormData,
  },
  [ModelAbility.Text2audio]: {
    ability: ModelAbility.Text2audio,
    label: 'Text to Audio',
    icon: AudioLines,
    requestApi: '/v1/audio/speech',
    codeExample: {
      method: 'POST',
      contentType: 'json',
      fields: [
        { key: 'model', required: true },
        {
          key: 'input',
          required: true,
          value: 'Today is a wonderful day to build something useful.',
        },
        { key: 'voice', value: 'Voice ID', comment: 'Optional' },
        { key: 'speed', value: 1, comment: 'Optional' },
      ],
    },
    initialValues: {
      input: '',
      voice: '',
      speed: 1,
      prompt_speech: [],
      prompt_text: '',
    },
    formPanel: SpeechPanel,
    resultPanel: ResultPanels.Universal,
    responseType: 'blob',
    transformValues: ({ modelUid, values }) => {
      const promptSpeech = firstUpload(values, 'prompt_speech');
      const kwargs: Record<string, unknown> = {};
      const promptText = stringValue(values.prompt_text).trim();

      if (promptText) {
        kwargs.prompt_text = promptText;
      }

      if (promptSpeech) {
        const formData = new FormData();
        formData.append('model', modelUid);
        formData.append('input', stringValue(values.input));
        formData.append('voice', stringValue(values.voice));
        formData.append('speed', String(numberValue(values.speed, 1)));
        formData.append('response_format', 'mp3');
        formData.append('stream', 'false');
        formData.append('prompt_speech', promptSpeech.file);
        appendKwargsIfPresent(formData, kwargs);
        return formData;
      }
      return withKwargsIfPresent(
        {
          model: modelUid,
          input: stringValue(values.input),
          voice: values.voice || undefined,
          speed: numberValue(values.speed, 1),
          response_format: 'mp3',
          stream: false,
        },
        kwargs
      );
    },
  },
};
