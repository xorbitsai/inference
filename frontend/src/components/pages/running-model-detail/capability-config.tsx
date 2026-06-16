import {
  AudioLines,
  FileText,
  FileSearch,
  ImagePlus,
  ImageUp,
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
  [ModelAbility.Ocr]: {
    ability: ModelAbility.Ocr,
    label: 'OCR',
    icon: ScanText,
    requestApi: '/v1/images/ocr',
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
