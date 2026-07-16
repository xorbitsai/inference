'use client';

import { useCallback, useEffect, useId, useMemo, useState, useRef } from 'react';
import { Ban, Rocket } from 'lucide-react';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import request from '@/lib/request';
import { ModelType, ModelAbility } from '@/constants';
import { ENGINES_WITH_WORKER } from '@/constants/launch';
import { ModelFormat } from '@/constants/register';
import { useI18n } from '@/contexts/i18n-context';
import { useForm, useFormValues, useWatch } from '@/hooks/use-form';
import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Form } from '@/components/ui/form';
import type { ModelEngine, ModelEngineItem, ReplicaItem } from '@/types/services';
import type { FormValues } from '@/types/form';
import CollapsibleConfig from './advanced-config';
import ConfigCache, { getLatestModelConfigHistory, saveLaunchConfigHistory } from './config-cache';
import type { CatalogModel, LaunchFieldConfig, RequestModelType } from '../types';
import {
  MODEL_ENGINE_TYPES,
  buildEngineIndex,
  createCacheKey,
  isCachedSpec,
  normalizeModelSize,
  range,
  renderLaunchFields,
  syncLinkedField,
  toOptionValue,
  transformFetchToForm,
  transformFormToFetch,
  normalizeProgress,
  normalizeReplicaStatuses,
  isEmptyLaunchValue,
  isVisibleRequiredLaunchField,
} from '../utils';
import CommandLine from './command-line';
import { FormField } from '@/components/ui/form-field';

interface LaunchDialogProps {
  model?: CatalogModel;
  modelType: RequestModelType;
  gpuAvailable: number;
  onOpenChange: (open: boolean) => void;
}

export default function LaunchDialog({
  model,
  modelType,
  gpuAvailable,
  onOpenChange,
}: LaunchDialogProps) {
  const isOpen = Boolean(model);
  const formId = useId();
  const [form] = useForm();
  const { t } = useI18n();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [canceling, setCanceling] = useState(false);
  const [saveAutostart, setSaveAutostart] = useState(false);
  const [progress, setProgress] = useState(0);
  const [replicaStatuses, setReplicaStatuses] = useState<ReplicaItem[]>([]);
  const [configCacheRefreshKey, setConfigCacheRefreshKey] = useState(0);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isCanceledLaunchRef = useRef(false);
  const isLLM = modelType === ModelType.LLM;
  const [modelEngineMap, setModelEngineMap] = useState<ModelEngine>({});
  const launchFormValues = useFormValues(form);
  const modelEngineValue = toOptionValue(useWatch('model_engine', form));
  const modelFormatValue = toOptionValue(useWatch('model_format', form));
  const modelSizeInBillionsValue = useWatch('model_size_in_billions', form) as
    | ModelEngineItem['model_size_in_billions']
    | undefined;
  const enableThinkingValue = useWatch('enable_thinking', form);
  const modelSizeInBillionsKey = toOptionValue(modelSizeInBillionsValue);
  const quantizationValue = toOptionValue(useWatch('quantization', form));
  const multimodalProjectorValue = toOptionValue(useWatch('multimodal_projector', form));
  const nGpuValue = useWatch('n_gpu', form);

  const fetchModelEngine = useCallback(async () => {
    if (!model?.model_name || !MODEL_ENGINE_TYPES.includes(modelType)) {
      setModelEngineMap({});
      return;
    }

    const url = isLLM
      ? `/v1/engines/${model.model_name}`
      : `/v1/engines/${modelType}/${model.model_name}`;

    setModelEngineMap({});

    const res = await request.get(url);

    setModelEngineMap(res || {});
  }, [isLLM, model?.model_name, modelType]);

  const engineIndex = useMemo(() => buildEngineIndex(modelEngineMap), [modelEngineMap]);
  const cacheIndex = useMemo(() => {
    const formats = new Set<string>();
    const sizes = new Set<string>();
    const quantizations = new Set<string>();

    (model?.modelSpecs || []).forEach((spec) => {
      const format = toOptionValue(spec.model_format);
      const size = normalizeModelSize(spec.model_size_in_billions);

      if (!format || !isCachedSpec(spec)) return;

      formats.add(format);

      if (size) {
        sizes.add(createCacheKey(format, size));
      }

      if (Array.isArray(spec.quantizations)) {
        spec.quantizations.forEach((quantization, index) => {
          const cached = Array.isArray(spec.cache_status)
            ? spec.cache_status[index]
            : spec.cache_status;

          if (!cached) return;

          quantizations.add(createCacheKey(format, size, quantization));
          quantizations.add(createCacheKey(format, '', quantization));
        });
      } else if (spec.quantization) {
        quantizations.add(createCacheKey(format, size, spec.quantization));
        quantizations.add(createCacheKey(format, '', spec.quantization));
      }
    });

    return { formats, sizes, quantizations };
  }, [model?.modelSpecs]);

  const modelEngineOptions = useMemo(() => {
    return Object.entries(modelEngineMap).map(([key, engineData]) => {
      if (typeof engineData === 'string') {
        return {
          label: `${key} (${engineData})`,
          value: key,
          disabled: true,
        };
      }
      const cached = engineData.some((item) => cacheIndex.formats.has(item.model_format));

      return {
        label: key,
        value: key,
        suffix: cached ? t('launchModel.cached') : undefined,
      };
    });
  }, [cacheIndex.formats, modelEngineMap, t]);

  const selectedEngineFormats = engineIndex.get(modelEngineValue);
  const selectedFormatIndex = selectedEngineFormats?.get(modelFormatValue);

  const modelFormatOptions = useMemo(
    () =>
      Array.from(selectedEngineFormats?.keys() || []).map((format) => ({
        label: format,
        value: format,
        suffix: cacheIndex.formats.has(format) ? t('launchModel.cached') : undefined,
      })),
    [cacheIndex.formats, selectedEngineFormats, t]
  );
  const modelSizeInBillionsOptions = useMemo(
    () =>
      Array.from(selectedFormatIndex?.sizes.values() || []).map((sizeIndex) => ({
        label: String(sizeIndex.value),
        value: sizeIndex.value,
        suffix: cacheIndex.sizes.has(
          createCacheKey(modelFormatValue, normalizeModelSize(sizeIndex.value))
        )
          ? t('launchModel.cached')
          : undefined,
      })),
    [cacheIndex.sizes, modelFormatValue, selectedFormatIndex, t]
  );

  const quantizationOptions = useMemo(() => {
    const quantizations =
      modelType === ModelType.LLM
        ? selectedFormatIndex?.sizes.get(modelSizeInBillionsKey)?.quantizations
        : selectedFormatIndex?.quantizations;

    return Array.from(quantizations || []).map((quantization) => ({
      label: quantization,
      value: quantization,
      suffix: cacheIndex.quantizations.has(
        createCacheKey(
          modelFormatValue,
          modelType === ModelType.LLM ? normalizeModelSize(modelSizeInBillionsKey) : '',
          quantization
        )
      )
        ? t('launchModel.cached')
        : undefined,
    }));
  }, [
    cacheIndex.quantizations,
    modelFormatValue,
    modelSizeInBillionsKey,
    modelType,
    selectedFormatIndex,
    t,
  ]);

  const multimodalProjectorOptions = useMemo(
    () =>
      Array.from(
        selectedFormatIndex?.sizes.get(modelSizeInBillionsKey)?.multimodalProjectors || []
      ).map((projector) => ({
        label: projector,
        value: projector,
      })),
    [modelSizeInBillionsKey, selectedFormatIndex]
  );
  const nGpuOptions = useMemo(() => {
    let options = [];
    if ([ModelType.LLM, ModelType.Image].includes(modelType)) {
      options = gpuAvailable > 0 ? ['auto', 'CPU', ...range(1, gpuAvailable)] : ['auto', 'CPU'];
    } else {
      options = gpuAvailable === 0 ? ['CPU'] : ['GPU', 'CPU'];
    }
    return options.map((item) => ({ label: String(item), value: item }));
  }, [gpuAvailable, modelType]);

  const downloadHubOptions = useMemo(
    () => ['none', ...(model?.download_hubs || [])].map((item) => ({ label: item, value: item })),
    [model?.download_hubs]
  );
  const ggufQuantizations =
    model?.gguf_quantizations ??
    model?.modelSpecs?.find((spec) => spec.gguf_quantizations)?.gguf_quantizations ??
    null;

  const ggufQuantizationsOptions = useMemo(() => {
    if (Array.isArray(ggufQuantizations)) {
      return ['none', ...ggufQuantizations].map((item) => ({
        label: item,
        value: item,
      }));
    }

    return [];
  }, [ggufQuantizations]);

  const lightningVersionOptions = useMemo(() => {
    if (Array.isArray(model?.lightning_versions)) {
      return ['none', ...(model?.lightning_versions || [])].map((item) => ({
        label: item,
        value: item,
      }));
    }
    return [];
  }, [model?.lightning_versions]);
  useEffect(() => {
    const selectableOptions = modelEngineOptions.filter((option) => !option.disabled);

    if (
      modelEngineValue &&
      selectableOptions.length > 0 &&
      !selectableOptions.some((option) => option.value === modelEngineValue)
    ) {
      form.setFieldValue('model_engine', '');
    }
  }, [form, modelEngineOptions, modelEngineValue]);

  useEffect(() => {
    if (modelEngineValue && modelFormatOptions.length === 0) return;

    syncLinkedField(form, 'model_format', modelFormatValue, modelFormatOptions);
  }, [form, modelEngineValue, modelFormatOptions, modelFormatValue]);

  useEffect(() => {
    if (
      modelType === ModelType.LLM &&
      modelFormatValue &&
      modelSizeInBillionsOptions.length === 0
    ) {
      return;
    }

    syncLinkedField(
      form,
      'model_size_in_billions',
      modelSizeInBillionsValue,
      modelType === ModelType.LLM ? modelSizeInBillionsOptions : []
    );
  }, [form, modelFormatValue, modelSizeInBillionsOptions, modelSizeInBillionsValue, modelType]);

  useEffect(() => {
    if (modelFormatValue && quantizationOptions.length === 0) return;

    syncLinkedField(form, 'quantization', quantizationValue, quantizationOptions);
  }, [form, modelFormatValue, quantizationOptions, quantizationValue]);

  useEffect(() => {
    if (multimodalProjectorValue && multimodalProjectorOptions.length === 0) return;

    syncLinkedField(
      form,
      'multimodal_projector',
      multimodalProjectorValue,
      multimodalProjectorOptions,
      true
    );
  }, [form, multimodalProjectorOptions, multimodalProjectorValue]);

  const modelTypeFields: Partial<Record<ModelType, LaunchFieldConfig[]>> = {
    [ModelType.LLM]: [
      {
        name: 'model_engine',
        type: 'select',
        label: t('launchModel.modelEngine'),
        rules: [{ required: true }],
        fieldProps: { options: modelEngineOptions },
      },
      {
        name: 'model_format',
        type: 'select',
        label: t('launchModel.modelFormat'),
        rules: [{ required: true }],
        disabled: !modelEngineValue,
        fieldProps: { options: modelFormatOptions },
      },
      {
        name: 'model_size_in_billions',
        type: 'select',
        label: t('launchModel.modelSize'),
        rules: [{ required: true }],
        disabled: !modelFormatValue,
        fieldProps: { options: modelSizeInBillionsOptions },
      },
      {
        name: 'quantization',
        type: 'select',
        label: t('launchModel.quantization'),
        rules: [{ required: true }],
        disabled: !modelSizeInBillionsValue,
        fieldProps: { options: quantizationOptions },
      },
      {
        name: 'multimodal_projector',
        type: 'select',
        label: t('launchModel.multimodelProjector'),
        rules: [{ required: true }],
        disabled: !modelSizeInBillionsValue,
        show: Boolean(multimodalProjectorOptions.length),
        fieldProps: { options: multimodalProjectorOptions },
      },
      {
        name: 'n_gpu',
        type: 'select',
        label: t(
          ENGINES_WITH_WORKER.includes(modelEngineValue)
            ? 'launchModel.nGPUPerWorker'
            : 'launchModel.nGPU'
        ),
        fieldProps: { options: nGpuOptions },
      },
      {
        name: 'n_gpu_layers',
        type: 'input',
        label: t('launchModel.nGpuLayers'),
        show: [ModelFormat.GGMLV3, ModelFormat.GGUFV2].includes(modelFormatValue as ModelFormat),
        fieldProps: { type: 'number', min: -1 },
      },
      {
        name: 'replica',
        type: 'input',
        label: t('launchModel.replica'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'model_uid',
        type: 'input',
        label: t('launchModel.modelUid'),
        placeholder: t('launchModel.modelUidPlaceholder'),
      },
      {
        name: 'request_limits',
        type: 'input',
        label: t('launchModel.requestLimits'),
        placeholder: t('launchModel.requestLimitsPlaceholder'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
        normalize: (v) => (v === '' ? undefined : Number(v)),
      },
      {
        name: 'n_worker',
        type: 'input',
        label: t('launchModel.workerCount'),
        placeholder: t('launchModel.workerCountPlaceholder'),
        show: ENGINES_WITH_WORKER.includes(modelEngineValue),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'gpu_idx',
        type: 'input',
        label: t('launchModel.GPUIdx'),
        placeholder: t('launchModel.GPUIdxPlaceholder'),
        rules: [
          {
            pattern: /^\d+(?:,\d+)*$/,
            message: t('launchModel.enterCommaSeparatedNumbers'),
          },
        ],
      },
      {
        name: 'download_hub',
        type: 'select',
        label: t('launchModel.downloadHub'),
        placeholder: t('launchModel.downloadHubPlaceholder'),
        fieldProps: { options: downloadHubOptions },
      },
      {
        name: 'enable_thinking',
        type: 'switch',
        label: t('launchModel.enableThinking'),
        valuePropName: 'checked',
        show: model?.abilities.includes(ModelAbility.Hybrid),
      },
      {
        name: 'reasoning_content',
        type: 'switch',
        label: t('launchModel.parsingReasoningContent'),
        valuePropName: 'checked',
        show: model?.abilities.includes(ModelAbility.Reasoning) && enableThinkingValue,
      },
      {
        name: 'worker_ip',
        type: 'input',
        label: t('launchModel.workerIp'),
        placeholder: t('launchModel.workerIpPlaceholder'),
        colSpan: 2,
        normalize: (v) => v || undefined,
      },
      {
        name: 'model_path',
        type: 'input',
        label: t('launchModel.modelPath'),
        placeholder: t('launchModel.modelPathPlaceholder'),
        colSpan: 2,
      },
      {
        name: 'collapsibleConfig',
        type: 'custom',
        colSpan: 2,
        content: <CollapsibleConfig form={form} modelType={modelType} />,
      },
    ],
    [ModelType.Embedding]: [
      {
        name: 'model_engine',
        type: 'select',
        label: t('launchModel.modelEngine'),
        rules: [{ required: true }],
        fieldProps: { options: modelEngineOptions },
      },
      {
        name: 'model_format',
        type: 'select',
        label: t('launchModel.modelFormat'),
        rules: [{ required: true }],
        disabled: !modelEngineValue,
        fieldProps: { options: modelFormatOptions },
      },
      {
        name: 'quantization',
        type: 'select',
        label: t('launchModel.quantization'),
        rules: [{ required: true }],
        disabled: !modelFormatValue,
        fieldProps: { options: quantizationOptions },
      },
      {
        name: 'replica',
        type: 'input',
        label: t('launchModel.replica'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'n_gpu',
        type: 'select',
        label: t('launchModel.nGPU'),
        fieldProps: { options: nGpuOptions },
      },
      {
        name: 'gpu_idx',
        type: 'input',
        label: t('launchModel.GPUIdx'),
        placeholder: t('launchModel.GPUIdxPlaceholder'),
        rules: [
          {
            pattern: /^\d+(?:,\d+)*$/,
            message: t('launchModel.enterCommaSeparatedNumbers'),
          },
        ],
        show: nGpuValue === 'GPU',
      },
      {
        name: 'model_uid',
        type: 'input',
        label: t('launchModel.modelUid'),
        placeholder: t('launchModel.modelUidPlaceholder'),
      },
      {
        name: 'download_hub',
        type: 'select',
        label: t('launchModel.downloadHub'),
        placeholder: t('launchModel.downloadHubPlaceholder'),
        fieldProps: { options: downloadHubOptions },
      },
      {
        name: 'request_limits',
        type: 'input',
        label: t('launchModel.requestLimits'),
        placeholder: t('launchModel.requestLimitsPlaceholder'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
        normalize: (v) => (v === '' ? undefined : Number(v)),
      },
      {
        name: 'worker_ip',
        type: 'input',
        label: t('launchModel.workerIp'),
        placeholder: t('launchModel.workerIpPlaceholder'),
        colSpan: 2,
        normalize: (v) => v || undefined,
      },
      {
        name: 'model_path',
        type: 'input',
        label: t('launchModel.modelPath'),
        placeholder: t('launchModel.modelPathPlaceholder'),
        colSpan: 2,
      },
      {
        name: 'collapsibleConfig',
        type: 'custom',
        colSpan: 2,
        content: <CollapsibleConfig form={form} modelType={modelType} />,
      },
    ],
    [ModelType.Rerank]: [
      {
        name: 'model_engine',
        type: 'select',
        label: t('launchModel.modelEngine'),
        rules: [{ required: true }],
        fieldProps: { options: modelEngineOptions },
      },
      {
        name: 'model_format',
        type: 'select',
        label: t('launchModel.modelFormat'),
        rules: [{ required: true }],
        disabled: !modelEngineValue,
        fieldProps: { options: modelFormatOptions },
      },
      {
        name: 'quantization',
        type: 'select',
        label: t('launchModel.quantization'),
        rules: [{ required: true }],
        disabled: !modelFormatValue,
        fieldProps: { options: quantizationOptions },
      },
      {
        name: 'replica',
        type: 'input',
        label: t('launchModel.replica'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'n_gpu',
        type: 'select',
        label: t('launchModel.nGPU'),
        fieldProps: { options: nGpuOptions },
      },
      {
        name: 'gpu_idx',
        type: 'input',
        label: t('launchModel.GPUIdx'),
        placeholder: t('launchModel.GPUIdxPlaceholder'),
        rules: [
          {
            pattern: /^\d+(?:,\d+)*$/,
            message: t('launchModel.enterCommaSeparatedNumbers'),
          },
        ],
        show: nGpuValue === 'GPU',
      },
      {
        name: 'model_uid',
        type: 'input',
        label: t('launchModel.modelUid'),
        placeholder: t('launchModel.modelUidPlaceholder'),
      },
      {
        name: 'download_hub',
        type: 'select',
        label: t('launchModel.downloadHub'),
        placeholder: t('launchModel.downloadHubPlaceholder'),
        fieldProps: { options: downloadHubOptions },
      },
      {
        name: 'gguf_quantization',
        label: t('launchModel.GGUFQuantization'),
        placeholder: t('launchModel.GGUFQuantizationPlaceholder'),
        type: 'select',
        fieldProps: { options: ggufQuantizationsOptions },
        show: !!ggufQuantizations,
      },
      {
        name: 'gguf_model_path',
        label: t('launchModel.GGUFModelPath'),
        placeholder: t('launchModel.GGUFModelPathPlaceholder'),
        type: 'input',
        show: !!ggufQuantizations,
      },
      {
        name: 'request_limits',
        type: 'input',
        label: t('launchModel.requestLimits'),
        placeholder: t('launchModel.requestLimitsPlaceholder'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
        normalize: (v) => (v === '' ? undefined : Number(v)),
      },
      {
        name: 'worker_ip',
        type: 'input',
        label: t('launchModel.workerIp'),
        placeholder: t('launchModel.workerIpPlaceholder'),
        colSpan: 2,
        normalize: (v) => v || undefined,
      },
      {
        name: 'model_path',
        type: 'input',
        label: t('launchModel.modelPath'),
        placeholder: t('launchModel.modelPathPlaceholder'),
        colSpan: 2,
      },
      {
        name: 'collapsibleConfig',
        type: 'custom',
        colSpan: 2,
        content: <CollapsibleConfig form={form} modelType={modelType} />,
      },
    ],
    [ModelType.Image]: [
      {
        name: 'model_uid',
        type: 'input',
        label: t('launchModel.modelUid'),
        placeholder: t('launchModel.modelUidPlaceholder'),
      },
      {
        name: 'model_engine',
        type: 'select',
        label: t('launchModel.modelEngine'),
        fieldProps: { options: modelEngineOptions },
        show: !!modelEngineOptions.length,
      },
      {
        name: 'model_format',
        type: 'select',
        label: t('launchModel.modelFormat'),
        disabled: !modelEngineValue,
        fieldProps: { options: modelFormatOptions },
        show: !!modelFormatOptions.length,
      },
      {
        name: 'quantization',
        type: 'select',
        label: t('launchModel.quantization'),
        disabled: !modelFormatValue,
        fieldProps: { options: quantizationOptions },
        show: !!quantizationOptions.length,
      },
      {
        name: 'replica',
        type: 'input',
        label: t('launchModel.replica'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'n_gpu',
        type: 'select',
        label: t('launchModel.nGPU'),
        fieldProps: { options: nGpuOptions },
      },
      {
        name: 'gpu_idx',
        type: 'input',
        label: t('launchModel.GPUIdx'),
        placeholder: t('launchModel.GPUIdxPlaceholder'),
        rules: [
          {
            pattern: /^\d+(?:,\d+)*$/,
            message: t('launchModel.enterCommaSeparatedNumbers'),
          },
        ],
        show: nGpuValue === 'GPU',
      },
      {
        name: 'download_hub',
        type: 'select',
        label: t('launchModel.downloadHub'),
        placeholder: t('launchModel.downloadHubPlaceholder'),
        fieldProps: { options: downloadHubOptions },
      },
      {
        name: 'request_limits',
        type: 'input',
        label: t('launchModel.requestLimits'),
        placeholder: t('launchModel.requestLimitsPlaceholder'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
        normalize: (v) => (v === '' ? undefined : Number(v)),
      },
      {
        name: 'gguf_quantization',
        label: t('launchModel.GGUFQuantization'),
        placeholder: t('launchModel.GGUFQuantizationPlaceholder'),
        type: 'select',
        fieldProps: { options: ggufQuantizationsOptions },
        show: !!ggufQuantizations,
      },
      {
        name: 'gguf_model_path',
        label: t('launchModel.GGUFModelPath'),
        placeholder: t('launchModel.GGUFModelPathPlaceholder'),
        type: 'input',
        show: !!ggufQuantizations,
      },
      {
        name: 'lightning_version',
        label: t('launchModel.lightningVersions'),
        placeholder: t('launchModel.lightningVersionsPlaceholder'),
        type: 'select',
        fieldProps: { options: lightningVersionOptions },
        show: !!lightningVersionOptions.length,
      },
      {
        name: 'lightning_model_path',
        label: t('launchModel.lightningModelPath'),
        placeholder: t('launchModel.lightningModelPathPlaceholder'),
        type: 'input',
        show: !!lightningVersionOptions.length,
      },
      {
        name: 'worker_ip',
        type: 'input',
        label: t('launchModel.workerIp'),
        placeholder: t('launchModel.workerIpPlaceholder'),
        colSpan: 2,
        normalize: (v) => v || undefined,
      },
      {
        name: 'model_path',
        type: 'input',
        label: t('launchModel.modelPath'),
        placeholder: t('launchModel.modelPathPlaceholder'),
        colSpan: 2,
      },
      {
        name: 'cpu_offload',
        label: t('launchModel.CPUOffload'),
        type: 'switch',
        valuePropName: 'checked',
        tooltip: t('launchModel.CPUOffloadTip'),
      },
      {
        name: 'collapsibleConfig',
        type: 'custom',
        colSpan: 2,
        content: <CollapsibleConfig form={form} modelType={modelType} />,
      },
    ],
    [ModelType.Audio]: [
      {
        name: 'model_uid',
        type: 'input',
        label: t('launchModel.modelUid'),
        placeholder: t('launchModel.modelUidPlaceholder'),
      },
      {
        name: 'replica',
        type: 'input',
        label: t('launchModel.replica'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'n_gpu',
        type: 'select',
        label: t('launchModel.nGPU'),
        fieldProps: { options: nGpuOptions },
      },
      {
        name: 'gpu_idx',
        type: 'input',
        label: t('launchModel.GPUIdx'),
        placeholder: t('launchModel.GPUIdxPlaceholder'),
        rules: [
          {
            pattern: /^\d+(?:,\d+)*$/,
            message: t('launchModel.enterCommaSeparatedNumbers'),
          },
        ],
        show: nGpuValue === 'GPU',
      },

      {
        name: 'download_hub',
        type: 'select',
        label: t('launchModel.downloadHub'),
        placeholder: t('launchModel.downloadHubPlaceholder'),
        fieldProps: { options: downloadHubOptions },
      },
      {
        name: 'request_limits',
        type: 'input',
        label: t('launchModel.requestLimits'),
        placeholder: t('launchModel.requestLimitsPlaceholder'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
        normalize: (v) => (v === '' ? undefined : Number(v)),
      },
      {
        name: 'gguf_quantization',
        label: t('launchModel.GGUFQuantization'),
        placeholder: t('launchModel.GGUFQuantizationPlaceholder'),
        type: 'select',
        fieldProps: { options: ggufQuantizationsOptions },
        show: !!ggufQuantizations,
      },
      {
        name: 'gguf_model_path',
        label: t('launchModel.GGUFModelPath'),
        placeholder: t('launchModel.GGUFModelPathPlaceholder'),
        type: 'input',
        show: !!ggufQuantizations,
      },
      {
        name: 'worker_ip',
        type: 'input',
        label: t('launchModel.workerIp'),
        placeholder: t('launchModel.workerIpPlaceholder'),
        colSpan: 2,
        normalize: (v) => v || undefined,
      },
      {
        name: 'model_path',
        type: 'input',
        label: t('launchModel.modelPath'),
        placeholder: t('launchModel.modelPathPlaceholder'),
        colSpan: 2,
      },
      {
        name: 'collapsibleConfig',
        type: 'custom',
        colSpan: 2,
        content: <CollapsibleConfig form={form} modelType={modelType} />,
      },
    ],
    [ModelType.Video]: [
      {
        name: 'model_uid',
        type: 'input',
        label: t('launchModel.modelUid'),
        placeholder: t('launchModel.modelUidPlaceholder'),
      },
      {
        name: 'replica',
        type: 'input',
        label: t('launchModel.replica'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'n_gpu',
        type: 'select',
        label: t('launchModel.nGPU'),
        fieldProps: { options: nGpuOptions },
      },
      {
        name: 'gpu_idx',
        type: 'input',
        label: t('launchModel.GPUIdx'),
        placeholder: t('launchModel.GPUIdxPlaceholder'),
        rules: [
          {
            pattern: /^\d+(?:,\d+)*$/,
            message: t('launchModel.enterCommaSeparatedNumbers'),
          },
        ],
        show: nGpuValue === 'GPU',
      },

      {
        name: 'download_hub',
        type: 'select',
        label: t('launchModel.downloadHub'),
        placeholder: t('launchModel.downloadHubPlaceholder'),
        fieldProps: { options: downloadHubOptions },
      },
      {
        name: 'request_limits',
        type: 'input',
        label: t('launchModel.requestLimits'),
        placeholder: t('launchModel.requestLimitsPlaceholder'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
        normalize: (v) => (v === '' ? undefined : Number(v)),
      },
      {
        name: 'gguf_quantization',
        label: t('launchModel.GGUFQuantization'),
        placeholder: t('launchModel.GGUFQuantizationPlaceholder'),
        type: 'select',
        fieldProps: { options: ggufQuantizationsOptions },
        show: !!ggufQuantizations,
      },
      {
        name: 'gguf_model_path',
        label: t('launchModel.GGUFModelPath'),
        placeholder: t('launchModel.GGUFModelPathPlaceholder'),
        type: 'input',
        show: !!ggufQuantizations,
      },
      {
        name: 'worker_ip',
        type: 'input',
        label: t('launchModel.workerIp'),
        placeholder: t('launchModel.workerIpPlaceholder'),
        colSpan: 2,
        normalize: (v) => v || undefined,
      },
      {
        name: 'model_path',
        type: 'input',
        label: t('launchModel.modelPath'),
        placeholder: t('launchModel.modelPathPlaceholder'),
        colSpan: 2,
      },
      {
        name: 'cpu_offload',
        label: t('launchModel.CPUOffload'),
        type: 'switch',
        valuePropName: 'checked',
        tooltip: t('launchModel.CPUOffloadTip'),
      },
      {
        name: 'collapsibleConfig',
        type: 'custom',
        colSpan: 2,
        content: <CollapsibleConfig form={form} modelType={modelType} />,
      },
    ],
    [ModelType.Flexible]: [
      {
        name: 'model_uid',
        type: 'input',
        label: t('launchModel.modelUid'),
        placeholder: t('launchModel.modelUidPlaceholder'),
      },
      {
        name: 'replica',
        type: 'input',
        label: t('launchModel.replica'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
      },
      {
        name: 'n_gpu',
        type: 'select',
        label: t('launchModel.nGPU'),
        fieldProps: { options: nGpuOptions },
      },
      {
        name: 'gpu_idx',
        type: 'input',
        label: t('launchModel.GPUIdx'),
        placeholder: t('launchModel.GPUIdxPlaceholder'),
        rules: [
          {
            pattern: /^\d+(?:,\d+)*$/,
            message: t('launchModel.enterCommaSeparatedNumbers'),
          },
        ],
        show: nGpuValue === 'GPU',
      },
      {
        name: 'request_limits',
        type: 'input',
        label: t('launchModel.requestLimits'),
        placeholder: t('launchModel.requestLimitsPlaceholder'),
        rules: [
          {
            pattern: /^[1-9]\d*$/,
            message: t('launchModel.enterIntegerGreaterThanZero'),
          },
        ],
        fieldProps: { type: 'number', min: 1 },
        normalize: (v) => (v === '' ? undefined : Number(v)),
      },
      {
        name: 'worker_ip',
        type: 'input',
        label: t('launchModel.workerIp'),
        placeholder: t('launchModel.workerIpPlaceholder'),
        colSpan: 2,
        normalize: (v) => v || undefined,
      },
      {
        name: 'model_path',
        type: 'input',
        label: t('launchModel.modelPath'),
        placeholder: t('launchModel.modelPathPlaceholder'),
        colSpan: 2,
      },
      {
        name: 'collapsibleConfig',
        type: 'custom',
        colSpan: 2,
        content: <CollapsibleConfig form={form} modelType={modelType} />,
      },
    ],
  };

  const currentLaunchFields = modelTypeFields[modelType] || [];
  // required fields is filled in
  const isReady = currentLaunchFields
    .filter(isVisibleRequiredLaunchField)
    .every((field) => !isEmptyLaunchValue(launchFormValues[field.name]));

  const stopPolling = useCallback(() => {
    if (pollingRef.current !== null) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const fetchProgress = useCallback(async () => {
    const modelUid = form.getFieldValue('model_uid') || model?.model_name;
    try {
      const [progressRes, replicaRes] = await Promise.all([
        request.get<number | string | { progress?: number | string }>(
          `/v1/models/${modelUid}/progress`
        ),
        request.get<unknown>(`/v1/models/${modelUid}/replicas`),
      ]);

      const progressValue =
        progressRes && typeof progressRes === 'object' ? progressRes.progress : progressRes;
      const nextProgress = normalizeProgress(progressValue);

      setProgress(nextProgress);
      setReplicaStatuses(normalizeReplicaStatuses(replicaRes));

      if (nextProgress >= 100) {
        stopPolling();
      }
    } catch {
      stopPolling();
    }
  }, [stopPolling, form, model]);

  const startPolling = useCallback(() => {
    if (pollingRef.current) return;
    pollingRef.current = setInterval(fetchProgress, 1000);
  }, [fetchProgress]);

  const renderReplicaStatuses = () => {
    if (!replicaStatuses.length) {
      return (
        <div className="text-sm text-muted-foreground">{t('launchModel.noReplicaStatus')}</div>
      );
    }

    return (
      <div className="max-w-md space-y-2 overflow-auto">
        <div className="text-sm font-medium">{t('launchModel.launchProgress')}</div>
        {replicaStatuses.map((replica) => {
          const statusClassName =
            replica.status === 'READY'
              ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
              : replica.status === 'ERROR'
                ? 'border-red-200 bg-red-50 text-red-700'
                : 'border-border bg-muted/40 text-muted-foreground';

          return (
            <div key={replica.replica_id} className="flex items-center justify-between gap-8">
              <div className="flex flex-col gap-0.5 text-xs">
                <span className="font-semibold">
                  {t('launchModel.replica')}&nbsp;{replica.replica_id}
                </span>
                <span className="text-muted-foreground">{replica?.worker_address || '-'}</span>
              </div>
              <div
                className={cn(
                  'rounded-md border px-2 py-1 text-xs font-medium leading-none',
                  statusClassName
                )}
              >
                {replica.status}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const handleCancelLaunch = async () => {
    const modelUid = form.getFieldValue('model_uid') || model?.model_name;
    setCanceling(true);

    try {
      await request.post(`/v1/models/${encodeURIComponent(modelUid)}/cancel`);
      isCanceledLaunchRef.current = true;
      stopPolling();
      setLoading(false);
      setProgress(0);
      setReplicaStatuses([]);
      toast.success(t('launchModel.launchCanceled'));
    } finally {
      setCanceling(false);
    }
  };

  const handleLaunch = async (values: FormValues) => {
    const newValues = transformFormToFetch(values);

    isCanceledLaunchRef.current = false;
    setLoading(true);
    setProgress(0);
    setReplicaStatuses([]);

    request
      .post<{ model_uid?: string }>('/v1/models', newValues, { noTimeout: true })
      .then(async (launchResponse) => {
        // Prevents a false deployment success notification when /v1/models returns model_uid after download cancellation, triggering the success logic below.
        if (isCanceledLaunchRef.current) {
          return;
        }

        const launchedValues = {
          ...newValues,
          model_uid: launchResponse?.model_uid || newValues.model_uid || newValues.model_name,
        };
        saveLaunchConfigHistory(launchedValues);
        setConfigCacheRefreshKey((key) => key + 1);
        let autostartSaved = false;
        if (saveAutostart) {
          try {
            await request.post('/v1/autostart/models', {
              enabled: true,
              priority: 100,
              launch: launchedValues,
            });
            autostartSaved = true;
          } catch (error) {
            console.error(error);
            toast.error(t('launchModel.autostartSaveFailed'));
          }
        }
        setLoading(false);
        stopPolling();
        onOpenChange(false);
        toast.success(
          t(
            autostartSaved
              ? 'launchModel.launchCompletedWithAutostart'
              : 'launchModel.launchCompleted'
          )
        );
        router.push('/running-model');
      })
      .catch(() => {
        stopPolling();
      })
      .finally(() => {
        setLoading(false);
      });
    startPolling();
  };
  const handleClose = () => {
    setLoading(false);
    setCanceling(false);
    setProgress(0);
    setReplicaStatuses([]);
    setSaveAutostart(false);
    stopPolling();
    onOpenChange(false);
    form.resetFields();
  };

  useEffect(() => {
    if (isOpen) {
      fetchModelEngine();
    }
  }, [fetchModelEngine, isOpen]);

  useEffect(() => {
    if (!isOpen) return;

    const latestConfig = getLatestModelConfigHistory(model?.model_name);

    form.resetFields();

    if (latestConfig) {
      form.setFieldsValue(transformFetchToForm(latestConfig.data));
    }
  }, [form, isOpen, model?.model_name]);

  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  const initialValues = {
    model_name: model?.model_name,
    model_type: modelType,
    n_gpu: [ModelType.LLM, ModelType.Image].includes(modelType)
      ? 'auto'
      : gpuAvailable === 0
        ? 'CPU'
        : 'GPU',
    n_gpu_layers: -1,
    replica: 1,
    enable_thinking: true,
    reasoning_content: false,
    enable_virtual_env: 'unset',
    cpu_offload: false,
  };
  return (
    <>
      <Dialog
        open={isOpen}
        onOpenChange={(open) => {
          if (!open) {
            handleClose();
            return;
          }
          onOpenChange(open);
        }}
      >
        <DialogContent className="!max-w-3xl" maskClosable={false}>
          <DialogHeader>
            <div className="flex min-w-0 items-center justify-between gap-3 pr-10">
              <DialogTitle className="min-w-0 truncate">{model?.model_name}</DialogTitle>
              <div className="flex gap-2">
                <ConfigCache
                  form={form}
                  modelName={model?.model_name}
                  refreshKey={configCacheRefreshKey}
                />
                <CommandLine form={form} canCopyCommandLine={isReady} />
              </div>
            </div>
          </DialogHeader>
          <Form
            id={formId}
            form={form}
            onFinish={handleLaunch}
            initialValues={initialValues}
            className="grid grid-cols-2 gap-x-4 gap-y-3 space-y-0"
          >
            <FormField hidden name="model_name" />
            <FormField hidden name="model_type" />
            {renderLaunchFields(currentLaunchFields)}
          </Form>
          <DialogFooter className={cn(loading ? '!flex-col' : '')}>
            {loading && <Progress value={progress} />}
            <div className="flex w-full flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <label className="flex items-center gap-2 text-sm text-muted-foreground">
                <Switch checked={saveAutostart} disabled={loading} onChange={setSaveAutostart} />
                {t('launchModel.saveAutostart')}
              </label>
              <div className="flex items-center justify-end gap-2">
                <TooltipProvider>
                  <Button variant="outline" onClick={handleClose}>
                    {t('common.cancel')}
                  </Button>
                  {loading ? (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          type="button"
                          variant="outline"
                          disabled={canceling}
                          loading={canceling}
                          onClick={handleCancelLaunch}
                          className="border-destructive text-destructive hover:bg-destructive/10 hover:text-destructive"
                        >
                          <Ban />
                          {t('common.stop')}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent side="top" align="end">
                        {renderReplicaStatuses()}
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    <Button type="submit" form={formId}>
                      <Rocket />
                      {t('common.deploy')}
                    </Button>
                  )}
                </TooltipProvider>
              </div>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
