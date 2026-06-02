'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Rocket } from 'lucide-react';
import request from '@/lib/request';
import { ModelType } from '@/constants';
import { useI18n } from '@/contexts/i18n-context';
import { useForm, useWatch } from '@/hooks/use-form';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Form } from '@/components/ui/form';
import { FormField } from '@/components/ui/form-field';
import { Select } from '@/components/ui/select';
import { isCachedSpec } from './utils';
import type { CatalogModel, RequestModelType } from './types';

import type { ModelEngine, ModelEngineItem } from '@/types/services';
import type { FormValues } from '@/types/form';

interface LaunchDialogProps {
  model?: CatalogModel;
  modelType: RequestModelType;
  onOpenChange: (open: boolean) => void;
}

export default function LaunchDialog({ model, modelType, onOpenChange }: LaunchDialogProps) {
  const isOpen = Boolean(model);
  const [form] = useForm();
  const { t } = useI18n();
  const [loading, setLoading] = useState(false);
  const [modelEngineMap, setModelEngineMap] = useState<ModelEngine>({});
  const modelEngineValue = useWatch('model_engine', form);

  const fetchModelEngine = useCallback(async () => {
    if (!model?.model_name || [ModelType.Audio, ModelType.Video].includes(modelType)) {
      setModelEngineMap({});
      return;
    }

    const url =
      modelType === ModelType.LLM
        ? `/v1/engines/${model.model_name}`
        : `/v1/engines/${modelType}/${model.model_name}`;
    const res = await request.get(url);

    setModelEngineMap(res || {});
  }, [model?.model_name, modelType]);

  const modelEngineOptions = useMemo(() => {
    const cachedFormats = new Set(
      (model?.modelSpecs || [])
        .filter((spec) => isCachedSpec(spec))
        .map((spec) => spec.model_format)
    );

    return Object.entries(modelEngineMap).map(([key, engineData]) => {
      if (typeof engineData === 'string') {
        return {
          label: `${key} (${engineData})`,
          value: key,
          disabled: true,
        };
      }
      const cached = engineData.some((item) => cachedFormats.has(item.model_format));

      return {
        label: key,
        value: key,
        suffix: cached ? t('launchModel.cached') : undefined,
      };
    });
  }, [model?.modelSpecs, modelEngineMap]);
  // console.log(modelEngineValue, 'modelEngineValue');

  const fields = {
    modelEngine: (
      <FormField
        name="model_engine"
        label={t('launchModel.modelEngine')}
        rules={[{ required: true }]}
      >
        <Select options={modelEngineOptions}></Select>
      </FormField>
    ),
    modelFormat: (
      <FormField
        name="model_format"
        label={t('launchModel.modelFormat')}
        rules={[{ required: true }]}
      >
        <Select options={[]}></Select>
      </FormField>
    ),
  };
  const modelTypeFields: Partial<Record<ModelType, (keyof typeof fields)[]>> = {
    [ModelType.LLM]: [
      'modelEngine',
      'modelFormat',
      //  'model_size_in_billions', 'quantization'
    ],
  };
  const handleLaunch = async (values: FormValues) => {
    console.log(values);
    // if (!model) return;

    // setLoading(true);

    // try {
    //   await request.post('/v1/models', {
    //     model_type: modelType,
    //     model_name: model.model_name,
    //     model_uid: form.modelUid || model.model_name,
    //     n_gpu: form.nGpu === '' ? undefined : Number(form.nGpu),
    //     replica: form.replica === '' ? undefined : Number(form.replica),
    //   });
    //   toast.success('Model launch request submitted');
    //   onOpenChange(false);
    // } finally {
    //   setLoading(false);
    // }
  };

  useEffect(() => {
    if (isOpen) {
      fetchModelEngine();
    }
  }, [fetchModelEngine, isOpen]);
  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <Form form={form} onFinish={handleLaunch}>
        <DialogContent className="!max-w-3xl min-h-[500px]" maskClosable={false}>
          <DialogHeader>
            <DialogTitle>{model?.model_name}</DialogTitle>
          </DialogHeader>

          {(modelTypeFields[modelType] || []).map((fieldKey) => (
            <div key={fieldKey}>{fields[fieldKey]}</div>
          ))}
          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={handleLaunch} loading={loading} type="submit">
              <Rocket />
              Launch
            </Button>
          </DialogFooter>
        </DialogContent>
      </Form>
    </Dialog>
  );
}
