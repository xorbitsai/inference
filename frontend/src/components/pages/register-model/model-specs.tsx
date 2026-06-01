'use client';

import { FC } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { FormField } from '@/components/ui/form-field';
import { FormList } from '@/components/ui/form-list';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { RadioGroup } from '@/components/ui/radio-group';
import { ModelType } from '@/constants';
import {
  MODEL_FORMAT_OPTIONS_MAP,
  ModelFormat,
  REGISTER_MODEL_INIT_DATA,
} from '@/constants/register';
import { useI18n } from '@/contexts/i18n-context';
import { useWatch } from '@/hooks/use-form';
import { FormInstance } from '@/types/form';

interface ModelSpecsProps {
  modelType: ModelType;
  form: FormInstance;
}
const ModelSpecs: FC<ModelSpecsProps> = ({ modelType, form }) => {
  const { t } = useI18n();
  const options = MODEL_FORMAT_OPTIONS_MAP[modelType] || [];
  const modelSpecsValue = useWatch('model_specs', form);
  const isLLM = modelType === ModelType.LLM;
  // @ts-ignore
  const defaultItem = REGISTER_MODEL_INIT_DATA?.[modelType]?.model_specs?.[0] || {};
  return (
    <FormList
      name="model_specs"
      label={t('registerModel.modelSpecs')}
      layout="horizontal"
      renderAction={({ add }) => (
        <Button size="sm" type="button" onClick={() => add(defaultItem)}>
          <Plus />
          {t('common.add')}
        </Button>
      )}
    >
      {({ fields, remove }) => (
        <>
          {fields.map((field, index) => {
            const modelFormat = modelSpecsValue?.[index]?.model_format;
            return (
              <div
                className="border border-border rounded-md p-4 flex items-center gap-2"
                key={field.name}
              >
                <div className="flex-1 space-y-3">
                  <FormField
                    label={t('registerModel.modelFormat')}
                    name={['model_specs', field.name, 'model_format']}
                    rules={[{ required: true }]}
                  >
                    <RadioGroup options={options} />
                  </FormField>
                  <FormField
                    name={['model_specs', field.name, 'model_uri']}
                    label={t('registerModel.modelPath')}
                    rules={[
                      { required: true },
                      // GGUF model path must start with "/" and contain at least two "/" separators.
                      ...(modelFormat && modelFormat === ModelFormat.GGUF
                        ? [
                            {
                              pattern: /^\/[^/]+(\/[^/]*)+$/,
                              message: t('common.patternError'),
                            },
                          ]
                        : []),
                    ]}
                    extra={t('registerModel.provideModelDirectoryOrFilePath')}
                  >
                    <Input />
                  </FormField>
                  {isLLM && (
                    <FormField
                      name={['model_specs', field.name, 'model_size_in_billions']}
                      label={t('registerModel.modelSizeBillions')}
                      rules={[
                        { required: true },
                        {
                          pattern: /^\d+(\.\d+)?$/,
                          message: t('registerModel.enterNumberGreaterThanZero'),
                        },
                      ]}
                    >
                      <Input type="number" />
                    </FormField>
                  )}

                  {modelFormat && modelFormat !== ModelFormat.PyTorch && (
                    <FormField
                      name={['model_specs', field.name, 'quantization']}
                      label={t('registerModel.quantization')}
                      rules={[{ required: modelFormat !== ModelFormat.GGUF }]}
                      extra={t('registerModel.carefulQuantizationForModelRegistration')}
                    >
                      <Input />
                    </FormField>
                  )}
                </div>
                {fields.length !== 1 && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="shrink-0 group hover:bg-destructive/10 rounded-full"
                    onClick={() => remove(field.name)}
                  >
                    <Trash2 className="text-muted-foreground group-hover:text-destructive" />
                  </Button>
                )}
              </div>
            );
          })}
        </>
      )}
    </FormList>
  );
};
export default ModelSpecs;
