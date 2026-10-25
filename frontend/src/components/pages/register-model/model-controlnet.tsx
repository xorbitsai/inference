'use client';
import { FC } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { FormField } from '@/components/ui/form-field';
import { FormList } from '@/components/ui/form-list';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { RadioGroup } from '@/components/ui/radio-group';
import { ControlnetModelFamily, CONTROLNET_MODEL_FAMILY_OPTIONS } from '@/constants/register';
import { useI18n } from '@/contexts/i18n-context';

const ModelControlnet: FC = () => {
  const { t } = useI18n();

  return (
    <FormList
      name="controlnet"
      label={t('registerModel.controlnet')}
      layout="horizontal"
      renderAction={({ add }) => (
        <Button
          size="sm"
          type="button"
          onClick={() =>
            add({
              model_name: 'custom-controlnet',
              model_uri: '/path/to/controlnet-model',
              model_family: ControlnetModelFamily.Controlnet,
            })
          }
        >
          <Plus />
          {t('common.add')}
        </Button>
      )}
    >
      {({ fields, remove }) => (
        <>
          {fields.map((field) => {
            return (
              <div className="border border-border rounded-md p-4 flex items-center gap-2" key={field.name}>
                <div className="flex-1 space-y-3">
                  <FormField
                    label={t('registerModel.modelName')}
                    name={['controlnet', field.name, 'model_name']}
                    rules={[{ required: true }]}
                  >
                    <Input />
                  </FormField>
                  <FormField
                    name={['controlnet', field.name, 'model_uri']}
                    label={t('registerModel.modelPath')}
                    rules={[{ required: true }]}
                  >
                    <Input />
                  </FormField>

                  <FormField
                    name={['controlnet', field.name, 'model_family']}
                    label={t('registerModel.modelFormat')}
                    rules={[{ required: true }]}
                  >
                    <RadioGroup options={CONTROLNET_MODEL_FAMILY_OPTIONS} />
                  </FormField>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 group hover:bg-destructive/10 rounded-full"
                  onClick={() => remove(field.name)}
                >
                  <Trash2 className="text-muted-foreground group-hover:text-destructive" />
                </Button>
              </div>
            );
          })}
        </>
      )}
    </FormList>
  );
};
export default ModelControlnet;
