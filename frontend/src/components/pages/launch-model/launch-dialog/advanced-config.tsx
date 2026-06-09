'use client';

import { FC, ReactNode } from 'react';
import { Settings } from 'lucide-react';

import { CollapsiblePanel } from '@/components/ui/collapsible';
import { FormField } from '@/components/ui/form-field';
import { RadioGroup } from '@/components/ui/radio-group';
import {
  KWARGS_OPTIONS_FOR_ENGINES,
  QUANTIZATION_OPTIONS,
  VIRTUAL_ENV_OPTIONS,
} from '@/constants/launch';
import { useI18n } from '@/contexts/i18n-context';
import { useWatch } from '@/hooks/use-form';
import type { FormInstance } from '@/types/form';
import type { RequestModelType } from '../types';
import { toOptionValue } from '../utils';
import CommonFormList from './common-form-list';
import { ModelType } from '@/constants';

interface AdvancedConfigProps {
  form: FormInstance;
  modelType: RequestModelType;
}

interface ConfigSectionProps {
  title: ReactNode;
  children: ReactNode;
}

function ConfigSection({ title, children }: ConfigSectionProps) {
  return (
    <section className="space-y-2">
      <h4 className="px-1 text-xs font-semibold tracking-normal text-muted-foreground">{title}</h4>
      <div className="divide-y divide-border/60 rounded-md border border-border/70 bg-background">
        {children}
      </div>
    </section>
  );
}

const AdvancedConfig: FC<AdvancedConfigProps> = ({ form, modelType }) => {
  const { t } = useI18n();
  const modelEngineValue = toOptionValue(useWatch('model_engine', form));

  const kwargsOptionsForEngine = modelEngineValue
    ? KWARGS_OPTIONS_FOR_ENGINES[modelEngineValue.toLowerCase()]
    : undefined;
  const showLora = [ModelType.LLM, ModelType.Image, ModelType.Video].includes(modelType);
  const showLoraKwargs = [ModelType.Image, ModelType.Video].includes(modelType);
  return (
    <CollapsiblePanel
      title={t('launchModel.advancedConfiguration')}
      icon={<Settings className="size-4" />}
      className="rounded-lg"
      contentClassName="space-y-4 bg-muted/10 p-4"
    >
      {showLora && (
        <ConfigSection title="Lora">
          <div className="p-2">
            <CommonFormList
              name={['peft_model_config', 'lora_list']}
              label={t('launchModel.loraModelConfig')}
              childFirstKey="lora_name"
              childSecondKey="local_path"
            />
          </div>
          {showLoraKwargs && (
            <>
              <div className="p-2">
                <CommonFormList
                  name={['peft_model_config', 'image_lora_load_kwargs']}
                  label={t('launchModel.loraLoadKwargsForImageModel')}
                />
              </div>
              <div className="p-2">
                <CommonFormList
                  name={['peft_model_config', 'image_lora_fuse_kwargs']}
                  label={t('launchModel.loraFuseKwargsForImageModel')}
                />
              </div>
            </>
          )}
        </ConfigSection>
      )}

      <ConfigSection title={t('launchModel.runtimeEnvironment')}>
        <div className="px-2 py-3.5 flex items-center justify-between ">
          <span className="text-sm font-medium">{t('launchModel.modelVirtualEnv')}</span>
          <FormField name="enable_virtual_env">
            <RadioGroup options={VIRTUAL_ENV_OPTIONS} />
          </FormField>
        </div>
        <div className="p-2">
          <CommonFormList
            name="virtual_env_packages"
            label={t('launchModel.virtualEnvPackage')}
            onlyValue
          />
        </div>

        <div className="p-2">
          <CommonFormList name="envs" label={t('launchModel.envVariable')} />
        </div>
      </ConfigSection>

      <ConfigSection title={t('launchModel.engineParameters')}>
        {modelEngineValue === 'Transformers' && (
          <div className="p-2">
            <CommonFormList
              name="quantization_config"
              label={t('launchModel.engineQuantizationParameters')}
              keyOptions={QUANTIZATION_OPTIONS}
            />
          </div>
        )}

        <div className="p-2">
          <CommonFormList
            name="kwargs"
            label={`${t('launchModel.engineAdditionalParameters')}${
              modelEngineValue ? ': ' + modelEngineValue : ''
            }`}
            keyOptions={kwargsOptionsForEngine}
          />
        </div>
      </ConfigSection>
    </CollapsiblePanel>
  );
};

export default AdvancedConfig;
