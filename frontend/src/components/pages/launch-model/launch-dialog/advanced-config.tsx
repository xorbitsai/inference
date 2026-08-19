'use client';

import { FC, ReactNode } from 'react';
import { Settings } from 'lucide-react';

import { CollapsiblePanel } from '@/components/ui/collapsible';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { RadioGroup } from '@/components/ui/radio-group';
import { Select } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { InfoTooltip } from '@/components/ui/tooltip';
import {
  GEMMA_4_SPECULATIVE_TOKENS_BY_SIZE,
  KWARGS_OPTIONS_FOR_ENGINES,
  QUANTIZATION_OPTIONS,
  SPECULATIVE_TOKENS_DEFAULT_BY_ENGINE,
  VIRTUAL_ENV_OPTIONS,
} from '@/constants/launch';
import { useI18n } from '@/contexts/i18n-context';
import { useWatch } from '@/hooks/use-form';
import type { FormInstance } from '@/types/form';
import type { Option } from '@/types/common';
import type { RequestModelType } from '../types';
import { toOptionValue } from '../utils';
import CommonFormList from './common-form-list';
import { ModelType } from '@/constants';

interface AdvancedConfigProps {
  form: FormInstance;
  modelType: RequestModelType;
  modelName?: string;
  /** whether the selected spec ships a drafter for speculative decoding */
  hasDrafter?: boolean;
  /** available drafter conversions, empty when there is nothing to pick */
  draftQuantizationOptions?: Option<string>[];
}

interface ConfigSectionProps {
  title: ReactNode;
  children: ReactNode;
}

function FieldLabel({ label, tip }: { label: string; tip: string }) {
  return (
    <span className="shrink-0 flex items-center gap-1 text-sm font-medium">
      {label}
      <InfoTooltip content={tip} />
    </span>
  );
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

const AdvancedConfig: FC<AdvancedConfigProps> = ({
  form,
  modelType,
  modelName,
  hasDrafter = false,
  draftQuantizationOptions = [],
}) => {
  const { t } = useI18n();
  const modelEngineValue = toOptionValue(useWatch('model_engine', form));
  const modelSizeValue = toOptionValue(useWatch('model_size_in_billions', form));
  const enableMtpValue = useWatch('enable_mtp', form);

  const engineKey = modelEngineValue.toLowerCase();
  const kwargsOptionsForEngine = engineKey ? KWARGS_OPTIONS_FOR_ENGINES[engineKey] : undefined;
  const speculativeTokensDefault =
    ['vllm', 'sglang', 'llama.cpp'].includes(engineKey) && modelName === 'gemma-4'
      ? GEMMA_4_SPECULATIVE_TOKENS_BY_SIZE[modelSizeValue]
      : SPECULATIVE_TOKENS_DEFAULT_BY_ENGINE[engineKey];
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
          <CommonFormList
            name="virtual_env_find_links"
            label={t('launchModel.virtualEnvFindLinks')}
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

      {hasDrafter && (
        <ConfigSection title={t('launchModel.speculativeDecoding')}>
          <div className="px-2 py-3.5 flex items-center justify-between">
            <FieldLabel label={t('launchModel.enableMtp')} tip={t('launchModel.enableMtpTip')} />
            <FormField name="enable_mtp" valuePropName="checked">
              <Switch />
            </FormField>
          </div>

          {enableMtpValue && draftQuantizationOptions.length > 0 && (
            <div className="px-2 py-3.5 flex items-center justify-between gap-4">
              <FieldLabel
                label={t('launchModel.draftModelQuantization')}
                tip={t('launchModel.draftModelQuantizationTip')}
              />
              <FormField
                className="w-56"
                name="draft_quantization"
                placeholder={t('launchModel.draftModelQuantizationPlaceholder')}
              >
                <Select options={draftQuantizationOptions} />
              </FormField>
            </div>
          )}

          {enableMtpValue && (
            <div className="px-2 py-3.5 flex items-center justify-between gap-4">
              <FieldLabel
                label={t('launchModel.numSpeculativeTokens')}
                tip={t('launchModel.numSpeculativeTokensTip')}
              />
              <FormField
                className="w-56"
                name="num_speculative_tokens"
                placeholder={
                  speculativeTokensDefault
                    ? t('launchModel.numSpeculativeTokensPlaceholderValue', {
                        value: speculativeTokensDefault,
                      })
                    : t('launchModel.numSpeculativeTokensPlaceholder')
                }
                rules={[
                  {
                    pattern: /^[1-9]\d*$/,
                    message: t('launchModel.enterIntegerGreaterThanZero'),
                  },
                ]}
                normalize={(v) => (v === '' ? undefined : Number(v))}
              >
                <Input type="number" min={1} />
              </FormField>
            </div>
          )}
        </ConfigSection>
      )}
    </CollapsiblePanel>
  );
};

export default AdvancedConfig;
