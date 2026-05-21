'use client';

import { FC, useMemo } from 'react';
import { FormField } from '@/components/ui/form-field';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { CheckboxGroup, CheckboxGroupChangeEvent } from '@/components/ui/checkbox-group';
import { MultiSelect } from '@/components/ui/multi-select';
import { Select } from '@/components/ui/select';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Switch } from '@/components/ui/switch';
import { useI18n } from '@/contexts/i18n-context';
import { useWatch } from '@/hooks/use-form';
import type { FormInstance } from '@/hooks/use-form';
import {
  ModelType,
  LANGUAGES_CHECKBOX_OPTIONS,
  MODEL_ABILITY_HAS_CHAT,
  MODEL_ABILITY_OPTIONS_MAP,
  LANGUAGES_OPTIONS,
} from '@/constants';
import type { ModelFamily } from '@/types/services';
import Languages from './languages';

interface FormContentProps {
  modelType: ModelType;
  form: FormInstance;
  modelFamilyMap: ModelFamily;
}

const modelAbilityComponentMap: Partial<Record<ModelType, 'radio' | 'checkbox'>> = {
  [ModelType.Audio]: 'radio',
  [ModelType.LLM]: 'checkbox',
  [ModelType.Image]: 'checkbox',
};

const FormContent: FC<FormContentProps> = ({ modelType, form, modelFamilyMap }) => {
  const { t } = useI18n();
  const modelAbilityValue = useWatch('model_ability', form);

  const isLLM = modelType === ModelType.LLM;
  const isAudio = modelType === ModelType.Audio;
  const showModelDescription = [ModelType.LLM, ModelType.Flexible].includes(modelType);
  const showLanguages = [ModelType.LLM, ModelType.Embedding, ModelType.Rerank].includes(modelType);
  const showModelAbility = [ModelType.LLM, ModelType.Image, ModelType.Audio].includes(modelType);

  const renderModelAbilityField = () => {
    const options = MODEL_ABILITY_OPTIONS_MAP[modelType] || [];
    const componentType = modelAbilityComponentMap[modelType];
    const handleChange = (
      values: string[],
      { changedValue, checked }: CheckboxGroupChangeEvent
    ) => {
      if (!isLLM) return;

      let nextValues = [...values];
      // Clear all chat data except for 'generate'
      if (changedValue === 'chat' && !checked) {
        nextValues = nextValues.filter((item) => item === 'generate');
      }
      // Advanced ability auto include chat
      const hasAdvancedAbility = MODEL_ABILITY_HAS_CHAT.some((item) => nextValues.includes(item));

      if (hasAdvancedAbility && !nextValues.includes('chat')) {
        nextValues.push('chat');
      }
      // vision and tools are mutually exclusive
      if (changedValue === 'vision' && checked) {
        nextValues = nextValues.filter((item) => item !== 'tools');
      }
      if (changedValue === 'tools' && checked) {
        nextValues = nextValues.filter((item) => item !== 'vision');
      }
      form.setFieldValue('model_ability', [...new Set(nextValues)]);
    };
    return (
      <FormField
        name="model_ability"
        label={t('registerModel.modelAbilities')}
        rules={[{ required: true }]}
      >
        {componentType === 'radio' ? (
          <RadioGroup options={options} />
        ) : (
          <CheckboxGroup options={options} onChange={handleChange} />
        )}
      </FormField>
    );
  };
  const modelFamilyOptions = useMemo(() => {
    const modelAbility = Array.isArray(modelAbilityValue) ? modelAbilityValue : [];

    const matchedAbilities = Object.keys(modelFamilyMap).filter((key) =>
      modelAbility.includes(key)
    );

    const matchedNonEditable = matchedAbilities.filter((key) =>
      MODEL_ABILITY_HAS_CHAT.includes(key)
    );

    if (matchedNonEditable.length > 1) {
      const baseSource = modelFamilyMap[matchedNonEditable[0]] || [];
      let intersection = new Set(baseSource);

      for (let i = 1; i < matchedNonEditable.length; i++) {
        const currentSource = new Set(modelFamilyMap[matchedNonEditable[i]] || []);
        intersection = new Set([...intersection].filter((item) => currentSource.has(item)));
      }
      return Array.from(intersection).map((item) => ({
        value: item,
        label: item,
      }));
    }

    const matched = matchedAbilities[0];
    return matched
      ? (modelFamilyMap[matched] || []).map((item) => ({
          value: item,
          label: item,
        }))
      : [];
  }, [modelFamilyMap, modelAbilityValue]);
  console.log(modelFamilyOptions, modelFamilyMap, 'modelFamilyMap');
  const fields = {
    modelName: (
      <FormField
        name="model_name"
        label={t('registerModel.modelName')}
        extra={t('registerModel.alphanumericWithHyphensUnderscores')}
        rules={[{ required: true }]}
      >
        <Input />
      </FormField>
    ),
    modelDescription: (
      <FormField name="model_description" label={t('registerModel.modelDescription')}>
        <Input />
      </FormField>
    ),
    contextLength: (
      <FormField
        name="context_length"
        label={t('registerModel.contextLength')}
        rules={[
          { required: true },
          { pattern: /^[1-9]\d*$/, message: t('registerModel.enterIntegerGreaterThanZero') },
        ]}
      >
        <Input type="number" />
      </FormField>
    ),
    languages: (
      <FormField
        name={isLLM ? 'model_lang' : 'language'}
        label={t('registerModel.languages')}
        rules={[{ required: true }]}
      >
        <Languages />
      </FormField>
    ),
    modelAbility: renderModelAbilityField(),
    modelFamily: (
      <FormField
        name="model_family"
        label={t('registerModel.modelFamily')}
        rules={[{ required: true }]}
      >
        <Select showSearch options={modelFamilyOptions} />
      </FormField>
    ),
  };
  const modelTypeFields: Record<ModelType, (keyof typeof fields)[]> = {
    [ModelType.LLM]: [
      'modelName',
      'modelDescription',
      'contextLength',
      'languages',
      'modelAbility',
      'modelFamily',
    ],

    [ModelType.Embedding]: ['modelName', 'languages'],

    [ModelType.Rerank]: ['modelName', 'languages'],

    [ModelType.Image]: ['modelName', 'modelAbility'],

    [ModelType.Audio]: ['modelName', 'modelAbility'],

    [ModelType.Flexible]: ['modelName', 'modelDescription'],
  };
  return (
    <div className="space-y-3">
      {modelTypeFields[modelType].map((fieldKey) => (
        <div key={fieldKey}>{fields[fieldKey]}</div>
      ))}
      <FormField
        name="switch"
        label={t('registerModel.modelAbilities')}
        rules={[{ required: true }]}
        valuePropName="checked"
        layout="vertical"
      >
        <Switch />
      </FormField>
    </div>
  );
};
export default FormContent;
