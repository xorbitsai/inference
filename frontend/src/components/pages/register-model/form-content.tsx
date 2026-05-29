'use client';

import { FC, useMemo, useRef } from 'react';
import { Trash2, Plus } from 'lucide-react';
import { FormField } from '@/components/ui/form-field';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { CheckboxGroup, CheckboxGroupChangeEvent } from '@/components/ui/checkbox-group';
import { RadioGroup } from '@/components/ui/radio-group';
import { Switch } from '@/components/ui/switch';
import { AutoComplete } from '@/components/ui/auto-complete';
import { FormList } from '@/components/ui/form-list';
import { useI18n } from '@/contexts/i18n-context';
import { useWatch } from '@/hooks/use-form';
import type { FormInstance } from '@/types/form';
import { ModelType, ModelAbility } from '@/constants';
import {
  MODEL_ABILITY_HAS_CHAT,
  MODEL_ABILITY_OPTIONS_MAP,
  MODEL_FAMILY_OPTIONS_MAP,
} from '@/constants/register';
import type { ModelPrompts, ModelFamily } from '@/types/services';
import Languages from './languages';
import ChatTepmlate, { ChatTemplateMethod } from './chat-tepmlate';
import ModelSpecs from './model-specs';
import ModelControlnet from './model-controlnet';
import { Textarea } from '@/components/ui/textarea';

interface FormContentProps {
  modelType: ModelType;
  form: FormInstance;
  modelPromptsMap: ModelPrompts;
  modelFamilyMap: ModelFamily;
}

const modelAbilityComponentMap: Partial<Record<ModelType, 'radio' | 'checkbox'>> = {
  [ModelType.LLM]: 'checkbox',
  [ModelType.Image]: 'checkbox',
  [ModelType.Audio]: 'radio',
};

const FormContent: FC<FormContentProps> = ({
  modelType,
  form,
  modelPromptsMap,
  modelFamilyMap,
}) => {
  const { t } = useI18n();
  const chatTemplateRef = useRef<ChatTemplateMethod>(null);
  const modelAbilityValue = useWatch('model_ability', form);

  const isLLM = modelType === ModelType.LLM;
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
      if (changedValue === ModelAbility.Chat && !checked) {
        nextValues = nextValues.filter((item) => item === ModelAbility.Generate);
      }
      // Advanced ability auto include chat
      const hasAdvancedAbility = MODEL_ABILITY_HAS_CHAT.some((item) => nextValues.includes(item));

      if (hasAdvancedAbility && !nextValues.includes(ModelAbility.Chat)) {
        nextValues.push(ModelAbility.Chat);
      }
      // vision and tools are mutually exclusive
      if (changedValue === ModelAbility.Vision && checked) {
        nextValues = nextValues.filter((item) => item !== ModelAbility.Tools);
      }
      if (changedValue === ModelAbility.Tools && checked) {
        nextValues = nextValues.filter((item) => item !== ModelAbility.Vision);
      }
      // clear chat-template status
      chatTemplateRef.current?.resetStatus?.();

      form.setFieldsValue({
        model_ability: [...new Set(nextValues)],
        model_family: undefined,
        chat_template: undefined,
        stop_token_ids: [''],
        stop: [''],
        reasoning_start_tag: undefined,
        reasoning_end_tag: undefined,
        tool_parser: undefined,
      });
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
  const { options: modelFamilyOptions, editable: isEditableFamily } = useMemo(() => {
    const modelAbility = Array.isArray(modelAbilityValue) ? modelAbilityValue : [];
    const abilityMap: Record<string, { editable: boolean; source: string[] }> = {
      reasoning: { editable: false, source: modelFamilyMap?.reasoning },
      audio: { editable: false, source: modelFamilyMap?.audio },
      omni: { editable: false, source: modelFamilyMap?.omni },
      hybrid: { editable: false, source: modelFamilyMap?.hybrid },
      vision: { editable: false, source: modelFamilyMap?.vision },
      tools: { editable: false, source: modelFamilyMap?.tools },
      chat: { editable: true, source: modelFamilyMap?.chat },
      generate: { editable: true, source: modelFamilyMap?.generate },
    };

    const matchedAbilities = Object.keys(abilityMap).filter((key) => modelAbility.includes(key));

    const matchedNonEditable = matchedAbilities.filter((key) =>
      MODEL_ABILITY_HAS_CHAT.includes(key as ModelAbility)
    );
    // Take the "intersection" of multiple ability family sources, then lock it (non-editable)
    if (matchedNonEditable.length > 1) {
      const baseSource = abilityMap[matchedNonEditable[0]]?.source || [];

      let intersection = new Set(baseSource);

      for (let i = 1; i < matchedNonEditable.length; i++) {
        const currentSource = new Set(abilityMap[matchedNonEditable[i]]?.source || []);

        intersection = new Set([...intersection].filter((item) => currentSource.has(item)));
      }

      return {
        editable: false,
        options: Array.from(intersection).map((item) => ({
          value: item,
          label: item,
        })),
      };
    }
    const matched = matchedAbilities[0];

    if (matched) {
      const { editable, source } = abilityMap[matched];

      return {
        editable,
        options: (source || []).map((item) => ({
          value: item,
          label: item,
        })),
      };
    }

    return {
      editable: true,
      options: [],
    };
  }, [modelFamilyMap, modelAbilityValue]);

  const handleModelFamily = (value?: string) => {
    const promptConfig = value ? modelPromptsMap?.[value] : undefined;

    form.setFieldsValue({
      chat_template: promptConfig?.chat_template,
      stop_token_ids: promptConfig?.stop_token_ids ?? [''],
      stop: promptConfig?.stop ?? [''],
      reasoning_start_tag: promptConfig?.reasoning_start_tag,
      reasoning_end_tag: promptConfig?.reasoning_end_tag,
      tool_parser: promptConfig?.tool_parser,
    });
    // clear chat-template status
    chatTemplateRef.current?.resetStatus?.();
  };
  const renderModelFamilyField = () => {
    if (isLLM) {
      return (
        <FormField
          name="model_family"
          label={t('registerModel.modelFamily')}
          rules={[{ required: true }]}
          extra={t('registerModel.chooseBuiltInOrCustomModel')}
        >
          <AutoComplete
            onChange={handleModelFamily}
            allowCustomValue={isEditableFamily}
            options={modelFamilyOptions}
          />
        </FormField>
      );
    }
    return (
      <FormField
        name="model_family"
        label={t('registerModel.modelFamily')}
        rules={[{ required: true }]}
      >
        <RadioGroup options={MODEL_FAMILY_OPTIONS_MAP[modelType] || []} />
      </FormField>
    );
  };
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
    modelUri: (
      <FormField
        name="model_uri"
        label={t('registerModel.modelPath')}
        rules={[{ required: true }]}
        extra={t('registerModel.provideModelDirectoryPath')}
      >
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
        normalize={(v) => (v === '' ? undefined : Number(v))}
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
    modelFamily: renderModelFamilyField(),
    chatTemplate: (
      <FormField
        label={t('registerModel.chatTemplate')}
        name="chat_template"
        rules={[{ required: true }]}
        normalize={(value) => (value ? value.replace(/\\n/g, '\n') : '')}
      >
        <ChatTepmlate />
      </FormField>
    ),
    stopTokenIds: (
      <FormList
        name="stop_token_ids"
        label={t('registerModel.stopTokenIds')}
        layout="horizontal"
        renderAction={({ add }) => (
          <Button size="sm" type="button" onClick={() => add()}>
            <Plus />
            {t('common.add')}
          </Button>
        )}
      >
        {({ fields, remove }) => (
          <>
            {fields.map((field) => (
              <div className="flex gap-2" key={field.name}>
                <FormField
                  className="flex-1"
                  key={field.key}
                  name={['stop_token_ids', field.name]}
                  normalize={(v) => (v === '' ? undefined : Number(v))}
                  rules={[
                    { required: true },
                    { pattern: /^-?[0-9]\d*$/, message: t('registerModel.enterInteger') },
                  ]}
                  placeholder={t('registerModel.stopControlForChatModels')}
                >
                  <Input type="number" />
                </FormField>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 group hover:bg-destructive/10 rounded-full"
                  onClick={() => remove(field.name)}
                >
                  <Trash2 className="text-muted-foreground group-hover:text-destructive" />
                </Button>
              </div>
            ))}
          </>
        )}
      </FormList>
    ),
    stop: (
      <FormList
        name="stop"
        label={t('registerModel.stop')}
        layout="horizontal"
        renderAction={({ add }) => (
          <Button size="sm" type="button" onClick={() => add()}>
            <Plus />
            {t('common.add')}
          </Button>
        )}
      >
        {({ fields, remove }) => (
          <>
            {fields.map((field) => (
              <div className="flex gap-2" key={field.name}>
                <FormField
                  className="flex-1"
                  key={field.key}
                  name={['stop', field.name]}
                  rules={[{ required: true }]}
                  placeholder={t('registerModel.stopControlStringForChatModels')}
                >
                  <Input />
                </FormField>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 group hover:bg-destructive/10 rounded-full"
                  onClick={() => remove(field.name)}
                >
                  <Trash2 className="text-muted-foreground group-hover:text-destructive" />
                </Button>
              </div>
            ))}
          </>
        )}
      </FormList>
    ),
    reasoningStartTag: (
      <FormField
        name="reasoning_start_tag"
        label={t('registerModel.reasoningStartTag')}
        rules={[{ required: true }]}
      >
        <Input />
      </FormField>
    ),
    reasoningEndTag: (
      <FormField
        name="reasoning_end_tag"
        label={t('registerModel.reasoningEndTag')}
        rules={[{ required: true }]}
      >
        <Input />
      </FormField>
    ),
    toolParser: (
      <FormField
        name="tool_parser"
        label={t('registerModel.toolParser')}
        rules={[{ required: true }]}
      >
        <Input />
      </FormField>
    ),
    dimensions: (
      <FormField
        name="dimensions"
        label={t('registerModel.dimensions')}
        rules={[
          { required: true },
          { pattern: /^[1-9]\d*$/, message: t('registerModel.enterIntegerGreaterThanZero') },
        ]}
        normalize={(v) => (v === '' ? undefined : Number(v))}
      >
        <Input type="number" />
      </FormField>
    ),
    maxTokens: (
      <FormField
        name="max_tokens"
        label={t('registerModel.maxTokens')}
        rules={[
          { required: true },
          { pattern: /^[1-9]\d*$/, message: t('registerModel.enterIntegerGreaterThanZero') },
        ]}
        normalize={(v) => (v === '' ? undefined : Number(v))}
      >
        <Input type="number" />
      </FormField>
    ),
    modelSpecs: <ModelSpecs modelType={modelType} form={form} />,
    multilingual: (
      <FormField
        name="multilingual"
        label={t('registerModel.multilingual')}
        rules={[{ required: true }]}
        valuePropName="checked"
        layout="vertical"
      >
        <Switch />
      </FormField>
    ),
    controlnet: <ModelControlnet />,
    launcher: (
      <FormField
        name="launcher"
        label={t('registerModel.launcher')}
        rules={[{ required: true }]}
        extra={t('registerModel.provideModelLauncher')}
      >
        <Input />
      </FormField>
    ),
    launcherArgs: (
      <FormField
        name="launcher_args"
        label={t('registerModel.launcherArguments')}
        extra={t('registerModel.jsonArgumentsForLauncher')}
      >
        <Textarea />
      </FormField>
    ),
  };
  const modelTypeFields: Partial<Record<ModelType, (keyof typeof fields)[]>> = {
    [ModelType.LLM]: [
      'modelName',
      'modelDescription',
      'contextLength',
      'languages',
      'modelAbility',
      'modelFamily',
      ...(isLLM && (modelAbilityValue || []).includes(ModelAbility.Chat)
        ? (['chatTemplate', 'stopTokenIds', 'stop'] as (keyof typeof fields)[])
        : []),
      ...((modelAbilityValue || []).includes(ModelAbility.Reasoning)
        ? (['reasoningStartTag', 'reasoningEndTag'] as (keyof typeof fields)[])
        : []),
      ...((modelAbilityValue || []).includes(ModelAbility.Tools)
        ? (['toolParser'] as (keyof typeof fields)[])
        : []),
      'modelSpecs',
    ],

    [ModelType.Embedding]: ['modelName', 'dimensions', 'maxTokens', 'languages', 'modelSpecs'],

    [ModelType.Rerank]: ['modelName', 'maxTokens', 'languages', 'modelSpecs'],

    [ModelType.Image]: ['modelName', 'modelUri', 'modelAbility', 'modelFamily', 'controlnet'],

    [ModelType.Audio]: ['modelName', 'modelUri', 'multilingual', 'modelAbility', 'modelFamily'],

    [ModelType.Flexible]: ['modelName', 'modelDescription', 'modelUri', 'launcher', 'launcherArgs'],
  };
  return (
    <div className="space-y-3">
      {(modelTypeFields[modelType] || []).map((fieldKey) => (
        <div key={fieldKey}>{fields[fieldKey]}</div>
      ))}
      <FormList
        name={['virtualenv', 'packages']}
        label={t('registerModel.packages')}
        layout="horizontal"
        renderAction={({ add }) => (
          <Button size="sm" type="button" onClick={() => add('')}>
            <Plus />
            {t('common.add')}
          </Button>
        )}
      >
        {({ fields, remove }) => (
          <>
            {fields.map((field) => (
              <div className="flex gap-2" key={field.name}>
                <FormField
                  className="flex-1"
                  key={field.key}
                  name={['virtualenv', 'packages', field.name]}
                >
                  <Input />
                </FormField>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 group hover:bg-destructive/10 rounded-full"
                  onClick={() => remove(field.name)}
                >
                  <Trash2 className="text-muted-foreground group-hover:text-destructive" />
                </Button>
              </div>
            ))}
          </>
        )}
      </FormList>
    </div>
  );
};
export default FormContent;
