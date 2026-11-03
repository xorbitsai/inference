'use client';

import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import type { ReasoningEffortCapability, RunningModelDetail } from '@/types/services';
import type { ChatSettings } from '../types';

export function supportsEffort(value: unknown, capability?: ReasoningEffortCapability | null) {
  if (!capability) return false;
  if (typeof value === 'string') return capability.values.includes(value);
  const range = capability.integer_range;
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    !!range &&
    value >= range[0] &&
    value <= range[1]
  );
}

export function supportedChatSettings(
  settings: ChatSettings,
  model: RunningModelDetail
): ChatSettings {
  const { reasoning_effort, thinking_budget, ...base } = settings;
  const result: ChatSettings = base;
  if (supportsEffort(reasoning_effort, model.reasoning_effort)) {
    result.reasoning_effort = reasoning_effort;
  }
  if (
    model.supports_thinking_budget === true &&
    typeof thinking_budget === 'number' &&
    Number.isInteger(thinking_budget) &&
    thinking_budget > 0 &&
    thinking_budget < result.max_tokens
  ) {
    result.thinking_budget = thinking_budget;
  }
  return result;
}

export function ReasoningEffortInput({
  value,
  onChange,
  capability,
  error,
  disabled,
}: {
  value?: string | number;
  onChange?: (value: string | number | undefined) => void;
  capability: ReasoningEffortCapability;
  error?: boolean;
  disabled?: boolean;
}) {
  const { t } = useI18n();
  const [custom, setCustom] = useState(false);
  const range = capability.integer_range;
  const customValue = -2;
  const defaultValue = -1;
  const numeric = !!range && (custom || typeof value === 'number');
  const label = (level: string) => {
    const key = `runningModels.detail.reasoningEffortLevels.${level}`;
    const translated = t(key);
    return translated === key ? level : translated;
  };
  return (
    <div className="space-y-2">
      <Select
        value={numeric ? customValue : value || defaultValue}
        placeholder={t('runningModels.detail.reasoningEffortDefault')}
        error={error}
        disabled={disabled}
        allowClear={false}
        options={[
          { value: defaultValue, label: t('runningModels.detail.reasoningEffortDefault') },
          ...capability.values.map((level) => ({ value: level, label: label(level) })),
          ...(range
            ? [{ value: customValue, label: t('runningModels.detail.reasoningEffortCustom') }]
            : []),
        ]}
        onChange={(next) => {
          setCustom(next === customValue);
          onChange?.(next === customValue || next === defaultValue ? undefined : next);
        }}
      />
      {numeric && range && (
        <Input
          type="number"
          aria-label={t('runningModels.detail.reasoningEffortCustom')}
          value={typeof value === 'number' ? value : ''}
          min={range[0]}
          max={range[1]}
          step={1}
          error={error}
          disabled={disabled}
          placeholder={`${range[0]}–${range[1]}`}
          onChange={(event) =>
            onChange?.(event.target.value === '' ? undefined : Number(event.target.value))
          }
        />
      )}
    </div>
  );
}
