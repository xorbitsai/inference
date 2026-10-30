'use client';

import { Button } from '@/components/ui/button';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { useI18n } from '@/contexts/i18n-context';
import type { BaseFormFieldProps } from '@/types/form';

const MAX_CONTEXT_LENGTH = 2 ** 31 - 1;

type ContextInputProps = BaseFormFieldProps<number | ''> & {
  modelMaximum?: number;
  extensionMaximum?: number;
};

function ContextInput({
  value,
  onChange,
  error,
  modelMaximum,
  extensionMaximum,
}: ContextInputProps) {
  const { t } = useI18n();
  const knownMaximum =
    Number.isSafeInteger(modelMaximum) && modelMaximum! > 0 ? modelMaximum : undefined;
  const selected = typeof value === 'number' ? value : undefined;
  const maximum = extensionMaximum || knownMaximum || MAX_CONTEXT_LENGTH;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <input
          type="range"
          aria-label={t('launchModel.contextLength')}
          className="w-full accent-primary disabled:opacity-40"
          min={1}
          max={maximum}
          step={1}
          value={Math.min(selected || knownMaximum || 1, maximum)}
          disabled={!knownMaximum}
          onChange={(event) => onChange?.(Number(event.target.value))}
        />
        <Input
          type="number"
          aria-label={t('launchModel.contextLength')}
          className="w-32 shrink-0"
          min={1}
          max={maximum}
          step={1}
          value={value ?? ''}
          placeholder={knownMaximum ? String(knownMaximum) : ''}
          error={error}
          onChange={(event) =>
            onChange?.(event.target.value === '' ? '' : Number(event.target.value))
          }
        />
        <Button type="button" variant="ghost" size="sm" onClick={() => onChange?.('')}>
          {t('launchModel.contextLengthDefault')}
        </Button>
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{knownMaximum ? '1' : t('launchModel.contextLengthUnknown')}</span>
        {knownMaximum && (
          <span>
            {t('launchModel.contextLengthDefault')}: {knownMaximum.toLocaleString()}
          </span>
        )}
      </div>
    </div>
  );
}

export default function ContextLengthField({
  modelMaximum,
  extensionMaximum,
}: {
  modelMaximum?: number;
  extensionMaximum?: number;
}) {
  const { t } = useI18n();
  const native =
    Number.isSafeInteger(modelMaximum) && modelMaximum! > 0
      ? Math.min(modelMaximum!, MAX_CONTEXT_LENGTH)
      : undefined;
  const extended =
    native && Number.isSafeInteger(extensionMaximum) && extensionMaximum! > native
      ? Math.min(extensionMaximum!, MAX_CONTEXT_LENGTH)
      : undefined;
  const maximum = extended || native;
  return (
    <FormField
      name="context_length"
      label={t('launchModel.contextLength')}
      tooltip={
        extended && native
          ? t('launchModel.contextLengthYarn', {
              native: native.toLocaleString(),
              max: extended.toLocaleString(),
            })
          : native
            ? t('launchModel.contextLengthNative', { native: native.toLocaleString() })
            : t('launchModel.contextLengthTip')
      }
      rules={[
        {
          validator: (value) =>
            value === undefined ||
            value === null ||
            value === '' ||
            (Number.isSafeInteger(value) && value >= 1 && value <= (maximum || MAX_CONTEXT_LENGTH)),
          message: t('launchModel.contextLengthInvalid', { max: maximum || MAX_CONTEXT_LENGTH }),
        },
      ]}
    >
      <ContextInput modelMaximum={native} extensionMaximum={extended} />
    </FormField>
  );
}
