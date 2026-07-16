'use client';

import * as React from 'react';

import { cn } from '@/lib/utils';
import { useI18n } from '@/contexts/i18n-context';
import { getNamePathString } from '@/lib/form';
import { InfoTooltip } from '@/components/ui/tooltip';
import type { BaseFormFieldProps, FormFieldProps } from '@/types/form';
import { useFormContext } from './form';

function isEmptyValue(value: any) {
  // undefined / null
  if (value === undefined || value === null) {
    return true;
  }

  // string
  if (typeof value === 'string') {
    return !value.trim();
  }

  // array
  if (Array.isArray(value)) {
    return value.length === 0;
  }

  // boolean
  // false not empty
  if (typeof value === 'boolean') {
    return false;
  }

  return false;
}

function getDefaultValue(value: any, valuePropName: string) {
  // Existing value
  if (value !== undefined) {
    return value;
  }

  // switch / checkbox
  if (valuePropName === 'checked') {
    return false;
  }

  // default input/select
  return '';
}

function FormField({
  name,
  hidden = false,
  label,
  extra,
  rules,
  children,
  placeholder,
  disabled = false,
  valuePropName = 'value',
  layout = 'vertical',
  className,
  tooltip,
  normalize,
}: FormFieldProps) {
  const { t } = useI18n();
  const fieldKey = getNamePathString(name);
  const { form, errors, touched, setFieldError, registerField, unregisterField, setFieldTouched } =
    useFormContext();

  const fieldValue = form.hasFieldValue(fieldKey)
    ? form.getFieldValue(fieldKey)
    : form.getInitialFieldValue(fieldKey);

  const value = getDefaultValue(fieldValue, valuePropName);

  const error = errors[fieldKey];

  const isTouched = touched[fieldKey];

  const validate = React.useCallback(
    (val: any) => {
      for (const rule of rules || []) {
        // required
        if (rule.required) {
          if (isEmptyValue(val)) {
            return rule.message || t('common.valueEmpty');
          }
        }

        // pattern
        if (
          rule.pattern &&
          val !== undefined &&
          val !== null &&
          val !== '' &&
          !rule.pattern.test(String(val))
        ) {
          return rule.message || t('common.patternError');
        }
        // validator
        if (rule.validator && !rule.validator(val)) {
          return rule.message || t('common.valueEmpty');
        }
      }

      return '';
    },
    [rules, t]
  );

  const validateRef =
    React.useRef(validate);

  React.useEffect(() => {
    validateRef.current =
      validate;
  }, [validate]);

  React.useEffect(() => {
    const unmountField =
      form.mountField(fieldKey);

    registerField(fieldKey, () =>
      validateRef.current(
        form.getFieldValue(fieldKey)
      )
    );

    return () => {
      unregisterField(fieldKey);

      unmountField();
    };
  }, [form, fieldKey, registerField, unregisterField]);

  if (hidden) {
    return null;
  }

  if (!children) {
    throw new Error('FormField requires children unless hidden is true');
  }

  const handleChange = (nextValueOrEvent: any, ...args: any[]) => {
    const rawValue = nextValueOrEvent?.target ? nextValueOrEvent.target.value : nextValueOrEvent;

    const nextValue = normalize ? normalize(rawValue) : rawValue;
    
    form.setFieldValue(fieldKey, nextValue);

    // Mark as touched after the first interaction
    if (!isTouched) {
      setFieldTouched(fieldKey, true);
    }

    // Real-time validation
    const nextError = validate(nextValue);

    setFieldError(fieldKey, nextError);

    // Pass through the child component's raw onChange
    const childOnChange = (
      children.props as {
        onChange?: (value: any, ...args: any[]) => void;
      }
    ).onChange;

    childOnChange?.(nextValueOrEvent, ...args);
  };
  const mergedPlaceholder = placeholder || label;
  const child = React.cloneElement(children, {
    [valuePropName]: value,
    error: Boolean(error),
    onChange: handleChange,
    disabled,
    ...(mergedPlaceholder && {
      placeholder: mergedPlaceholder,
    }),
  } as BaseFormFieldProps);

  const isRequired = rules?.some((rule) => rule.required);

  return (
    <div className={cn('space-y-1.5', className)}>
      <div
        className={cn(
          layout === 'horizontal' ? 'flex items-center gap-3' : 'flex flex-col gap-1.5'
        )}
      >
        {label && (
          <label className={cn('text-sm font-medium flex items-center gap-1', layout === 'horizontal' && 'shrink-0')}>
            {label}
            {!!tooltip && <InfoTooltip content={tooltip} />}
            {isRequired && <span className="text-destructive">*</span>}
          </label>
        )}

        {child}
      </div>

      {error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : extra ? (
        <p className="text-xs text-muted-foreground">{extra}</p>
      ) : null}
    </div>
  );
}

export { FormField };
