'use client';

import * as React from 'react';

import { cn } from '@/lib/utils';
import { useI18n } from '@/contexts/i18n-context';
import type { BaseFormFieldProps } from '@/types/common';
import { useFormContext } from './form';

type RequiredRule = {
  required: true;

  message?: string;

  pattern?: never;

  validator?: never;
};

type PatternRule = {
  pattern: RegExp;

  message: string;

  required?: never;

  validator?: never;
};

type ValidatorRule = {
  validator: (value: any) => boolean;

  message: string;

  required?: never;

  pattern?: never;
};

export type Rule =
  | RequiredRule
  | PatternRule
  | ValidatorRule;

interface FormFieldProps {
  name: string;

  label?: React.ReactNode;

  extra?: React.ReactNode;

  rules?: Rule[];

  placeholder?: React.ReactNode;

  disabled?: boolean;

  children: React.ReactElement;

  /**
   * Input:
   * value
   *
   * Switch:
   * checked
   */
  valuePropName?: string;

  /**
   * horizontal:
   * label + control on the same row
   */
  layout?: 'vertical' | 'horizontal';
}


function isEmptyValue(value: any) {
  // undefined / null
  if (
    value === undefined ||
    value === null
  ) {
    return true;
  }

  // string
  if (
    typeof value === 'string'
  ) {
    return !value.trim();
  }

  // array
  if (Array.isArray(value)) {
    return value.length === 0;
  }

  // boolean
  // false not empty
  if (
    typeof value === 'boolean'
  ) {
    return false;
  }

  return false;
}

function getDefaultValue(
  value: any,
  valuePropName: string
) {
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
  label,
  extra,
  rules,
  children,
  placeholder,
  disabled = false,
  valuePropName = 'value',
  layout = 'vertical',
}: FormFieldProps) {
  const { t } = useI18n();

  const {
    form,
    errors,
    touched,
    setFieldError,
    registerField,
    unregisterField,
    setFieldTouched,
  } = useFormContext();

  const value = getDefaultValue(
    form.getFieldValue(name),
    valuePropName
  );

  const error = errors[name];

  const isTouched =
    touched[name];

  const validate =
    React.useCallback(
      (val: any) => {
        for (const rule of rules || []) {
          // required
          if (rule.required) {
            if (isEmptyValue(val)) {
              return (
                rule.message ||
                t(
                  'common.valueEmpty'
                )
              );
            }
          }

          // pattern
          if (
            rule.pattern &&
            typeof val ===
              'string' &&
            val &&
            !rule.pattern.test(val)
          ) {
            return (
              rule.message ||
              t(
                'common.patternError'
              )
            );
          }

          // validator
          if (
            rule.validator &&
            !rule.validator(val)
          ) {
            return (
              rule.message ||
              t(
                'common.valueEmpty'
              )
            );
          }
        }

        return '';
      },
      [rules, t]
    );

  React.useEffect(() => {
    registerField(name, () =>
      validate(
        form.getFieldValue(name)
      )
    );

    return () => {
      unregisterField(name);
    };
  }, [
    form,
    name,
    validate,
    registerField,
    unregisterField,
  ]);

  const handleChange = (
    nextValueOrEvent: any,
    ...args: any[]
  ) => {
    const nextValue =
      nextValueOrEvent?.target
        ? nextValueOrEvent.target
            .value
        : nextValueOrEvent;
  
    form.setFieldValue(
      name,
      nextValue
    );
  
    // Mark as touched after the first interaction
    if (!isTouched) {
      setFieldTouched(name, true);
    }
  
    // Real-time validation
    const nextError =
      validate(nextValue);
  
    setFieldError(
      name,
      nextError
    );
  
    // Pass through the child component's raw onChange
    const childOnChange =
      (
        children.props as {
          onChange?: (
            value: any,
            ...args: any[]
          ) => void;
        }
      ).onChange;
  
    childOnChange?.(
      nextValueOrEvent,
      ...args
    );
  };
  const mergedPlaceholder = placeholder || label
  const child =
    React.cloneElement(children, {
      [valuePropName]: value,
      error: Boolean(error),
      onChange: handleChange,
      disabled,
      ...(mergedPlaceholder && {
        placeholder: mergedPlaceholder,
      }),
    } as BaseFormFieldProps);

  const isRequired =
    rules?.some(
      (rule) => rule.required
    );

  return (
    <div className="space-y-1.5">
      <div
        className={cn(
          layout === 'horizontal'
            ? 'flex items-center gap-3'
            : 'flex flex-col gap-1.5'
        )}
      >
        {label && (
          <label
            className={cn(
              'text-sm font-medium',
              layout ===
                'horizontal' &&
                'shrink-0'
            )}
          >
            {label}

            {isRequired && (
              <span className="ml-1 text-destructive">
                *
              </span>
            )}
          </label>
        )}

        {child}
      </div>

      {error ? (
        <p className="text-sm text-destructive">
          {error}
        </p>
      ) : extra ? (
        <p className="text-xs text-muted-foreground">
          {extra}
        </p>
      ) : null}
    </div>
  );
}

export { FormField };