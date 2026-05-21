// components/form/form-field.tsx
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
   * 组件值属性
   *
   * Input:
   * value
   *
   * Switch:
   * checked
   */
  valuePropName?: string;

  /**
   * 布局模式
   *
   * vertical:
   * label
   * control
   * error
   *
   * horizontal:
   * label + control 同行
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
  // false 不是 empty
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
  // 已存在值
  if (value !== undefined) {
    return value;
  }

  // switch / checkbox
  if (valuePropName === 'checked') {
    return false;
  }

  // 默认 input/select
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
                'common.valueEmpty'
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
  
    // 首次操作后标记 touched
    if (!isTouched) {
      setFieldTouched(name, true);
    }
  
    // 实时校验
    const nextError =
      validate(nextValue);
  
    setFieldError(
      name,
      nextError
    );
  
    // 透传子组件原始 onChange
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