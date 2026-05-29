'use client';

import * as React from 'react';

import { cn } from '@/lib/utils';
import { getNamePathString } from '@/lib/form';

import { useFormContext } from './form';

import type {
  FormListProps,
  FormListRenderProps,
} from '@/types/form';

export function FormList<T = any>({
  name,
  label,
  extra,
  children,
  renderAction,
  layout = 'vertical',
  className
}: FormListProps<T>) {
  const { form, clearFieldState } =
    useFormContext();

  const fieldKey =
    getNamePathString(name);

  const [, forceUpdate] =
    React.useState({});

  React.useEffect(() => {
    return form.subscribe(() => {
      forceUpdate({});
    });
  }, [form]);

  React.useEffect(() => {
    const unmountField =
      form.mountField(fieldKey);

    return () => {
      clearFieldState(fieldKey);

      unmountField();
    };
  }, [form, fieldKey, clearFieldState]);

  const values =
    (form.hasFieldValue(fieldKey)
      ? form.getFieldValue(fieldKey)
      : form.getInitialFieldValue(fieldKey)) ??
    [];

  const fields = values.map(
    (_: any, index: number) => ({
      key: `${index}`,
      name: index,
    })
  );

  const add = (
    defaultValue?: T
  ) => {
    form.setFieldValue(name, [
      ...values,
      defaultValue,
    ]);
  };

  const remove = (
    index: number
  ) => {
    form.setFieldValue(
      name,
      values.filter(
        (
          _: any,
          i: number
        ) => i !== index
      )
    );
  };

  const renderProps: FormListRenderProps<T> =
    {
      fields,
      add,
      remove,
    };

  return (
    <div className={cn("space-y-2", className)}>
      {(label || renderAction) && (
        <div
          className={cn(
            layout ===
              'horizontal'
              ? 'flex items-center gap-3'
              : 'flex items-center justify-between'
          )}
        >
          {label && (
            <label className="text-sm font-medium">
              {label}
            </label>
          )}

          {renderAction?.(
            renderProps
          )}
        </div>
      )}

      {children(renderProps)}

      {extra && (
        <p className="text-xs text-muted-foreground">
          {extra}
        </p>
      )}
    </div>
  );
}
