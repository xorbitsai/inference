'use client';

import * as React from 'react';
import { getNamePathString } from '@/lib/form';
import { cn } from '@/lib/utils';
import { FieldName, FormInstance, FormContextType, FormValues } from '@/types/form';

const FormContext = React.createContext<FormContextType | null>(null);

export function useFormContext() {
  const context = React.useContext(FormContext);

  if (!context) {
    throw new Error('FormField must be used within Form');
  }

  return context;
}

interface FormProps extends Omit<React.ComponentProps<'form'>, 'onSubmit'> {
  form: FormInstance;

  initialValues?: Record<string, unknown>;

  onFinish?: (values: FormValues) => void;
}

export function Form({ form, initialValues, children, onFinish, className, ...props }: FormProps) {
  const [, forceUpdate] = React.useState({});

  const [errors, setErrors] = React.useState<Record<string, string>>({});

  const [touched, setTouched] = React.useState<Record<string, boolean>>({});

  const touchedRef = React.useRef(touched);

  const validatorsRef = React.useRef<Record<string, () => string>>({});

  touchedRef.current = touched;

  // Keep initial values as a template; visible fields pull from it when they mount.
  if (initialValues && form.initialValues.current !== initialValues) {
    form.initialValues.current = initialValues;
  }

  React.useEffect(() => {
    return form.subscribe(() => {
      forceUpdate({});

      setErrors((prev) => {
        const next = { ...prev };

        Object.keys(touchedRef.current).forEach((name) => {
          const validate = validatorsRef.current[name];

          if (!validate) return;

          const error = validate();

          if (error) {
            next[name] = error;
          } else {
            delete next[name];
          }
        });

        const keys = Object.keys(next);

        if (
          keys.length === Object.keys(prev).length &&
          keys.every((name) => next[name] === prev[name])
        ) {
          return prev;
        }

        return next;
      });
    });
  }, [form]);

  // Clear UI state on resetFields
  React.useEffect(() => {
    return form.registerReset(() => {
      setErrors({});

      setTouched({});
    });
  }, [form]);


  const removeFieldState = React.useCallback(
    <T extends string | boolean,>(
      state: Record<string, T>,
      key: string
    ) => {
      return Object.fromEntries(
        Object.entries(state).filter(
          ([name]) =>
            name !== key &&
            !name.startsWith(`${key}.`)
        )
      ) as Record<string, T>;
    },
    []
  );

  const setFieldError = React.useCallback((
    name: FieldName,
    error: string
  ) => {
    const key =
      getNamePathString(name);

    setErrors((prev) => {
      if (!error) {
        return removeFieldState(
          prev,
          key
        );
      }

      return {
        ...prev,
        [key]: error,
      };
    });
  }, [removeFieldState]);

  const setFieldTouched = React.useCallback((
    name: FieldName,
    value: boolean
  ) => {
    const key =
      getNamePathString(name);

    setTouched((prev) => {
      if (!value) {
        return removeFieldState(
          prev,
          key
        );
      }

      return {
        ...prev,
        [key]: value,
      };
    });
  }, [removeFieldState]);

  const clearFieldState = React.useCallback((
    name: FieldName
  ) => {
    const key =
      getNamePathString(name);

    setErrors((prev) =>
      removeFieldState(prev, key)
    );

    setTouched((prev) =>
      removeFieldState(prev, key)
    );
  }, [removeFieldState]);

  const registerField = React.useCallback((
    name: FieldName,
    validate: () => string
  ) => {
    const key =
      getNamePathString(name);
  
    validatorsRef.current[key] =
      validate;
  }, []);

  const unregisterField = React.useCallback((
    name: FieldName
  ) => {
    const key =
      getNamePathString(name);
  
    delete validatorsRef.current[key];

    clearFieldState(key);
  }, [clearFieldState]);


  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const nextErrors: Record<string, string> = {};

    const nextTouched: Record<string, boolean> = {};

    Object.entries(validatorsRef.current).forEach(([name, validate]) => {
      nextTouched[name] = true;

      const error = validate();

      if (error) {
        nextErrors[name] = error;
      }
    });

    setTouched(nextTouched);

    setErrors(nextErrors);

    if (Object.keys(nextErrors).length === 0) {
      onFinish?.(form.getFieldsValue());
    }
  };

  return (
    <FormContext.Provider
      value={{
        form,
        errors,
        touched,
        setFieldError,
        setFieldTouched,
        clearFieldState,
        registerField,
        unregisterField,
      }}
    >
      <form {...props} onSubmit={handleSubmit} className={cn('space-y-3', className)}>
        {children}
      </form>
    </FormContext.Provider>
  );
}
