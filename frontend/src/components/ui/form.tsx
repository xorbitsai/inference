'use client';

import * as React from 'react';

import { FormInstance } from '@/hooks/use-form';

type FormContextType = {
  form: FormInstance;

  errors: Record<string, string>;

  touched: Record<string, boolean>;

  setFieldError: (
    name: string,
    error: string
  ) => void;

  setFieldTouched: (
    name: string,
    touched: boolean
  ) => void;

  registerField: (
    name: string,
    validate: () => string
  ) => void;

  unregisterField: (
    name: string
  ) => void;
};

const FormContext =
  React.createContext<FormContextType | null>(
    null
  );

export function useFormContext() {
  const context =
    React.useContext(FormContext);

  if (!context) {
    throw new Error(
      'FormField must be used within Form'
    );
  }

  return context;
}

interface FormProps {
  form: FormInstance;

  initialValues?: Record<
    string,
    any
  >;

  children: React.ReactNode;

  onFinish?: (
    values: Record<string, any>
  ) => void;
}

export function Form({
  form,
  initialValues,
  children,
  onFinish,
}: FormProps) {
  const [, forceUpdate] =
    React.useState({});

  const [errors, setErrors] =
    React.useState<
      Record<string, string>
    >({});

  const [touched, setTouched] =
    React.useState<
      Record<string, boolean>
    >({});

  const validatorsRef = React.useRef<
    Record<string, () => string>
  >({});

  // init initialValues
  if (
    initialValues &&
    Object.keys(form.store.current)
      .length === 0
  ) {
    form.initialValues.current =
      initialValues;

    form.store.current = {
      ...initialValues,
    };
  }

  React.useEffect(() => {
    return form.subscribe(() => {
      forceUpdate({});
    });
  }, [form]);

  // Clear UI state on resetFields
  React.useEffect(() => {
    return form.registerReset(() => {
      setErrors({});

      setTouched({});
    });
  }, [form]);

  const setFieldError = (
    name: string,
    error: string
  ) => {
    setErrors((prev) => ({
      ...prev,
      [name]: error,
    }));
  };

  const setFieldTouched = (
    name: string,
    value: boolean
  ) => {
    setTouched((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const registerField = (
    name: string,
    validate: () => string
  ) => {
    validatorsRef.current[name] =
      validate;

    // Do not override initialValues
    if (
      !(name in form.store.current) &&
      !(name in form.initialValues.current)
    ) {
      form.store.current[name] =
        undefined;
    }
  };

  const unregisterField = (
    name: string
  ) => {
    delete validatorsRef.current[name];
  };

  const handleSubmit = (
    e: React.FormEvent
  ) => {
    e.preventDefault();

    const nextErrors: Record<
      string,
      string
    > = {};

    const nextTouched: Record<
      string,
      boolean
    > = {};

    Object.entries(
      validatorsRef.current
    ).forEach(([name, validate]) => {
      nextTouched[name] = true;

      const error = validate();

      if (error) {
        nextErrors[name] = error;
      }
    });

    setTouched(nextTouched);

    setErrors(nextErrors);

    if (
      Object.keys(nextErrors).length === 0
    ) {
      onFinish?.(
        form.getFieldsValue()
      );
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
        registerField,
        unregisterField,
      }}
    >
      <form
        onSubmit={handleSubmit}
        className="space-y-3"
      >
        {children}
      </form>
    </FormContext.Provider>
  );
}