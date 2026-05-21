'use client';

import * as React from 'react';

export type FormValues = Record<
  string,
  any
>;

export interface FormInstance {
  store: React.MutableRefObject<FormValues>;

  initialValues: React.MutableRefObject<FormValues>;

  getFieldValue: (
    name: string
  ) => any;

  getFieldsValue: () => FormValues;

  setFieldValue: (
    name: string,
    value: any
  ) => void;

  resetFields: () => void;

  subscribe: (
    callback: () => void
  ) => () => void;

  registerReset: (
    callback: () => void
  ) => () => void;
}

export function createForm(): FormInstance {
  const store = {
    current: {} as FormValues,
  };

  const initialValues = {
    current: {} as FormValues,
  };

  const listeners = new Set<
    () => void
  >();

  const resetCallbacks = new Set<
    () => void
  >();

  const notify = () => {
    listeners.forEach((fn) => fn());
  };

  return {
    store,

    initialValues,

    getFieldValue(name) {
      return store.current[name];
    },

    getFieldsValue() {
      return store.current;
    },

    setFieldValue(name, value) {
      store.current[name] = value;

      notify();
    },

    resetFields() {
      store.current = {
        ...initialValues.current,
      };

      resetCallbacks.forEach((fn) =>
        fn()
      );

      notify();
    },

    subscribe(callback) {
      listeners.add(callback);

      return () => {
        listeners.delete(callback);
      };
    },

    registerReset(callback) {
      resetCallbacks.add(callback);

      return () => {
        resetCallbacks.delete(callback);
      };
    },
  };
}

export function useForm() {
  const formRef =
    React.useRef<FormInstance | null>(
      null
    );

  if (!formRef.current) {
    formRef.current =
      createForm();
  }

  return [formRef.current] as const;
}

export function useWatch(
  name: string,
  form: FormInstance
) {
  const [value, setValue] =
    React.useState(
      form.getFieldValue(name)
    );

  React.useEffect(() => {
    return form.subscribe(() => {
      setValue(
        form.getFieldValue(name)
      );
    });
  }, [form, name]);

  return value;
}