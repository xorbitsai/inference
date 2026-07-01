'use client';

import * as React from 'react';

import {
  FieldName,
  FormInstance,
  FormValues,
} from '@/types/form';

import {
  deleteValue,
  deepMerge,
  getNamePathString,
  getValue,
  hasValue,
  setValue,
} from '@/lib/form';

export function createForm(): FormInstance {
  const store = {
    current: {} as FormValues,
  };

  const initialValues = {
    current: {},
  };

  const listeners =
    new Set<() => void>();

  const resetListeners =
    new Set<() => void>();

  const mountedFields =
    new Map<string, number>();

  const pendingUnmounts =
    new Map<
      string,
      {
        name: FieldName;
        timer: ReturnType<typeof setTimeout>;
      }
    >();

  const notify = () => {
    listeners.forEach((fn) =>
      fn()
    );
  };

  const initFieldValue = (
    name: FieldName
  ) => {
    if (
      hasValue(store.current, name) ||
      !hasValue(initialValues.current, name)
    ) {
      return;
    }

    store.current = setValue(
      store.current,
      name,
      structuredClone(
        getValue(
          initialValues.current,
          name
        )
      )
    );

    notify();
  };

  const deleteFieldValue = (
    name: FieldName
  ) => {
    if (!hasValue(store.current, name)) {
      return;
    }

    store.current = deleteValue(
      store.current,
      name
    );

    notify();
  };

  const flushPendingUnmounts = () => {
    Array.from(pendingUnmounts)
      .forEach(
        ([
          key,
          {
            name,
            timer,
          },
        ]) => {
          clearTimeout(timer);

          pendingUnmounts.delete(key);

          if (mountedFields.has(key)) {
            return;
          }

          deleteFieldValue(name);
        }
      );
  };

  const mountField = (
    name: FieldName
  ) => {
    const key =
      getNamePathString(name);

    const pendingUnmount =
      pendingUnmounts.get(key);

    if (pendingUnmount) {
      clearTimeout(
        pendingUnmount.timer
      );

      pendingUnmounts.delete(key);
    }

    mountedFields.set(
      key,
      (mountedFields.get(key) ?? 0) + 1
    );

    initFieldValue(name);

    let mounted = true;

    return () => {
      if (!mounted) {
        return;
      }

      mounted = false;

      const nextCount =
        (mountedFields.get(key) ?? 1) - 1;

      if (nextCount > 0) {
        mountedFields.set(
          key,
          nextCount
        );

        return;
      }

      mountedFields.delete(key);

      const unmountTimer =
        setTimeout(() => {
          pendingUnmounts.delete(key);

          if (mountedFields.has(key)) {
            return;
          }

          deleteFieldValue(name);
        }, 0);

      pendingUnmounts.set(
        key,
        {
          name,
          timer: unmountTimer,
        }
      );
    };
  };

  return {
    store,

    initialValues,

    getFieldValue(
      name: FieldName
    ) {
      return getValue(
        store.current,
        name
      );
    },

    getInitialFieldValue(
      name: FieldName
    ) {
      return getValue(
        initialValues.current,
        name
      );
    },

    hasFieldValue(
      name: FieldName
    ) {
      return hasValue(
        store.current,
        name
      );
    },

    getFieldsValue() {
      flushPendingUnmounts();

      return store.current;
    },

    initFieldValue,

    deleteFieldValue,

    mountField,

    setFieldValue(
      name: FieldName,
      value: unknown
    ) {
      store.current = setValue(
        store.current,
        name,
        value
      );

      notify();
    },

    setFieldsValue(values) {
      store.current = deepMerge(
        store.current,
        values
      );
    
      notify();
    },
    
    resetFields() {
      flushPendingUnmounts();

      let nextStore =
        {} as FormValues;

      Array.from(mountedFields.keys())
        .sort(
          (a, b) =>
            a.split('.').length -
            b.split('.').length
        )
        .forEach((name) => {
          if (
            !hasValue(
              initialValues.current,
              name
            )
          ) {
            return;
          }

          nextStore = setValue(
            nextStore,
            name,
            structuredClone(
              getValue(
                initialValues.current,
                name
              )
            )
          );
        });

      store.current =
        nextStore;

      notify();

      resetListeners.forEach(
        (fn) => fn()
      );
    },

    subscribe(callback) {
      listeners.add(callback);

      return () => {
        listeners.delete(
          callback
        );
      };
    },

    registerReset(callback) {
      resetListeners.add(
        callback
      );

      return () => {
        resetListeners.delete(
          callback
        );
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
  name: FieldName,
  form: FormInstance
) {
  const getWatchedValue = React.useCallback(
    () =>
      form.hasFieldValue(name)
        ? form.getFieldValue(name)
        : form.getInitialFieldValue(name),
    [form, name]
  );

  const [value, setValue] =
    React.useState(
      getWatchedValue
    );

  React.useEffect(() => {
    setValue(getWatchedValue());

    return form.subscribe(() => {
      setValue(
        getWatchedValue()
      );
    });
  }, [form, getWatchedValue]);

  return value;
}

export function useFormValues(form: FormInstance) {
  const getValues = React.useCallback(
    () => structuredClone(deepMerge(form.initialValues.current, form.getFieldsValue())),
    [form]
  );

  const [values, setValues] = React.useState(getValues);

  React.useEffect(() => {
    setValues(getValues());

    return form.subscribe(() => {
      setValues(getValues());
    });
  }, [form, getValues]);

  return values;
}
