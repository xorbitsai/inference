'use client';

import { useCallback, useRef } from 'react';
import type { FormInstance } from '@/types/form';
import { transformFetchToForm } from '../utils';
import type { LaunchConfigHistoryItem } from './launch-history';
import { selectLatestModelLaunchHistory } from './launch-history-utils.mjs';

export function replaceLaunchHistoryFormSnapshot(
  form: FormInstance,
  item?: LaunchConfigHistoryItem
) {
  form.resetFields();
  if (item) {
    form.setFieldsValue(transformFetchToForm(item.data));
  }
}

export function useLaunchHistoryForm(form: FormInstance, isOpen: boolean, modelName?: string) {
  const formEditedRef = useRef(false);

  const markFormEdited = useCallback(() => {
    formEditedRef.current = true;
  }, []);

  const resetFormEdited = useCallback(() => {
    formEditedRef.current = false;
  }, []);

  const handleHistoryRefreshed = useCallback(
    (history: LaunchConfigHistoryItem[]) => {
      if (!isOpen) return;

      const latestConfig = selectLatestModelLaunchHistory(
        history,
        modelName,
        formEditedRef.current
      );
      if (latestConfig) {
        replaceLaunchHistoryFormSnapshot(form, latestConfig);
      }
    },
    [form, isOpen, modelName]
  );

  return {
    markFormEdited,
    resetFormEdited,
    handleHistoryRefreshed,
  };
}
