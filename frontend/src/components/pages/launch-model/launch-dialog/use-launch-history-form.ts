'use client';

import { useCallback, useRef } from 'react';
import type { FormInstance } from '@/types/form';
import { transformFetchToForm } from '../utils';
import type { LaunchConfigHistoryItem } from './launch-history';
import { selectLatestModelLaunchHistory } from './launch-history-utils.mjs';

export function replaceLaunchHistoryFormSnapshot(
  form: FormInstance,
  item?: LaunchConfigHistoryItem,
  onSnapshotApplied?: (values: Record<string, unknown>) => void
) {
  form.resetFields();
  if (item) {
    const values = transformFetchToForm(item.data);
    onSnapshotApplied?.(values);
    form.setFieldsValue(values);
  }
}

export function useLaunchHistoryForm(
  form: FormInstance,
  isOpen: boolean,
  modelName?: string,
  onSnapshotApplied?: (values: Record<string, unknown>) => void
) {
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
        replaceLaunchHistoryFormSnapshot(form, latestConfig, onSnapshotApplied);
      }
    },
    [form, isOpen, modelName, onSnapshotApplied]
  );

  return {
    markFormEdited,
    resetFormEdited,
    handleHistoryRefreshed,
  };
}
