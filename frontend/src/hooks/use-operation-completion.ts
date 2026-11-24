'use client';
import { useEffect, useRef } from 'react';
import { isOperationFinished } from '@/lib/operation-status';
type Operation = { operation_id: string; status: string };
/** Historical completed jobs do not fire; observed transitions fire once. */
export function useOperationCompletion(operations: readonly Operation[], onCompleted?: () => void) {
  const previous = useRef(new Map<string, string>());
  const callback = useRef(onCompleted);
  callback.current = onCompleted;
  useEffect(() => {
    const completed = operations.some((operation) => {
      const status = previous.current.get(operation.operation_id);
      return (
        status !== undefined &&
        !isOperationFinished(status) &&
        isOperationFinished(operation.status)
      );
    });
    previous.current = new Map(
      operations.map((operation) => [operation.operation_id, operation.status])
    );
    if (completed) callback.current?.();
  }, [operations]);
}
