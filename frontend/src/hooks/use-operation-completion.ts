'use client';
import { useEffect, useRef } from 'react';
type Operation = { operation_id: string; status: string };
const terminal = (status: string) => status === 'succeeded' || status === 'failed';
/** Historical completed jobs do not fire; observed transitions fire once. */
export function useOperationCompletion(operations: readonly Operation[], onCompleted?: () => void) {
  const previous = useRef(new Map<string, string>());
  const callback = useRef(onCompleted);
  callback.current = onCompleted;
  useEffect(() => {
    const completed = operations.some((operation) => {
      const status = previous.current.get(operation.operation_id);
      return status !== undefined && !terminal(status) && terminal(operation.status);
    });
    previous.current = new Map(
      operations.map((operation) => [operation.operation_id, operation.status])
    );
    if (completed) callback.current?.();
  }, [operations]);
}
