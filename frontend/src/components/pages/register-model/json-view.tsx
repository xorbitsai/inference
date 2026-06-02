'use client';

import { FC, useMemo } from 'react';
import { useFormValues } from '@/hooks/use-form';
import { JSONSyntaxHighlighter } from '@/components/ui/json-syntax-highlighter';
import { FormInstance } from '@/types/form';

interface JsonViewProps {
  form: FormInstance;
  transformFormValues: (v: Record<string, unknown>) => void;
}
const JsonView: FC<JsonViewProps> = ({ form, transformFormValues }) => {
  const values = useFormValues(form);
  const data = useMemo(() => transformFormValues(values), [values, transformFormValues]);

  return (
    <JSONSyntaxHighlighter
      className="whitespace-pre-wrap break-words overflow-x-hidden"
      data={data}
    />
  );
};
export default JsonView;
