'use client';

import { FC, useMemo, MouseEvent } from 'react';
import { Copy } from 'lucide-react';
import { useFormValues } from '@/hooks/use-form';
import { JSONSyntaxHighlighter } from '@/components/ui/json-syntax-highlighter';
import { FormInstance } from '@/types/form';
import { copyToClipboard } from '@/lib/utils';

interface JsonViewProps {
  form: FormInstance;
  transformFormValues: (v: Record<string, unknown>) => void;
}
const JsonView: FC<JsonViewProps> = ({ form, transformFormValues }) => {
  const values = useFormValues(form);
  const data = useMemo(() => transformFormValues(values), [values, transformFormValues]);
  const handleCopy = (e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    const str = JSON.stringify(data, null, 2);
    copyToClipboard(str);
  };
  return (
    <div className="relative">
      <button
        className="absolute right-2 top-2 p-1 text-muted-foreground transition-colors hover:text-foreground"
        onClick={handleCopy}
      >
        <Copy className="size-4" />
      </button>
      <JSONSyntaxHighlighter
        className="whitespace-pre-wrap break-words overflow-x-hidden"
        data={data}
      />
    </div>
  );
};
export default JsonView;
