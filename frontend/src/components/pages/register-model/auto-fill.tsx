'use client';

import { FC, useMemo, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from '@/components/ui/dialog';
import { Form } from '@/components/ui/form';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import { useForm } from '@/hooks/use-form';
import { isEmpty } from '@/lib/is';
import request from '@/lib/request';
import type { ModelFamily } from '@/types/services';
import type { Option } from '@/types/common';

interface AutoFillProps {
  modelFamilyMap: ModelFamily;
  autoFillBack: (v: Record<string, unknown>) => void;
}
const AutoFill: FC<AutoFillProps> = ({ modelFamilyMap, autoFillBack }) => {
  const { t } = useI18n();
  const [form] = useForm();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const modelFamilyOptions = useMemo(() => {
    const options: Option[] = [];

    if (isEmpty(modelFamilyMap)) return options;

    const seen = new Set<string>();

    Object.values(modelFamilyMap).forEach((list) => {
      list.forEach((item) => {
        if (seen.has(item)) return;

        seen.add(item);
        options.push({
          label: item,
          value: item,
        });
      });
    });

    return options;
  }, [modelFamilyMap]);

  const handleSubmit = async (values: Record<string, unknown>) => {
    setLoading(true);
    try {
      const res = await request.post('/v1/models/llm/auto-register', values);
      const newValues = { ...(res || {}) };
      if (Array.isArray(newValues?.model_specs)) {
        newValues.model_specs = newValues.model_specs.map((item: any) => ({
          ...item,
          model_size_in_billions:
            typeof item?.model_size_in_billions === 'string' &&
            item.model_size_in_billions.includes('_')
              ? item.model_size_in_billions.replace('_', '.')
              : item?.model_size_in_billions,
        }));
      }
      autoFillBack(newValues);
      setOpen(false);
    } finally {
      setLoading(false);
    }
  };
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="mb-1">
          <Sparkles size={14} />
          {t('registerModel.autoFill')}
        </Button>
      </DialogTrigger>

      <DialogContent showCloseButton={true}>
        <DialogHeader>
          <DialogTitle>{t('registerModel.autoFill')}</DialogTitle>
        </DialogHeader>
        <Form form={form} onFinish={handleSubmit}>
          <FormField
            name="model_path"
            label={t('registerModel.modelPath')}
            rules={[{ required: true }]}
          >
            <Input />
          </FormField>
          <FormField
            name="model_family"
            label={t('registerModel.modelFamily')}
            rules={[{ required: true }]}
          >
            <Select options={modelFamilyOptions} showSearch />
          </FormField>
          <DialogFooter>
            <Button type="submit" loading={loading}>
              {t('registerModel.autoFill')}
            </Button>
          </DialogFooter>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
export default AutoFill;
