'use client';

import { useEffect, useState, FC, useMemo } from 'react';
import { PanelRightClose, PanelRight } from 'lucide-react';
import { useRouter } from 'next/navigation';
import request from '@/lib/request';
import PageContainer from '@/components/ui/page-container';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Form } from '@/components/ui/form';
import { Button } from '@/components/ui/button';
import { useI18n } from '@/contexts/i18n-context';
import { useForm } from '@/hooks/use-form';
import { cn } from '@/lib/utils';
import { ModelType, CUSTOM_MODEL_OPTIONS } from '@/constants';
import { ModelFormat, REGISTER_MODEL_INIT_DATA } from '@/constants/register';
import type { ModelPrompts, ModelFamily } from '@/types/services';
import type { RegisterModelType } from '@/types/common';
import AutoFill from './auto-fill';
import FormContent from './form-content';
import JsonView from './json-view';

const getPath = (path: string) => {
  const normalizedPath = (path || '').replace(/\\/g, '/');
  const baseDir = normalizedPath.substring(0, normalizedPath.lastIndexOf('/'));
  const filename = normalizedPath.substring(normalizedPath.lastIndexOf('/') + 1);
  return { baseDir, filename };
};

interface RegisterModelProps {
  modelType: RegisterModelType;
  modelName?: string;
}
const RegisterModel: FC<RegisterModelProps> = ({ modelType, modelName }) => {
  const isEdit = !!modelName;
  const isLLM = modelType === ModelType.LLM;
  const [form] = useForm();
  const { t } = useI18n();
  const router = useRouter();
  const tabs = useMemo(
    () => CUSTOM_MODEL_OPTIONS.map((item) => ({ value: item.value, label: t(item.labelKey) })),
    []
  );
  const [loading, setLoading] = useState(false);
  const [showJsonPreview, setShowJsonPreview] = useState(isEdit || false);
  const [modelPromptsMap, setModelPromptsMap] = useState<ModelPrompts>({});
  const [modelFamilyMap, setModelFamilyMap] = useState<ModelFamily>({});

  const fetchPrompts = async () => {
    const res = await request.get('/v1/models/prompts');
    setModelPromptsMap(res || {});
  };
  const fetchFamilies = async () => {
    const res = await request.get('/v1/models/families');
    setModelFamilyMap(res || {});
  };
  const handleJsonView = () => setShowJsonPreview(!showJsonPreview);
  const handleTabChange = (tab: string) => {
    router.push(`/register-model/${tab}`);
  };

  const transformFormValues = (values: Record<string, unknown>) => {
    const newValues: Record<string, unknown> = { ...values };

    if (Array.isArray(newValues?.model_specs)) {
      newValues.model_specs = newValues.model_specs.map((item: any) => {
        const newItem = { ...item };
        // Convert decimal model_size_in_billions (e.g., 7.2) to string (e.g., '7_2'), keep integers as numbers.
        if ('model_size_in_billions' in item) {
          const modelSizeInBillions = String(item?.model_size_in_billions);
          newItem.model_size_in_billions = modelSizeInBillions.includes('.')
            ? modelSizeInBillions.replace('.', '_')
            : Number(modelSizeInBillions);
        }
        // model_file_name_template is the part after the last '/', while model_uri is the part before it.
        if (item.model_format === ModelFormat.GGUFV2) {
          const { baseDir, filename } = getPath(item.model_uri);
          newItem.model_uri = baseDir;
          newItem.model_file_name_template = filename;
        }
        return newItem;
      });
    }
    if (modelType === ModelType.Audio && typeof newValues.model_ability === 'string') {
      newValues.model_ability = [newValues.model_ability];
    }
    return newValues;
  };
  const handleSubmit = async (values: Record<string, unknown>) => {
    const data = transformFormValues(values);
    setLoading(true);
    try {
      // Edit the registration model logic to first delete and then create
      if (isEdit) {
        await request.delete(`/v1/model_registrations/${modelType}/${modelName}`);
      }
      await request.post(`/v1/model_registrations/${modelType}`, {
        model: JSON.stringify(data),
        persist: true,
      });
      router.push(`/launch-model/custom?activeType=${modelType}`);
    } finally {
      setLoading(false);
    }
  };
  const onAutoFillBack = (data: Record<string, unknown>) => {
    form.setFieldsValue(data);
  };

  const handleGoBack = () => {
    router.push(`/launch-model/custom?activeType=${modelType}`);
  };
  useEffect(() => {
    fetchPrompts();
    fetchFamilies();
  }, []);

  useEffect(() => {
    if (modelType && modelName) {
      try {
        // read the model detail
        const modelStr = sessionStorage.getItem('customJsonData');
        if (!modelStr) {
          router.replace(`/register-model/${modelType}`);
          return;
        }
        const formData = JSON.parse(modelStr);
        if (formData.model_name !== modelName) {
          router.replace(`/register-model/${modelType}`);
          return;
        }

        if (Array.isArray(formData?.model_specs)) {
          formData.model_specs = formData.model_specs.map((item: any) => ({
            ...item,
            model_size_in_billions:
              typeof item?.model_size_in_billions === 'string' &&
              item.model_size_in_billions.includes('_')
                ? item.model_size_in_billions.replace('_', '.')
                : item?.model_size_in_billions,
          }));
        }
        form.setFieldsValue(formData);
      } catch {
        router.replace(`/register-model/${modelType}`);
      }
    }
  }, [modelType, modelName]);
  return (
    <PageContainer
      title={t('menu.registerModel')}
      extraContent={
        <div
          onClick={handleJsonView}
          className="flex gap-1 items-center text-muted-foreground hover:text-foreground cursor-pointer"
        >
          {showJsonPreview ? (
            <>
              {t('registerModel.hiddenJsonView')}
              <PanelRight size={16} />
            </>
          ) : (
            <>
              {t('registerModel.showJsonView')}
              <PanelRightClose size={16} />
            </>
          )}
        </div>
      }
    >
      <Form form={form} onFinish={handleSubmit} initialValues={REGISTER_MODEL_INIT_DATA[modelType]}>
        <Tabs
          value={modelType}
          onValueChange={handleTabChange}
          className="w-full gap-6 overflow-hidden"
        >
          <TabsList className="w-full justify-between bg-transparent p-0 h-auto border-b border-border/80 rounded-none flex overflow-x-auto">
            <div className="flex space-x-4">
              {tabs.map((item) => (
                <TabsTrigger
                  value={item.value}
                  key={item.value}
                  className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
                >
                  {item.label}
                </TabsTrigger>
              ))}
            </div>
            {isLLM && <AutoFill modelFamilyMap={modelFamilyMap} autoFillBack={onAutoFillBack} />}
          </TabsList>
          <TabsContent value={modelType} className="flex px-1 pb-1">
            <div className="flex-1 min-w-0">
              <FormContent
                modelType={modelType}
                form={form}
                modelPromptsMap={modelPromptsMap}
                modelFamilyMap={modelFamilyMap}
              />
              <div className="mt-4 flex items-center gap-2">
                <Button type="submit" loading={loading}>
                  {isEdit ? t('common.edit') : t('registerModel.registerModel')}
                </Button>
                {isEdit && (
                  <Button variant="outline" onClick={handleGoBack}>
                    {t('common.goBack')}
                  </Button>
                )}
              </div>
            </div>
            <div
              className={cn(
                'ml-6 w-[420px] flex-shrink-0 overflow-hidden transition-[width,margin-left,opacity] duration-200 ease-out will-change-[width,margin-left,opacity]',
                {
                  '!w-0 !ml-0 opacity-0 pointer-events-none': !showJsonPreview,
                }
              )}
            >
              <div
                className={cn(
                  'w-[420px] min-w-[420px] transition-[opacity,transform] duration-200 ease-out will-change-transform',
                  {
                    'translate-x-4 opacity-0': !showJsonPreview,
                  }
                )}
              >
                <JsonView form={form} transformFormValues={transformFormValues} />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </Form>
    </PageContainer>
  );
};
export default RegisterModel;
