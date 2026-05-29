'use client';

import { useCallback, useEffect, useState } from 'react';
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
import { ModelType } from '@/constants';
import { ModelFormat, REGISTER_MODEL_INIT_DATA } from '@/constants/register';
import type { ModelPrompts, ModelFamily } from '@/types/services';
import AutoFill from './auto-fill';
import FormContent from './form-content';
import JsonView from './json-view';

type RegisterModelType = Exclude<ModelType, ModelType.Video | ModelType.Custom>;

const getPath = (path: string) => {
  const normalizedPath = (path || '').replace(/\\/g, '/');
  const baseDir = normalizedPath.substring(0, normalizedPath.lastIndexOf('/'));
  const filename = normalizedPath.substring(normalizedPath.lastIndexOf('/') + 1);
  return { baseDir, filename };
};
const RegisterModel = () => {
  const [form] = useForm();
  const { t } = useI18n();
  const router = useRouter();
  const tabs = [
    { key: ModelType.LLM, label: t('model.languageModels') },
    { key: ModelType.Embedding, label: t('model.embeddingModels') },
    { key: ModelType.Rerank, label: t('model.rerankModels') },
    { key: ModelType.Image, label: t('model.imageModels') },
    { key: ModelType.Audio, label: t('model.audioModels') },
    { key: ModelType.Flexible, label: t('model.flexibleModels') },
  ];
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<RegisterModelType>(ModelType.LLM);
  const [showJsonPreview, setShowJsonPreview] = useState(false);
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
    setActiveTab(tab as RegisterModelType);

    setTimeout(() => form.setFieldsValue(REGISTER_MODEL_INIT_DATA[tab as RegisterModelType]), 0);
  };

  const transformFormValues = (values: Record<string, unknown>) => {
    let newValues: Record<string, unknown> = { version: 2, ...values };

    if (Array.isArray(newValues?.model_specs)) {
      newValues.model_specs = newValues.model_specs.map((item: any) => {
        let newItem = { ...item };
        // Convert decimal model_size_in_billions (e.g., 7.2) to string (e.g., '7_2'), keep integers as numbers.
        if ('model_size_in_billions' in item) {
          const modelSizeInBillions = String(item?.model_size_in_billions);
          newItem.model_size_in_billions = modelSizeInBillions.includes('.')
            ? modelSizeInBillions.replace('.', '_')
            : Number(modelSizeInBillions);
        }
        // model_file_name_template is the part after the last '/', while model_uri is the part before it.
        if (item.model_format === ModelFormat.GGUF) {
          const { baseDir, filename } = getPath(item.model_uri);
          newItem.model_uri = baseDir;
          newItem.model_file_name_template = filename;
        }
        return newItem;
      });
    }
    if (activeTab === ModelType.Audio && typeof newValues.model_ability === 'string') {
      newValues.model_ability = [newValues.model_ability];
    }
    return newValues;
  };
  const handleSubmit = async (values: Record<string, unknown>) => {
    const data = transformFormValues(values);
    setLoading(true);
    try {
      await request.post(`/v1/model_registrations/${activeTab}`, {
        model: JSON.stringify(data),
        persist: true,
      });
      router.push('/launch-model');
    } finally {
      setLoading(false);
    }
  };
  const onAutoFillBack = (data: Record<string, unknown>) => {
    form.setFieldsValue(data);
  };
  useEffect(() => {
    fetchPrompts();
    fetchFamilies();
  }, []);
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
      <Form form={form} onFinish={handleSubmit} initialValues={REGISTER_MODEL_INIT_DATA[activeTab]}>
        <Tabs
          value={activeTab}
          onValueChange={handleTabChange}
          className="w-full gap-6 overflow-hidden"
        >
          <TabsList className="w-full justify-between bg-transparent p-0 h-auto border-b border-border/80 rounded-none flex overflow-x-auto">
            <div className="flex space-x-4">
              {tabs.map((item) => (
                <TabsTrigger
                  value={item.key}
                  key={item.key}
                  className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
                >
                  {item.label}
                </TabsTrigger>
              ))}
            </div>
            <AutoFill modelFamilyMap={modelFamilyMap} autoFillBack={onAutoFillBack} />
          </TabsList>
          <TabsContent value={activeTab} className="flex">
            <div className="flex-1 min-w-0">
              <FormContent
                modelType={activeTab}
                form={form}
                modelPromptsMap={modelPromptsMap}
                modelFamilyMap={modelFamilyMap}
              />
              <Button className="mt-4" type="submit" loading={loading}>
                {t('registerModel.registerModel')}
              </Button>
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
