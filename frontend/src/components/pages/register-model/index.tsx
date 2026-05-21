'use client';

import { useEffect, useState } from 'react';
import { PanelRightClose, PanelRight } from 'lucide-react';
import request from '@/lib/request';
import PageContainer from '@/components/ui/page-container';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Form } from '@/components/ui/form';
import { Button } from '@/components/ui/button';
import { useI18n } from '@/contexts/i18n-context';
import { useForm } from '@/hooks/use-form';
import { cn } from '@/lib/utils';
import { ModelType } from '@/constants';
import type { ModelFamily } from '@/types/services';
import AutoFill from './auto-fill';
import FormContent from './form-content';

const RegisterModel = () => {
  const [form] = useForm();
  const { t } = useI18n();
  const tabs = [
    { key: ModelType.LLM, label: t('model.languageModels') },
    { key: ModelType.Embedding, label: t('model.embeddingModels') },
    { key: ModelType.Rerank, label: t('model.rerankModels') },
    { key: ModelType.Image, label: t('model.imageModels') },
    { key: ModelType.Audio, label: t('model.audioModels') },
    { key: ModelType.Flexible, label: t('model.flexibleModels') },
  ];
  const [activeTab, setActiveTab] = useState<ModelType>(ModelType.LLM);
  const [showJsonPreview, setShowJsonPreview] = useState(true);
  const [modelFamilyMap, setModelFamilyMap] = useState<ModelFamily>({});

  const fetchPrompts = async () => {
    const res = await request.get('/v1/models/prompts');
    console.log(res, '-prompts--');
  };
  const fetchFamilies = async () => {
    const res = await request.get('/v1/models/families');
    setModelFamilyMap(res);
  };
  const handleJsonView = () => setShowJsonPreview(!showJsonPreview);
  const handleTabChange = (tab: string) => setActiveTab(tab as ModelType);
  const handleSubmit = (values: any) => {
    console.log(values, 'values');
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
          {t('registerModel.showCustomJsonConfig')}
          {showJsonPreview ? <PanelRightClose size={16} /> : <PanelRight size={16} />}
        </div>
      }
    >
      <Form form={form} onFinish={handleSubmit} >
        <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full gap-6">
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
            <AutoFill />
          </TabsList>
          <TabsContent value={activeTab} className="flex">
            <div className="flex-1 min-w-0">
              <FormContent modelType={activeTab} form={form} modelFamilyMap={modelFamilyMap} />
              <Button className="mt-4" type="submit">
                {t('registerModel.registerModel')}
              </Button>
            </div>
            <div
              className={cn(
                'ml-6 w-[420px] flex-shrink-0 relative transition-[width] duration-200 ease-linear',
                {
                  '!w-0 !ml-0': !showJsonPreview,
                }
              )}
            >
              jsonview
            </div>
          </TabsContent>
        </Tabs>
      </Form>
    </PageContainer>
  );
};
export default RegisterModel;
