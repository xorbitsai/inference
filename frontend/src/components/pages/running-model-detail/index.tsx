'use client';

import { FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Info, WandSparkles, Code } from 'lucide-react';
import { useRouter } from 'next/navigation';

import { Button } from '@/components/ui/button';
import { CollapsiblePanel } from '@/components/ui/collapsible';
import PageContainer from '@/components/ui/page-container';
import { ModelAbility } from '@/constants';
import request from '@/lib/request';
import type { RunningModelDetail as RunningModelDetailType } from '@/types/services';

import { CAPABILITY_CONFIGS } from './capability-config';
import CapabilityTaskPanel, { CapabilityTaskPanelMethod } from './panels/capability-task-panel';
import { ChatPanel } from './panels/chat-panel';
import { Select } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import { TryApiDrawer } from './components/try-api-drawer';
import { transformRunningModelDetail } from './utils';

interface RunningModelDetailProps {
  modelUid: string;
}

function DetailItem({ label, value }: { label: string; value?: string | number | null }) {
  return (
    <div className="min-w-0 rounded-2xl bg-muted/40 px-4 py-3">
      <div className="text-xs font-medium tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 truncate text-sm font-medium text-foreground">{value || '-'}</div>
    </div>
  );
}

function ModelDetails({ model, modelUid }: { model: RunningModelDetailType; modelUid: string }) {
  return (
    <CollapsiblePanel
      defaultOpen={false}
      title="Model Details"
      description="Runtime metadata is collapsed by default so the capability workspace stays in focus."
      icon={<Info className="size-5 text-primary" />}
      className="rounded-xl"
      contentClassName="p-5"
    >
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <DetailItem label="Model UID" value={modelUid} />
        <DetailItem label="Model Name" value={model.model_name} />
        <DetailItem label="Model Type" value={model.model_type} />
        <DetailItem label="Model Engine" value={model.model_hub} />
        <DetailItem label="Model Format" value={model.model_format} />
        <DetailItem label="Model Size" value={model.model_size_in_billions} />
        <DetailItem label="Quantization" value={model.quantization} />
        <DetailItem label="Context" value={model.context_length} />
        <DetailItem label="Replica" value={model.replica} />
        <DetailItem label="Address" value={model.address} />
      </div>
      {!!model.model_description && (
        <div className="mt-4 rounded-2xl bg-muted/40 px-4 py-3 text-sm leading-6 text-muted-foreground">
          {model.model_description}
        </div>
      )}
    </CollapsiblePanel>
  );
}
const EmptyForAbility = () => (
  <div className="flex min-h-[calc(100vh-216px)] flex-col items-center justify-center rounded-3xl border bg-card text-center">
    <WandSparkles className="mb-4 size-10 text-muted-foreground" />
    <h2 className="text-lg font-semibold">No supported interactive capability</h2>
    <p className="mt-2 text-sm text-muted-foreground">
      This model is running, but the current UI does not have a panel for its abilities yet.
    </p>
  </div>
);
const RunningModelDetail: FC<RunningModelDetailProps> = ({ modelUid }) => {
  const router = useRouter();
  const { t } = useI18n();
  const [model, setModel] = useState<RunningModelDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const isChat = (model?.model_ability || []).includes(ModelAbility.Chat);
  const [selectAbility, setSelectAbility] = useState<ModelAbility | undefined>(undefined);
  const [tryApiOpen, setTryApiOpen] = useState(false);
  const capabilityTaskPanelRef = useRef<CapabilityTaskPanelMethod>(null);
  const tryApiAbility = isChat ? ModelAbility.Chat : selectAbility;

  const fetchModel = useCallback(() => {
    setLoading(true);
    request
      .get<RunningModelDetailType>(`/v1/models/${modelUid}`)
      .then((res) => {
        const newModelDetail = transformRunningModelDetail(res) as RunningModelDetailType;
        const firstAbility = newModelDetail.model_ability.filter(
          (item) => !item.includes('_')
        )?.[0];
        setSelectAbility(firstAbility);
        setModel(newModelDetail);
      })
      .finally(() => setLoading(false));
  }, [modelUid]);

  const abilityOptions = useMemo(() => {
    const abilities = model?.model_ability || [];
    if (!abilities.length) return [];
    return abilities
      .filter((item) => !item.includes('_')) // Filter out sub-capabilities (as agreed upon by the front-end and back-end, where those underlined are sub-capabilities)
      .map((item) => {
        const Icon = CAPABILITY_CONFIGS[item]?.icon;
        return {
          value: item,
          prefix: Icon ? <Icon className="size-4" /> : undefined,
          label: t(`launchModel.${item}`),
        };
      });
  }, [model, t]);

  const handleAbility = (value?: ModelAbility) => {
    if (!value) return;
    setSelectAbility(value as ModelAbility);
    capabilityTaskPanelRef.current?.reset?.();
  };
  const renderCapability = () => {
    if (!model) return null;

    if (isChat) {
      return <ChatPanel model={model} modelUid={modelUid} />;
    }

    if (!selectAbility || !CAPABILITY_CONFIGS[selectAbility]) {
      return <EmptyForAbility />;
    }

    return (
      <CapabilityTaskPanel
        config={CAPABILITY_CONFIGS[selectAbility]}
        model={model}
        modelUid={modelUid}
        ref={capabilityTaskPanelRef}
      />
    );
  };

  useEffect(() => {
    fetchModel();
  }, [fetchModel]);
  return (
    <PageContainer
      title={
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 rounded-full"
            onClick={() => router.back()}
          >
            <ArrowLeft className="size-5" />
          </Button>
          {modelUid}
        </div>
      }
      loading={loading}
      className="gap-5"
      extraContent={
        <div className="flex items-center gap-2">
          {!isChat && (
            <Select
              className="w-40"
              allowClear={false}
              options={abilityOptions}
              value={selectAbility}
              onChange={handleAbility}
            />
          )}
          <Button type="button"  onClick={() => setTryApiOpen(true)}>
            <Code />
            Try To API
          </Button>
        </div>
      }
    >
      {model && (
        <div className="space-y-5">
          <ModelDetails model={model} modelUid={modelUid} />
          {renderCapability()}
        </div>
      )}
      <TryApiDrawer
        open={tryApiOpen}
        onOpenChange={setTryApiOpen}
        modelUid={modelUid}
        ability={tryApiAbility}
      />
    </PageContainer>
  );
};

export default RunningModelDetail;
