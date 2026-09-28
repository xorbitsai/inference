'use client';

import { useState } from 'react';

import { Button } from '@/components/ui/button';
import PageContainer from '@/components/ui/page-container';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';

import ElasticLogs from './elastic-logs';
import RuntimeLogs from './runtime-logs';

export default function LogCenter() {
  const { t } = useI18n();
  const { clusterUIConfig, globalReady } = useGlobal();
  const [view, setView] = useState<'runtime' | 'indexed'>('runtime');

  if (!globalReady) return <PageContainer loading />;

  const indexedEnabled = Boolean(clusterUIConfig?.es_enabled);
  return (
    <div className="flex flex-col gap-4">
      {indexedEnabled && (
        <div className="flex gap-2">
          <Button
            aria-pressed={view === 'runtime'}
            variant={view === 'runtime' ? 'default' : 'outline'}
            onClick={() => setView('runtime')}
          >
            {t('logCenter.runtimeLogs')}
          </Button>
          <Button
            aria-pressed={view === 'indexed'}
            variant={view === 'indexed' ? 'default' : 'outline'}
            onClick={() => setView('indexed')}
          >
            {t('logCenter.indexedLogs')}
          </Button>
        </div>
      )}
      {view === 'indexed' && indexedEnabled ? <ElasticLogs /> : <RuntimeLogs />}
    </div>
  );
}
