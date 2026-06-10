'use client';

import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';

const MonitorCenter = () => {
  const { clusterUIConfig } = useGlobal();
  const { t } = useI18n();
  if (!clusterUIConfig?.grafana_url) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground font-medium">
        {t('monitoring.notConfigured')}
      </div>
    );
  }
  return null;
};
export default MonitorCenter;
