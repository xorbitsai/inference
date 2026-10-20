'use client';

import { useEffect, useState } from 'react';

import { useI18n } from '@/contexts/i18n-context';
import { formatProgressDuration, type GenerationProgressTiming } from '@/lib/generation-progress';

export function ProgressTiming({ timing }: { timing: GenerationProgressTiming }) {
  const { t } = useI18n();
  const initialElapsed = timing.elapsed_seconds;
  const [elapsed, setElapsed] = useState(initialElapsed);

  useEffect(() => {
    setElapsed(initialElapsed);
    if (initialElapsed === undefined) return;
    const startedAt = performance.now();
    const timer = setInterval(() => {
      setElapsed(initialElapsed + (performance.now() - startedAt) / 1000);
    }, 1000);
    return () => clearInterval(timer);
  }, [initialElapsed]);

  return (
    <div className="flex w-full max-w-xs justify-between gap-4 text-xs tabular-nums text-muted-foreground">
      <span className="text-left">
        {t('runningModels.detail.elapsed')}: {formatProgressDuration(elapsed)}
      </span>
      <span className="text-right">
        {t('runningModels.detail.eta')}:{' '}
        {timing.eta_seconds == null
          ? t('runningModels.detail.estimating')
          : formatProgressDuration(timing.eta_seconds)}
      </span>
    </div>
  );
}
