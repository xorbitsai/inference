'use client';
import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import RunningModel from './index';

function RunningModelWithSelection() {
  const params = useSearchParams();
  return <RunningModel modelUid={params.get('model_uid') || undefined} />;
}

export default function RunningModelPageEntry() {
  return (
    <Suspense fallback={null}>
      <RunningModelWithSelection />
    </Suspense>
  );
}
