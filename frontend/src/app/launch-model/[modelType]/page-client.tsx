'use client';

import { Suspense } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import LaunchModel from '@/components/pages/launch-model';
import {
  getLaunchModelRouteType,
  getInitialCustomType,
} from '@/components/pages/launch-model/utils';
import { decodeRouteParam } from '@/lib/route-params';

function LaunchModelPageInner() {
  const params = useParams<{ modelType: string }>();
  const searchParams = useSearchParams();
  const modelType = decodeRouteParam(params.modelType);
  const activeType = searchParams.get('activeType') ?? undefined;

  return (
    <LaunchModel
      routeType={getLaunchModelRouteType(modelType)}
      initialCustomType={getInitialCustomType(activeType)}
    />
  );
}

export default function LaunchModelPageClient() {
  // useSearchParams requires a Suspense boundary during prerender.
  return (
    <Suspense fallback={null}>
      <LaunchModelPageInner />
    </Suspense>
  );
}
