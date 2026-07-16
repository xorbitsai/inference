'use client';

import { Suspense } from 'react';
import { usePathname, useSearchParams } from 'next/navigation';
import LaunchModel from '@/components/pages/launch-model';
import {
  getLaunchModelRouteType,
  getInitialCustomType,
} from '@/components/pages/launch-model/utils';
import { getPathSegmentsAfter } from '@/lib/route-params';

function LaunchModelPageInner() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [modelType = ''] = getPathSegmentsAfter(pathname, '/launch-model');
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
