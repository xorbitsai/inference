'use client';

import { usePathname } from 'next/navigation';
import RunningModelDetail from '@/components/pages/running-model-detail';
import { getPathSegmentsAfter } from '@/lib/route-params';

export default function RunningModelDetailPageClient() {
  const pathname = usePathname();
  const [modelUid = ''] = getPathSegmentsAfter(pathname, '/running-model');

  return <RunningModelDetail modelUid={modelUid} />;
}
