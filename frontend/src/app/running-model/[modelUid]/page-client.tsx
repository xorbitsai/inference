'use client';

import { useParams } from 'next/navigation';
import RunningModelDetail from '@/components/pages/running-model-detail';
import { decodeRouteParam } from '@/lib/route-params';

export default function RunningModelDetailPageClient() {
  const params = useParams<{ modelUid: string }>();
  const modelUid = decodeRouteParam(params.modelUid);

  return <RunningModelDetail modelUid={modelUid} />;
}
