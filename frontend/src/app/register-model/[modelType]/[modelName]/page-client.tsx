'use client';

import { useParams } from 'next/navigation';
import RegisterModel from '@/components/pages/register-model';
import { getRigisterModelTyps } from '@/components/pages/register-model/utils';
import { decodeRouteParam } from '@/lib/route-params';

export default function RegisterModelEditPageClient() {
  const params = useParams<{ modelType: string; modelName: string }>();
  const modelType = decodeRouteParam(params.modelType);
  const modelName = decodeRouteParam(params.modelName);

  return <RegisterModel modelType={getRigisterModelTyps(modelType)} modelName={modelName} />;
}
