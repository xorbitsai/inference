'use client';

import { useParams } from 'next/navigation';
import RegisterModel from '@/components/pages/register-model';
import { getRigisterModelTyps } from '@/components/pages/register-model/utils';
import { decodeRouteParam } from '@/lib/route-params';

export default function RegisterModelPageClient() {
  const params = useParams<{ modelType: string }>();
  const modelType = decodeRouteParam(params.modelType);

  return <RegisterModel modelType={getRigisterModelTyps(modelType)} />;
}
