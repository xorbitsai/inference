'use client';

import { usePathname } from 'next/navigation';
import RegisterModel from '@/components/pages/register-model';
import { getRigisterModelTyps } from '@/components/pages/register-model/utils';
import { getPathSegmentsAfter } from '@/lib/route-params';

export default function RegisterModelPageClient() {
  const pathname = usePathname();
  const [modelType = ''] = getPathSegmentsAfter(pathname, '/register-model');

  return <RegisterModel modelType={getRigisterModelTyps(modelType)} />;
}
