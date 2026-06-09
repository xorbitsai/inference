import RegisterModel from '@/components/pages/register-model';
import { getRigisterModelTyps } from '@/components/pages/register-model/utils';

interface RegisterModelPageProps {
  params: Promise<{
    modelType: string;
  }>;
}

export default async function RegisterModelPage({ params }: RegisterModelPageProps) {
  const { modelType } = await params;
  return <RegisterModel modelType={getRigisterModelTyps(modelType)} />;
}
