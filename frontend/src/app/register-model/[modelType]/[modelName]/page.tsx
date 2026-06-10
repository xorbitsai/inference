import RegisterModel from '@/components/pages/register-model';
import { getRigisterModelTyps } from '@/components/pages/register-model/utils';

interface RegisterModelPageProps {
  params: Promise<{
    modelType: string;
    modelName: string;
  }>;
}

export default async function RegisterModelPage({ params }: RegisterModelPageProps) {
  const { modelType, modelName } = await params;
  return <RegisterModel modelType={getRigisterModelTyps(modelType)} modelName={modelName} />;
}
