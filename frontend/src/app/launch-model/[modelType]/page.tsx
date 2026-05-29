import LaunchModel from '@/components/pages/launch-model';
import { getLaunchModelRouteType } from '@/components/pages/launch-model/utils';

interface LaunchModelPageProps {
  params: Promise<{
    modelType: string;
  }>;
}

export default async function LaunchModelPage({ params }: LaunchModelPageProps) {
  const { modelType } = await params;
  return <LaunchModel routeType={getLaunchModelRouteType(modelType)} />;
}
