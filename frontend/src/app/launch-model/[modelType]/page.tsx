import LaunchModel from '@/components/pages/launch-model';
import { getLaunchModelRouteType, getInitialCustomType } from '@/components/pages/launch-model/utils';

interface LaunchModelPageProps {
  params: Promise<{
    modelType: string;
  }>;

  searchParams: Promise<{
    activeType?: string;
  }>;
}

export default async function LaunchModelPage({ params, searchParams }: LaunchModelPageProps) {
  const { modelType } = await params;
  const { activeType } = await searchParams;

  return (
    <LaunchModel
      routeType={getLaunchModelRouteType(modelType)}
      initialCustomType={getInitialCustomType(activeType)}
    />
  );
}
