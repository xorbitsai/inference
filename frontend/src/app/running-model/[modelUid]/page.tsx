import RunningModelDetail from '@/components/pages/running-model-detail';

interface RunningModelDetailPageProps {
  params: Promise<{
    modelUid: string;
  }>;
}
export default async function RunningModelDetailPage({ params }: RunningModelDetailPageProps) {
  const { modelUid } = await params;
  return <RunningModelDetail modelUid={modelUid} />;
}
