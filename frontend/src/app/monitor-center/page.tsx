import { PermissionGuard } from '@/components/auth/permission-guard';
import MonitorCenter from '@/components/pages/monitor-center';

export default function MonitorCenterPage() {
  return (
    <PermissionGuard scope="monitor:view">
      <MonitorCenter />
    </PermissionGuard>
  );
}