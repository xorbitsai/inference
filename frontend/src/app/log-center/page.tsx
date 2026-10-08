import { PermissionGuard } from '@/components/auth/permission-guard';
import LogCenter from '@/components/pages/log-center';

export default function LogCenterPage() {
  return (
    <PermissionGuard scope="logs:list">
      <LogCenter />
    </PermissionGuard>
  );
}