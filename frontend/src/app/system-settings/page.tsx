import { PermissionGuard } from '@/components/auth/permission-guard';
import SystemSettings from '@/components/pages/system-settings';

export default function SystemSettingsPage() {
  return (
    <PermissionGuard scope="settings:read">
      <SystemSettings />
    </PermissionGuard>
  );
}
