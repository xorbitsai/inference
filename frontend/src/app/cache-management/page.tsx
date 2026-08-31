import { PermissionGuard } from '@/components/auth/permission-guard';
import CacheManagement from '@/components/pages/cache-management';

export default function CacheManagementPage() {
  return (
    <PermissionGuard scope={['models:read', 'cache:list', 'virtualenv:list']}>
      <CacheManagement />
    </PermissionGuard>
  );
}
