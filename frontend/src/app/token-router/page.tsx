import { PermissionGuard } from '@/components/auth/permission-guard';
import TokenRouterPage from '@/components/pages/token-router';

export default function Page() {
  return (
    <PermissionGuard scope="routers:list">
      <TokenRouterPage />
    </PermissionGuard>
  );
}
