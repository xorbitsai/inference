import { FC, PropsWithChildren } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PageHeaderProps {
  showPageHeader?: boolean;
  title?: React.ReactNode;
  subTitle?: React.ReactNode;
  extraContent?: React.ReactNode;
  loading?: boolean;
  className?: string;
  headerClassName?: string;
}
const PageContainer: FC<PropsWithChildren<PageHeaderProps>> = ({
  showPageHeader = true,
  title,
  subTitle,
  extraContent,
  children,
  loading = false,
  className,
  headerClassName,
}) => {
  return (
    <div className={cn('w-full flex flex-col gap-6', className)}>
      {showPageHeader && (
        <div className={cn('flex items-center justify-between', headerClassName)}>
          <div className="min-w-0">
            <h1 className="mb-1 min-w-0 text-3xl font-bold">{title}</h1>
            {subTitle && <div className="text-muted-foreground">{subTitle}</div>}
          </div>
          {extraContent}
        </div>
      )}
      {loading ? (
        <div className="min-h-[60vh] flex items-center justify-center bg-background">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div>{children}</div>
      )}
    </div>
  );
};
export default PageContainer;
