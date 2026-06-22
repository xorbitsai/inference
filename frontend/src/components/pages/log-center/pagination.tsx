'use client';

import { Button } from '@/components/ui/button';
import { LOG_PAGE_SIZE } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';

interface LogPaginationProps {
  total: number;
  pageFrom: number;
  onPageFromChange: (value: number) => void;
}

export function LogPagination({ total, pageFrom, onPageFromChange }: LogPaginationProps) {
  const { t } = useI18n();
  const maxAccessibleTotal = Math.min(total, 10000);
  const totalPages = Math.ceil(maxAccessibleTotal / LOG_PAGE_SIZE);
  const currentPage = Math.floor(pageFrom / LOG_PAGE_SIZE) + 1;
  const isNextDisabled = pageFrom + LOG_PAGE_SIZE >= maxAccessibleTotal;

  return (
    <div className="flex min-h-12 items-center justify-end gap-3 border-t px-4 py-2">
      <span className="text-sm text-muted-foreground">{t('logCenter.totalHits', { count: total })}</span>
      <Button
        variant="outline"
        size="sm"
        disabled={pageFrom === 0}
        onClick={() => onPageFromChange(Math.max(0, pageFrom - LOG_PAGE_SIZE))}
      >
        {t('logCenter.prevPage')}
      </Button>
      <span className="text-sm">
        {currentPage} / {totalPages || 1}
      </span>
      <Button
        variant="outline"
        size="sm"
        disabled={isNextDisabled}
        onClick={() => onPageFromChange(pageFrom + LOG_PAGE_SIZE)}
      >
        {t('logCenter.nextPage')}
      </Button>
    </div>
  );
}
