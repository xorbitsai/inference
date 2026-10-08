'use client';

import { type FormEvent, useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { LOG_PAGE_SIZE } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';

interface LogPaginationProps {
  total: number;
  pageFrom: number;
  onPageFromChange: (value: number) => void;
}

export function LogPagination({ total, pageFrom, onPageFromChange }: LogPaginationProps) {
  const { t } = useI18n();
  const [jumpPage, setJumpPage] = useState('1');
  const totalPages = Math.ceil(total / LOG_PAGE_SIZE) || 1;
  const currentPage = Math.floor(pageFrom / LOG_PAGE_SIZE) + 1;
  const lastPageFrom = (totalPages - 1) * LOG_PAGE_SIZE;
  const isNextDisabled = pageFrom + LOG_PAGE_SIZE >= total;

  useEffect(() => {
    setJumpPage(String(currentPage));
  }, [currentPage]);

  const goToPage = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const page = Number(jumpPage);
    if (!Number.isInteger(page) || page < 1 || page > totalPages) {
      setJumpPage(String(currentPage));
      return;
    }

    onPageFromChange((page - 1) * LOG_PAGE_SIZE);
    setJumpPage(String(page));
  };

  return (
    <div className="flex min-h-12 flex-wrap items-center justify-end gap-3 border-t px-4 py-2">
      <span className="text-sm text-muted-foreground">{t('logCenter.totalHits', { count: total })}</span>
      <form className="flex items-center gap-2 text-sm" noValidate onSubmit={goToPage}>
        <Label className="sr-only" htmlFor="log-jump-page">
          {t('logCenter.pageNumber')}
        </Label>
        <span>{t('logCenter.pagePrefix')}</span>
        <Input
          id="log-jump-page"
          className="h-8 w-16 text-center tabular-nums"
          type="number"
          inputMode="numeric"
          min={1}
          max={totalPages}
          step={1}
          value={jumpPage}
          aria-label={t('logCenter.jumpToPage')}
          disabled={total === 0}
          onChange={(event) => setJumpPage(event.target.value)}
        />
        <span>{t('logCenter.pageOf', { count: totalPages })}</span>
      </form>
      <Button
        variant="outline"
        size="icon"
        className="size-8"
        aria-label={t('logCenter.firstPage')}
        title={t('logCenter.firstPage')}
        disabled={pageFrom === 0}
        onClick={() => onPageFromChange(0)}
      >
        <ChevronsLeft />
      </Button>
      <Button
        variant="outline"
        size="icon"
        className="size-8"
        aria-label={t('logCenter.prevPage')}
        title={t('logCenter.prevPage')}
        disabled={pageFrom === 0}
        onClick={() => onPageFromChange(Math.max(0, pageFrom - LOG_PAGE_SIZE))}
      >
        <ChevronLeft />
      </Button>
      <Button
        variant="outline"
        size="icon"
        className="size-8"
        aria-label={t('logCenter.nextPage')}
        title={t('logCenter.nextPage')}
        disabled={isNextDisabled}
        onClick={() => onPageFromChange(pageFrom + LOG_PAGE_SIZE)}
      >
        <ChevronRight />
      </Button>
      <Button
        variant="outline"
        size="icon"
        className="size-8"
        aria-label={t('logCenter.lastPage')}
        title={t('logCenter.lastPage')}
        disabled={pageFrom === lastPageFrom}
        onClick={() => onPageFromChange(lastPageFrom)}
      >
        <ChevronsRight />
      </Button>
    </div>
  );
}
