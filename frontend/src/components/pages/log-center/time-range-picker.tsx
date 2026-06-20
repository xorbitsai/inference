'use client';

import { Check, ChevronDown, Clock } from 'lucide-react';
import { useState } from 'react';

import { Button } from '@/components/ui/button';
import { DateTimePicker } from '@/components/ui/date-time-picker';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { LOG_TIME_RANGES } from '@/constants/logs';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';

import type { TimeRangeValue } from './types';
import { formatDateTime, toMilliseconds } from './utils';

interface TimeRangePickerProps {
  value: TimeRangeValue;
  onChange: (value: TimeRangeValue) => void;
}

export function TimeRangePicker({ value, onChange }: TimeRangePickerProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [absoluteFrom, setAbsoluteFrom] = useState('');
  const [absoluteTo, setAbsoluteTo] = useState('');

  const selectedRelativeRange = LOG_TIME_RANGES.find(
    (item) => item.from === value.from && item.to === value.to
  );
  const isAbsoluteRange = value.to !== 'now';
  const hasAbsoluteRange = Boolean(absoluteFrom && absoluteTo);
  const absoluteRangeInvalid =
    hasAbsoluteRange && new Date(absoluteTo).getTime() <= new Date(absoluteFrom).getTime();
  const canApplyAbsoluteRange = hasAbsoluteRange && !absoluteRangeInvalid;
  const triggerLabel = isAbsoluteRange
    ? `${formatDateTime(value.from)} ~ ${formatDateTime(value.to)}`
    : t(selectedRelativeRange?.labelKey || 'monitorCenter.time.1h');

  const handleApplyAbsoluteRange = () => {
    if (!canApplyAbsoluteRange) return;

    onChange({
      from: toMilliseconds(absoluteFrom),
      to: toMilliseconds(absoluteTo),
    });
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            'h-9 min-w-0 justify-between px-3',
            isAbsoluteRange ? 'w-[23rem] max-w-full' : 'w-40'
          )}
        >
          <span className="flex min-w-0 items-center gap-2">
            <Clock className="size-4 text-muted-foreground" />
            <span className="truncate">{triggerLabel}</span>
          </span>
          <ChevronDown className="size-4 text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[min(calc(100vw-2rem),30rem)] p-0">
        <div className="grid max-h-[28rem] grid-cols-1 overflow-hidden sm:grid-cols-[18rem_12rem]">
          <div className="border-b p-4 sm:border-b-0 sm:border-r">
            <div className="mb-4 text-sm font-medium text-muted-foreground">
              {t('monitorCenter.absoluteRange')}
            </div>
            <div className="space-y-3">
              <DateTimePicker
                value={absoluteFrom}
                onChange={setAbsoluteFrom}
                aria-label={t('common.from')}
                placeholder={t('common.from')}
                showClear={false}
                showSelectedTime={false}
              />
              <DateTimePicker
                value={absoluteTo}
                onChange={setAbsoluteTo}
                aria-label={t('common.to')}
                placeholder={t('common.to')}
                inputClassName={absoluteRangeInvalid ? 'border-destructive' : undefined}
                showClear={false}
                showSelectedTime={false}
              />
              {absoluteRangeInvalid && (
                <div className="text-xs text-destructive">
                  {t('monitorCenter.invalidTimeRange')}
                </div>
              )}
              <Button
                className="w-full"
                disabled={!canApplyAbsoluteRange}
                onClick={handleApplyAbsoluteRange}
              >
                {t('monitorCenter.applyTimeRange')}
              </Button>
            </div>
          </div>
          <div className="overflow-y-auto py-2">
            {LOG_TIME_RANGES.map((item) => {
              const selected = value.from === item.from && value.to === item.to;

              return (
                <button
                  key={item.labelKey}
                  type="button"
                  className={cn(
                    'flex h-10 w-full items-center justify-between px-4 text-left text-sm transition-colors hover:bg-accent',
                    selected && 'bg-accent text-accent-foreground'
                  )}
                  onClick={() => {
                    onChange({ from: item.from, to: item.to });
                    setOpen(false);
                  }}
                >
                  <span>{t(item.labelKey)}</span>
                  {selected && <Check className="size-4 text-primary" />}
                </button>
              );
            })}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
