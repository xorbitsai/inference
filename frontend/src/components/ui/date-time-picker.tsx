'use client';

import * as React from 'react';
import { CalendarDays, ChevronLeft, ChevronRight } from 'lucide-react';
import { format } from 'date-fns';

import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';
import type { Locale } from '@/types/common';

interface DateTimePickerProps extends Omit<
  React.ComponentProps<'button'>,
  'type' | 'value' | 'onChange'
> {
  value?: string;
  onChange?: (value: string) => void;
  label?: string;
  placeholder?: string;
  inputClassName?: string;
  showClear?: boolean;
  showSelectedTime?: boolean;
  showTime?: boolean;
}

const HOURS = Array.from({ length: 24 }, (_, index) => index);
const MINUTES = Array.from({ length: 60 }, (_, index) => index);
const MONTH_FORMAT_MAP: Record<Locale, string> = {
  en: 'MMM yyyy',
  zh: 'yyyy年MM月',
  ja: 'yyyy年MM月',
  ko: 'yyyy년 MM월',
};

const pad = (value: number) => String(value).padStart(2, '0');

const parseDateTimeValue = (value?: string) => {
  if (!value) return undefined;

  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [year, month, day] = value.split('-').map(Number);
    return new Date(year, month - 1, day);
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? undefined : date;
};

const toDateTimeValue = (date: Date) => format(date, "yyyy-MM-dd'T'HH:mm");
const toDateValue = (date: Date) => format(date, 'yyyy-MM-dd');

const getMonthDays = (date: Date) => {
  const year = date.getFullYear();
  const month = date.getMonth();
  const firstDay = new Date(year, month, 1);
  const startDate = new Date(year, month, 1 - firstDay.getDay());

  return Array.from({ length: 42 }, (_, index) => {
    const day = new Date(startDate);
    day.setDate(startDate.getDate() + index);
    return day;
  });
};

const isSameDay = (left?: Date, right?: Date) => {
  if (!left || !right) return false;

  return (
    left.getFullYear() === right.getFullYear() &&
    left.getMonth() === right.getMonth() &&
    left.getDate() === right.getDate()
  );
};

export function DateTimePicker({
  value,
  onChange,
  placeholder = 'Select date and time',
  label,
  className,
  inputClassName,
  showClear = true,
  showSelectedTime = true,
  showTime = true,
  disabled,
  ...props
}: DateTimePickerProps) {
  const { locale, t } = useI18n();
  const selectedDate = React.useMemo(() => parseDateTimeValue(value), [value]);
  const [open, setOpen] = React.useState(false);
  const [viewDate, setViewDate] = React.useState(() => selectedDate || new Date());

  React.useEffect(() => {
    if (selectedDate) {
      setViewDate(selectedDate);
    }
  }, [selectedDate]);

  const calendarDays = React.useMemo(() => getMonthDays(viewDate), [viewDate]);
  const displayValue = selectedDate
    ? format(selectedDate, showTime ? 'yyyy/MM/dd HH:mm' : 'yyyy/MM/dd')
    : placeholder;
  const weekDays = React.useMemo(
    () => [
      t('common.weekDays.sun'),
      t('common.weekDays.mon'),
      t('common.weekDays.tue'),
      t('common.weekDays.wed'),
      t('common.weekDays.thu'),
      t('common.weekDays.fri'),
      t('common.weekDays.sat'),
    ],
    [t]
  );
  const monthLabel = format(viewDate, MONTH_FORMAT_MAP[locale]);

  const updateDate = (updater: (current: Date) => Date) => {
    const current = selectedDate || viewDate;
    const nextDate = updater(new Date(current));

    setViewDate(nextDate);
    onChange?.(showTime ? toDateTimeValue(nextDate) : toDateValue(nextDate));
  };

  const handleSelectDay = (day: Date) => {
    updateDate((current) => {
      const nextDate = new Date(day);
      nextDate.setHours(current.getHours(), current.getMinutes(), 0, 0);
      return nextDate;
    });
  };

  const handleSelectHour = (hour: number) => {
    updateDate((current) => {
      current.setHours(hour);
      current.setSeconds(0, 0);
      return current;
    });
  };

  const handleSelectMinute = (minute: number) => {
    updateDate((current) => {
      current.setMinutes(minute);
      current.setSeconds(0, 0);
      return current;
    });
  };

  const handleChangeMonth = (offset: number) => {
    setViewDate((current) => new Date(current.getFullYear(), current.getMonth() + offset, 1));
  };

  const handleToday = () => {
    const now = new Date();
    setViewDate(now);
    onChange?.(showTime ? toDateTimeValue(now) : toDateValue(now));
  };

  const handleClear = () => {
    onChange?.('');
    setViewDate(new Date());
  };

  return (
    <div className={className}>
      {label && <Label className="mb-2 block text-sm font-medium">{label}</Label>}

      <div className="flex gap-2">
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              disabled={disabled}
              className={cn(
                'h-10 flex-1 justify-between px-3 text-left font-normal',
                !selectedDate && 'text-muted-foreground',
                inputClassName
              )}
              {...props}
            >
              <span className="truncate">{displayValue}</span>
              <CalendarDays className="size-4 text-muted-foreground" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-auto p-0">
            <div
              className={cn(
                'grid grid-cols-1 overflow-hidden',
                showTime && 'sm:grid-cols-[18rem_4.5rem_4.5rem]'
              )}
            >
              <div className="p-3">
                <div className="mb-3 flex items-center justify-between">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-8"
                    onClick={() => handleChangeMonth(-1)}
                  >
                    <ChevronLeft className="size-4" />
                  </Button>
                  <div className="text-sm font-semibold">{monthLabel}</div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-8"
                    onClick={() => handleChangeMonth(1)}
                  >
                    <ChevronRight className="size-4" />
                  </Button>
                </div>

                <div className="mb-1 grid grid-cols-7 text-center text-sm text-muted-foreground">
                  {weekDays.map((day) => (
                    <div key={day} className="py-1">
                      {day}
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-7 gap-1">
                  {calendarDays.map((day) => {
                    const selected = isSameDay(day, selectedDate);
                    const today = isSameDay(day, new Date());
                    const outsideMonth = day.getMonth() !== viewDate.getMonth();

                    return (
                      <button
                        key={day.toISOString()}
                        type="button"
                        className={cn(
                          'flex size-9 items-center justify-center rounded-md text-sm transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
                          outsideMonth && 'text-muted-foreground/60',
                          today && !selected && 'border border-primary/40 text-primary',
                          selected && 'bg-primary text-primary-foreground hover:bg-primary'
                        )}
                        onClick={() => handleSelectDay(day)}
                      >
                        {day.getDate()}
                      </button>
                    );
                  })}
                </div>

                <div className="mt-3 flex items-center justify-between">
                  <Button variant="ghost" size="sm" onClick={handleClear}>
                    {t('common.clear')}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={handleToday}>
                    {t('common.today')}
                  </Button>
                </div>
              </div>

              {showTime && (
                <>
                  <div className="max-h-[22rem] overflow-y-auto border-l p-2">
                    {HOURS.map((hour) => {
                      const selected = selectedDate?.getHours() === hour;

                      return (
                        <button
                          key={hour}
                          type="button"
                          className={cn(
                            'mb-1 flex h-9 w-full items-center justify-center rounded-md text-sm transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
                            selected && 'bg-primary text-primary-foreground hover:bg-primary'
                          )}
                          onClick={() => handleSelectHour(hour)}
                        >
                          {pad(hour)}
                        </button>
                      );
                    })}
                  </div>

                  <div className="max-h-[22rem] overflow-y-auto border-l p-2">
                    {MINUTES.map((minute) => {
                      const selected = selectedDate?.getMinutes() === minute;

                      return (
                        <button
                          key={minute}
                          type="button"
                          className={cn(
                            'mb-1 flex h-9 w-full items-center justify-center rounded-md text-sm transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
                            selected && 'bg-primary text-primary-foreground hover:bg-primary'
                          )}
                          onClick={() => handleSelectMinute(minute)}
                        >
                          {pad(minute)}
                        </button>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          </PopoverContent>
        </Popover>

        {showClear && value && (
          <Button variant="outline" onClick={handleClear}>
            {t('common.clear')}
          </Button>
        )}
      </div>

      {showSelectedTime && value && (
        <div className="mt-1 text-xs text-muted-foreground">Selected time: {displayValue}</div>
      )}
    </div>
  );
}
