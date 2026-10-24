'use client';

import { type ReactNode, useId } from 'react';
import { AlertCircle, ArrowUpRight, Check, ChevronDown, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { useI18n } from '@/contexts/i18n-context';

interface ComponentRestartDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  icon: ReactNode;
  target?: string | null;
  python: string;
  onPythonChange: (value: string) => void;
  environmentHint: string;
  busy: boolean;
  canSubmit?: boolean;
  error: string;
  onSubmit: () => Promise<void>;
  onCloseAutoFocus?: (event: Event) => void;
}

export function ComponentRestartDialog({
  open,
  onOpenChange,
  title,
  description,
  icon,
  target,
  python,
  onPythonChange,
  environmentHint,
  busy,
  canSubmit = true,
  error,
  onSubmit,
  onCloseAutoFocus,
}: ComponentRestartDialogProps) {
  const { t } = useI18n();
  const formId = useId();
  const prefix = 'clusterInfo.relaunch';
  const changingEnvironment = Boolean(python.trim());

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-h-[90vh] max-w-[calc(100%-2rem)] gap-5 sm:max-w-xl"
        onCloseAutoFocus={onCloseAutoFocus}
      >
        <DialogHeader className="gap-3 pr-7 text-left">
          <div className="flex items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary [&>svg]:size-5">
              {icon}
            </div>
            <div className="min-w-0 space-y-1">
              <DialogTitle className="text-lg">{title}</DialogTitle>
              {target && (
                <p className="truncate font-mono text-xs text-muted-foreground" title={target}>
                  {target}
                </p>
              )}
            </div>
          </div>
          <DialogDescription className="text-sm leading-6">{description}</DialogDescription>
        </DialogHeader>
        <form
          id={formId}
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            void onSubmit();
          }}
        >
          <div className="overflow-hidden rounded-xl border">
            <div className="flex items-start gap-3 bg-muted/30 p-4">
              <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                {changingEnvironment ? (
                  <ArrowUpRight className="size-4" />
                ) : (
                  <Check className="size-4" />
                )}
              </div>
              <div className="min-w-0 space-y-1">
                <p className="text-sm font-medium">
                  {t(
                    `${prefix}.${changingEnvironment ? 'preparedEnvironment' : 'currentEnvironment'}`
                  )}
                </p>
                <p
                  className="truncate text-xs leading-5 text-muted-foreground"
                  title={changingEnvironment ? python.trim() : undefined}
                >
                  {python.trim() || t(`${prefix}.currentEnvironmentHint`)}
                </p>
              </div>
            </div>
            <details className="group border-t">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm outline-none hover:bg-muted/20 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
                {t(`${prefix}.changeEnvironment`)}
                <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
              </summary>
              <div className="space-y-3 border-t p-4">
                <label htmlFor={`${formId}-python`} className="block text-sm font-medium">
                  {t(`${prefix}.python`)}
                </label>
                <Input
                  id={`${formId}-python`}
                  aria-label={t(`${prefix}.python`)}
                  className="font-mono text-xs"
                  value={python}
                  disabled={busy}
                  onChange={(event) => onPythonChange(event.target.value)}
                  placeholder="/opt/xinference/bin/python"
                />
                <p className="text-xs leading-5 text-muted-foreground">{environmentHint}</p>
              </div>
            </details>
          </div>
          {error && (
            <div
              role="alert"
              className="flex items-start gap-3 rounded-xl border border-destructive/20 bg-destructive/5 p-4 text-sm text-destructive"
            >
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              <p className="min-w-0 break-words leading-6 [overflow-wrap:anywhere]">{error}</p>
            </div>
          )}
        </form>
        <DialogFooter className="border-t pt-4">
          <Button variant="outline" disabled={busy} onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button type="submit" form={formId} aria-label={title} disabled={busy || !canSubmit}>
            {busy && <Loader2 className="size-4 animate-spin" />}
            {t(`${prefix}.${busy ? 'submitting' : changingEnvironment ? 'upgrade' : 'restart'}`)}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
