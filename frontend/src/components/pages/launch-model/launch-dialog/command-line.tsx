'use client';

import { FC, useState } from 'react';
import { Copy, Terminal } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useI18n } from '@/contexts/i18n-context';
import { copyToClipboard } from '@/lib/utils';
import type { FormInstance } from '@/types/form';
import {
  generateCommandLineStatement,
  parseXinferenceCommand,
  transformFetchToForm,
  transformFormToFetch,
  validateReplicaPlacement,
} from '../utils';

interface CommandLineProps {
  canCopyCommandLine: boolean;
  form: FormInstance;
}

const CommandLine: FC<CommandLineProps> = ({ canCopyCommandLine, form }) => {
  const [commandLineParsingOpen, setCommandLineParsingOpen] = useState(false);
  const [commandLineParsingValue, setCommandLineParsingValue] = useState('');
  const { t } = useI18n();

  const showCommandLineError = (error: unknown) => {
    toast.error(
      t('launchModel.commandLineParsingFailed', {
        error: error instanceof Error ? error.message : String(error),
      })
    );
  };

  const handleCopyCommandLine = () => {
    if (!canCopyCommandLine) return;

    try {
      const values = form.getFieldsValue();
      const placementError = validateReplicaPlacement(values);
      if (placementError === 'incomplete') {
        toast.error(t('launchModel.replicaPlacementIncomplete'));
        return;
      }
      if (placementError === 'duplicate-alias') {
        toast.error(t('launchModel.replicaAliasDuplicate'));
        return;
      }

      const params = transformFormToFetch(values);

      copyToClipboard(generateCommandLineStatement(params));
    } catch (error) {
      showCommandLineError(error);
    }
  };

  const onOpenChange = (open: boolean) => {
    if (!open) setCommandLineParsingValue('');
    setCommandLineParsingOpen(open);
  };

  const handleClose = () => {
    setCommandLineParsingOpen(false);
    setCommandLineParsingValue('');
  };

  const handleCommandLineParsingConfirm = () => {
    try {
      if (!commandLineParsingValue.trim()) {
        throw new Error('Command line cannot be empty.');
      }

      const params = parseXinferenceCommand(commandLineParsingValue);
      const formData = transformFetchToForm(params);

      // Parsing and conversion finish before this single merged write, so a
      // malformed command never partially overwrites the current form.
      form.setFieldsValue(formData);
      handleClose();
    } catch (error) {
      showCommandLineError(error);
    }
  };

  return (
    <>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="icon"
              aria-label={t('launchModel.commandLineParsing')}
              onClick={() => setCommandLineParsingOpen(true)}
              className="h-8"
            >
              <Terminal />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('launchModel.commandLineParsing')}</TooltipContent>
        </Tooltip>
        {canCopyCommandLine && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="outline"
                size="icon"
                aria-label={t('launchModel.copyToCommandLine')}
                onClick={handleCopyCommandLine}
                className="h-8"
              >
                <Copy />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t('launchModel.copyToCommandLine')}</TooltipContent>
          </Tooltip>
        )}
      </TooltipProvider>
      <Dialog open={commandLineParsingOpen} onOpenChange={onOpenChange}>
        <DialogContent className="!max-w-2xl" showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>{t('launchModel.commandLineParsing')}</DialogTitle>
          </DialogHeader>
          <Textarea
            className="min-h-48"
            placeholder={t('launchModel.placeholderTip')}
            value={commandLineParsingValue}
            onChange={(event) => setCommandLineParsingValue(event.target.value)}
          />
          <DialogFooter>
            <Button variant="outline" onClick={handleClose}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleCommandLineParsingConfirm}>{t('common.confirm')}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default CommandLine;
