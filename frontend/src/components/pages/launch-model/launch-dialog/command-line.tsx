'use client';

import { FC, useState } from 'react';
import { Terminal, Copy } from 'lucide-react';
import { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogHeader,
  DialogContent,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { useI18n } from '@/contexts/i18n-context';
import { copyText } from '@/lib/utils';
import type { FormInstance } from '@/types/form';
import { transformFormToFetch, generateCommandLineStatement, transformFetchToForm, parseXinferenceCommand } from '../utils';

interface CommandLineProps {
  canCopyCommandLine: boolean;
  form: FormInstance;
}
const CommandLine: FC<CommandLineProps> = ({ canCopyCommandLine, form }) => {
  const [commandLineParsingOpen, setCommandLineParsingOpen] = useState(false);
  const [commandLineParsingValue, setCommandLineParsingValue] = useState('');
  const { t } = useI18n();
  const handleCopyCommandLine = () => {
    if (!canCopyCommandLine) return;

    const params = transformFormToFetch(form.getFieldsValue());

    copyText(generateCommandLineStatement(params));
  };
  const onOpenChange = (open: boolean)=>{
    if(!open) setCommandLineParsingValue('');
    setCommandLineParsingOpen(open)
  }
  const handleClose = () => {
    setCommandLineParsingOpen(false);
    setCommandLineParsingValue('');
  };
  const handleCommandLineParsingConfirm = () => {
    if(!commandLineParsingValue) {
      setCommandLineParsingOpen(false);
      return;
    }
    const params = parseXinferenceCommand(commandLineParsingValue);
    const formData = transformFetchToForm(params);
    form.setFieldsValue(formData);
    setCommandLineParsingOpen(false)
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
              className='h-8'
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
                className='h-8'
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
            <Button  onClick={handleCommandLineParsingConfirm}>
              {t('common.confirm')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
export default CommandLine;
