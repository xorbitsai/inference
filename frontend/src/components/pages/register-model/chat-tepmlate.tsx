'use client';

import { forwardRef, useImperativeHandle, useState } from 'react';
// Use the browser build instead of `import nunjucks from 'nunjucks'`
// because this component runs on the client side.
// The default package may pull in Node.js-specific dependencies during bundling,
// which can cause webpack/module parse errors in Next.js browser builds.
import nunjucks from 'nunjucks/browser/nunjucks';
import { CircleCheck, CircleX, MoveDiagonal } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import { useI18n } from '@/contexts/i18n-context';
import { MODEL_TEMPLATE_TEST_MSG } from '@/constants/register';

import type { BaseFormFieldProps } from '@/types/form';

const env = new nunjucks.Environment(undefined, {
  autoescape: false,
});
type ChatTemplateProps = BaseFormFieldProps<string>;

interface ChatTemplateState {
  testResult: string;
  isSuccess?: boolean;
}
export interface ChatTemplateMethod {
  resetStatus: () => void;
}
const ChatTemplate = forwardRef<ChatTemplateMethod, ChatTemplateProps>(
  ({ value, ...reset }, ref) => {
    const { t } = useI18n();
    const [{ testResult, isSuccess }, setState] = useState<ChatTemplateState>({
      testResult: '',
      isSuccess: undefined,
    });

    const handleTest = () => {
      if (!value) return;

      try {
        const test_res = env.renderString(value, {
          messages: MODEL_TEMPLATE_TEST_MSG,
        });

        setState({
          testResult: test_res,
          isSuccess: !!test_res,
        });
      } catch (error) {
        setState({
          testResult: String(error),
          isSuccess: false,
        });
      }
    };
    const resetStatus = () => {
      setState({
        testResult: '',
        isSuccess: undefined,
      });
    };
    useImperativeHandle(ref, () => ({
      resetStatus,
    }));

    return (
      <>
        <div className="w-full flex items-start gap-x-[10px]">
          <div className="flex-1">
            <Textarea
              className="flex-1"
              value={value}
              {...reset}
              rows={5}
              onChange={(e) => reset?.onChange?.(e.target.value)}
            />
            <div className="text-xs text-muted-foreground mt-2">
              {t('registerModel.ensureChatTemplatePassesTest')}
            </div>
          </div>

          <Button className="mt-10 shrink-0" onClick={handleTest}>
            {t('registerModel.test')}
          </Button>
          <div className="shrink-0 w-[30%]">
            <div className="border bg-background rounded-md px-[11px] py-[4px] h-[120px] overflow-auto">
              <Dialog>
                <DialogTrigger>
                  <div className="cursor-pointer flex gap-1 items-center group">
                    {t('registerModel.messagesExample')}
                    <MoveDiagonal className="w-4 h-4 text-muted-foreground group-hover:text-foreground" />
                  </div>
                </DialogTrigger>

                <DialogContent showCloseButton={true}>
                  <DialogHeader>
                    <DialogTitle>{t('registerModel.messagesExample')}</DialogTitle>
                  </DialogHeader>
                  <pre className="border border-border rounded-md p-4 whitespace-pre-wrap break-all">
                    {JSON.stringify(MODEL_TEMPLATE_TEST_MSG, null, 2)}
                  </pre>
                </DialogContent>
              </Dialog>

              <div className="flex gap-1 items-center">
                {t('registerModel.testResult')}
                {isSuccess === undefined ? (
                  ''
                ) : isSuccess ? (
                  <CircleCheck className="text-green-500 w-4 h-4" />
                ) : (
                  <CircleX className="text-destructive w-4 h-4" />
                )}
              </div>

              <div>{testResult}</div>
            </div>
            <div
              className={cn('mt-2 text-muted-foreground text-xs', {
                '!text-orange-500': isSuccess === false,
              })}
            >
              {t('registerModel.testFailurePreventsChatWorking')}
            </div>
          </div>
        </div>
      </>
    );
  }
);

ChatTemplate.displayName = 'ChatTemplate';

export default ChatTemplate;
