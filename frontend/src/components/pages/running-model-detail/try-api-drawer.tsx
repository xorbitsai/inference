'use client';

import { useMemo, useState } from 'react';
import { Copy, ExternalLink, X } from 'lucide-react';
import { useRouter } from 'next/navigation';

import { Button } from '@/components/ui/button';
import { JSONSyntaxHighlighter } from '@/components/ui/json-syntax-highlighter';
import { Sheet, SheetClose, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ModelAbility } from '@/constants';
import { copyText, getApiUrl } from '@/lib/utils';

import { CAPABILITY_CONFIGS } from './capability-config';
import { CHAT_CODE_EXAMPLE, CODE_LANGUAGE_OPTIONS, generateCodeExample } from './api-code-example';
import type { CodeExampleConfig } from './types';
import { useMenuAuth } from '@/hooks/use-menu-auth';

interface TryApiDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelUid?: string;
  ability?: ModelAbility;
}

function getCodeExample(
  ability?: ModelAbility
): { requestApi: string; config: CodeExampleConfig } | null {
  if (ability === ModelAbility.Chat) {
    return {
      requestApi: '/v1/chat/completions',
      config: CHAT_CODE_EXAMPLE,
    };
  }

  if (!ability) return null;

  const capabilityConfig = CAPABILITY_CONFIGS[ability];
  if (!capabilityConfig?.codeExample) return null;

  return {
    requestApi: capabilityConfig.requestApi,
    config: capabilityConfig.codeExample,
  };
}

export function getTryApiAbility(abilities: ModelAbility[] = []) {
  const primaryAbilities = abilities.filter((ability) => !ability.includes('_'));
  return (
    primaryAbilities.find((ability) => ability === ModelAbility.Chat) ||
    primaryAbilities.find((ability) => !!CAPABILITY_CONFIGS[ability]?.codeExample)
  );
}

export function TryApiDrawer({
  open,
  onOpenChange,
  modelUid = '{MODEL_UID}',
  ability,
}: TryApiDrawerProps) {
  const router = useRouter();
  const { keysManagePage } = useMenuAuth();
  const [language, setLanguage] = useState(CODE_LANGUAGE_OPTIONS[0].value);
  const codeExample = getCodeExample(ability);
  const url = codeExample ? `${getApiUrl()}${codeExample.requestApi}` : '';
  const code = useMemo(() => {
    if (!codeExample) return '';
    return generateCodeExample(language, codeExample.config, url, modelUid);
  }, [codeExample, language, modelUid, url]);
  const activeLanguage = CODE_LANGUAGE_OPTIONS.find((item) => item.value === language);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent showClose={false} className="w-[min(50vw,760px)] gap-0 p-0 sm:max-w-none">
        <SheetHeader className="flex-row items-center justify-between gap-4 border-b px-6 py-5">
          <div className="flex min-w-0 items-center gap-3">
            <SheetClose asChild>
              <Button
                variant="ghost"
                size="icon"
                className="size-9 rounded-full text-muted-foreground"
              >
                <X className="size-5" />
              </Button>
            </SheetClose>
            <SheetTitle className="truncate text-xl">Try To API</SheetTitle>
          </div>
          <Button
            type="button"
            variant="outline"
            disabled={!keysManagePage}
            onClick={() => router.push('/api-key-management')}
          >
            <ExternalLink className="size-4" />
            Get API Key
          </Button>
        </SheetHeader>
        <div className="flex-1 overflow-y-auto p-6">
          {codeExample ? (
            <Tabs value={language} onValueChange={(value) => setLanguage(value as typeof language)}>
              <TabsList className="grid h-11 w-full grid-cols-5 rounded-xl p-1">
                {CODE_LANGUAGE_OPTIONS.map((item) => (
                  <TabsTrigger
                    key={item.value}
                    value={item.value}
                    className="p-0 rounded-md text-base data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
                  >
                    {item.label}
                  </TabsTrigger>
                ))}
              </TabsList>
              {CODE_LANGUAGE_OPTIONS.map((item) => (
                <TabsContent key={item.value} value={item.value} className="mt-6">
                  <div className="relative">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="absolute right-3 top-3 z-10 size-8 text-muted-foreground"
                      onClick={() => copyText(code)}
                    >
                      <Copy className="size-4" />
                    </Button>
                    <JSONSyntaxHighlighter
                      code={code}
                      language={activeLanguage?.highlight || 'python'}
                      className="rounded-xl pr-12 text-base whitespace-pre-wrap break-all"
                    />
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          ) : (
            <div className="flex min-h-80 items-center justify-center rounded-2xl border bg-card text-sm text-muted-foreground">
              No API example available for this model ability.
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
