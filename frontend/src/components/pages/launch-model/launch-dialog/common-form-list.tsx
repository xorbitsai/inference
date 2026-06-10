'use client';

import { FC } from 'react';
import { Trash2, Plus } from 'lucide-react';
import { FormList } from '@/components/ui/form-list';
import { FormField } from '@/components/ui/form-field';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { AutoComplete, type AutoCompleteOption } from '@/components/ui/auto-complete';
import { useI18n } from '@/contexts/i18n-context';

export interface CommonFormListProps {
  name: string | string[];
  label: string;
  childFirstKey?: string;
  childSecondKey?: string;
  onlyValue?: boolean;
  keyOptions?: AutoCompleteOption[];
}
const CommonFormList: FC<CommonFormListProps> = ({
  name,
  label,
  childFirstKey = 'key',
  childSecondKey = 'value',
  onlyValue = false,
  keyOptions,
}) => {
  const { t } = useI18n();

  return (
    <FormList name={name} layout="horizontal">
      {({ fields, add, remove }) => {
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="flex min-w-0 flex-1 items-center gap-3 text-left">
                <span className="min-w-0 flex-1 truncate text-sm font-medium">{label}</span>
                <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                  {t('common.itemsCount', { count: fields.length })}
                </span>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8"
                onClick={() => add('')}
              >
                <Plus />
                {t('common.add')}
              </Button>
            </div>

            {fields.map((field) => (
              <div className="flex gap-2" key={field.name}>
                {!onlyValue && (
                  <FormField
                    className="flex-1"
                    rules={[{ required: true }]}
                    name={[...(Array.isArray(name) ? name : [name]), field.name, childFirstKey]}
                    placeholder={childFirstKey}
                  >
                    {Array.isArray(keyOptions) ? (
                      <AutoComplete
                        optionsTips={t('common.autoCompleteOptionsTips')}
                        options={keyOptions}
                      />
                    ) : (
                      <Input />
                    )}
                  </FormField>
                )}

                <FormField
                  className="flex-1"
                  rules={[{ required: true }]}
                  name={[...(Array.isArray(name) ? name : [name]), field.name, childSecondKey]}
                  placeholder={childSecondKey}
                >
                  <Input />
                </FormField>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 rounded-full text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                  onClick={() => remove(field.name)}
                >
                  <Trash2 />
                </Button>
              </div>
            ))}
          </div>
        );
      }}
    </FormList>
  );
};
export default CommonFormList;
