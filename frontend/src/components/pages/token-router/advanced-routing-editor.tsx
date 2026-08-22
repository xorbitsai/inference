'use client';

import type { ReactNode } from 'react';
import { Plus, Trash2 } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import type {
  TokenRouterBackendCandidate,
  TokenRouterDynamicBackendConfig,
  TokenRouterRoutingAction,
  TokenRouterRoutingRule,
} from '@/types/services';

import { BackendModelSelect } from './backend-model-select';
import type { TypedRouterDraft } from './router-config-normalizer';

interface Props {
  value: TypedRouterDraft;
  candidates: TokenRouterBackendCandidate[];
  candidatesLoading?: boolean;
  candidatesLoadFailed?: boolean;
  candidateErrors?: string[];
  tokenizerCompatibilityVerified?: boolean;
  onChange: (value: TypedRouterDraft) => void;
}

const toNumber = (value: string, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const conditionOptions = (anyLabel: string, trueLabel: string, falseLabel: string) => [
  { value: 'any', label: anyLabel },
  { value: 'true', label: trueLabel },
  { value: 'false', label: falseLabel },
];

export function AdvancedRoutingEditor({
  value,
  candidates,
  candidatesLoading = false,
  candidatesLoadFailed = false,
  candidateErrors = [],
  tokenizerCompatibilityVerified = true,
  onChange,
}: Props) {
  const { t } = useI18n();
  const backendIds = value.backends.map((backend) => backend.id).filter(Boolean);

  const updateBackend = (index: number, patch: Partial<TokenRouterDynamicBackendConfig>) => {
    const backends = value.backends.map((backend, itemIndex) =>
      itemIndex === index ? { ...backend, ...patch } : backend
    );
    onChange({ ...value, backends });
  };

  const removeBackend = (index: number) => {
    if (value.backends.length <= 1) return;
    onChange({ ...value, backends: value.backends.filter((_, itemIndex) => itemIndex !== index) });
  };

  const addBackend = () => {
    const suffix = value.backends.length + 1;
    onChange({
      ...value,
      backends: [
        ...value.backends,
        {
          id: `backend-${suffix}`,
          model_uid: '',
          max_context_tokens: 131072,
          admission: {
            max_active: 1,
            max_queue: 4,
            queue_timeout_seconds: 5,
            retry_after_seconds: 1,
          },
        },
      ],
    });
  };

  const updateRule = (index: number, patch: Partial<TokenRouterRoutingRule>) => {
    const rules = value.rules.map((rule, itemIndex) =>
      itemIndex === index ? { ...rule, ...patch } : rule
    );
    onChange({ ...value, rules });
  };

  const updateRuleMatch = (
    index: number,
    key: keyof TokenRouterRoutingRule['match'],
    fieldValue: number | boolean | undefined
  ) => {
    const rule = value.rules[index];
    const match = { ...rule.match };
    if (fieldValue === undefined) delete match[key];
    else Object.assign(match, { [key]: fieldValue });
    updateRule(index, { match });
  };

  const addRule = () => {
    const suffix = value.rules.length + 1;
    onChange({
      ...value,
      rules: [
        ...value.rules,
        {
          id: `rule-${suffix}`,
          priority: Math.max(1, 100 - value.rules.length * 10),
          match: { total_tokens_gte: 0 },
          action: { type: 'route', backend_id: backendIds[0] || '' },
        },
      ],
    });
  };

  const removeRule = (index: number) => {
    if (value.rules.length <= 1) return;
    onChange({ ...value, rules: value.rules.filter((_, itemIndex) => itemIndex !== index) });
  };

  const updateDefaultAction = (action: TokenRouterRoutingAction) =>
    onChange({ ...value, defaultAction: action });

  return (
    <div className="space-y-5 md:col-span-2">
      <div className="grid gap-3 rounded-lg border bg-muted/20 p-4 sm:grid-cols-2 lg:grid-cols-5">
        <ProfileItem label={t('tokenRouter.profile.modelType')} value="LLM" />
        <ProfileItem label={t('tokenRouter.profile.task')} value="Chat" />
        <ProfileItem label={t('tokenRouter.profile.protocol')} value="OpenAI Chat Completions" />
        <ProfileItem label={t('tokenRouter.profile.endpoint')} value="/v1/chat/completions" mono />
        <ProfileItem label={t('tokenRouter.profile.strategy')} value="Typed Rules" />
      </div>

      <section className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h4 className="text-sm font-semibold">{t('tokenRouter.dynamicBackends')}</h4>
            <p className="text-xs text-muted-foreground">{t('tokenRouter.dynamicBackendsHint')}</p>
          </div>
          <Button type="button" size="sm" variant="outline" onClick={addBackend}>
            <Plus className="mr-1 size-4" />
            {t('tokenRouter.addBackend')}
          </Button>
        </div>
        <div className="space-y-3">
          {value.backends.map((backend, index) => {
            const candidate = candidates.find((item) => item.model_uid === backend.model_uid);
            return (
              <div key={`${backend.id}-${index}`} className="rounded-lg border p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <Badge variant="outline">#{index + 1}</Badge>
                    <span className="truncate font-mono text-sm">{backend.id || '—'}</span>
                    {candidate && (
                      <Badge variant={candidate.eligible ? 'secondary' : 'destructive'}>
                        {candidate.compatibility_status}
                      </Badge>
                    )}
                  </div>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    disabled={value.backends.length <= 1}
                    onClick={() => removeBackend(index)}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
                <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                  <EditorField label={t('tokenRouter.backendId')}>
                    <Input
                      value={backend.id}
                      onChange={(event) => updateBackend(index, { id: event.target.value })}
                    />
                  </EditorField>
                  <EditorField label={t('tokenRouter.backendModel')}>
                    <BackendModelSelect
                      value={backend.model_uid}
                      candidates={candidates}
                      loading={candidatesLoading}
                      loadFailed={candidatesLoadFailed}
                      candidateErrors={candidateErrors}
                      tokenizerCompatibilityVerified={tokenizerCompatibilityVerified}
                      onChange={(modelUid) =>
                        updateBackend(index, { model_uid: String(modelUid || '') })
                      }
                    />
                  </EditorField>
                  <EditorField label={t('tokenRouter.maxContext')}>
                    <Input
                      type="number"
                      min={1}
                      value={backend.max_context_tokens}
                      onChange={(event) =>
                        updateBackend(index, {
                          max_context_tokens: toNumber(event.target.value, 1),
                        })
                      }
                    />
                  </EditorField>
                  <EditorField label={t('tokenRouter.maxActive')}>
                    <Input
                      type="number"
                      min={1}
                      value={backend.admission.max_active}
                      onChange={(event) =>
                        updateBackend(index, {
                          admission: {
                            ...backend.admission,
                            max_active: toNumber(event.target.value, 1),
                          },
                        })
                      }
                    />
                  </EditorField>
                  <EditorField label={t('tokenRouter.maxQueue')}>
                    <Input
                      type="number"
                      min={0}
                      value={backend.admission.max_queue}
                      onChange={(event) =>
                        updateBackend(index, {
                          admission: {
                            ...backend.admission,
                            max_queue: toNumber(event.target.value),
                          },
                        })
                      }
                    />
                  </EditorField>
                  <div className="rounded-md bg-muted/30 p-2 text-xs text-muted-foreground">
                    {candidate ? (
                      <>
                        <div>{candidate.model_name || '—'}</div>
                        <div>{candidate.model_format || '—'}</div>
                        <div className="break-words">
                          {candidate.model_ability.join(', ') || '—'}
                        </div>
                      </>
                    ) : (
                      t('tokenRouter.selectBackendCandidate')
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h4 className="text-sm font-semibold">{t('tokenRouter.orderedRules')}</h4>
            <p className="text-xs text-muted-foreground">{t('tokenRouter.firstMatchHint')}</p>
          </div>
          <Button type="button" size="sm" variant="outline" onClick={addRule}>
            <Plus className="mr-1 size-4" />
            {t('tokenRouter.addRule')}
          </Button>
        </div>
        <div className="space-y-3">
          {[...value.rules]
            .map((rule, sourceIndex) => ({ rule, sourceIndex }))
            .sort((left, right) => right.rule.priority - left.rule.priority)
            .map(({ rule, sourceIndex }, displayIndex) => (
              <div key={`${rule.id}-${sourceIndex}`} className="rounded-lg border p-4">
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge>{displayIndex + 1}</Badge>
                    <span className="font-mono text-sm">{rule.id || '—'}</span>
                  </div>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    disabled={value.rules.length <= 1}
                    onClick={() => removeRule(sourceIndex)}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
                <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                  <EditorField label={t('tokenRouter.ruleId')}>
                    <Input
                      value={rule.id}
                      onChange={(event) => updateRule(sourceIndex, { id: event.target.value })}
                    />
                  </EditorField>
                  <EditorField label={t('tokenRouter.priority')}>
                    <Input
                      type="number"
                      min={1}
                      max={10000}
                      value={rule.priority}
                      onChange={(event) =>
                        updateRule(sourceIndex, { priority: toNumber(event.target.value, 1) })
                      }
                    />
                  </EditorField>
                  <EditorField label={t('tokenRouter.tokenGte')}>
                    <Input
                      type="number"
                      min={0}
                      value={rule.match.total_tokens_gte ?? ''}
                      onChange={(event) =>
                        updateRuleMatch(
                          sourceIndex,
                          'total_tokens_gte',
                          event.target.value === '' ? undefined : toNumber(event.target.value)
                        )
                      }
                    />
                  </EditorField>
                  <EditorField label={t('tokenRouter.tokenLte')}>
                    <Input
                      type="number"
                      min={0}
                      value={rule.match.total_tokens_lte ?? ''}
                      onChange={(event) =>
                        updateRuleMatch(
                          sourceIndex,
                          'total_tokens_lte',
                          event.target.value === '' ? undefined : toNumber(event.target.value)
                        )
                      }
                    />
                  </EditorField>
                  {(['thinking', 'tools_present', 'stream'] as const).map((key) => (
                    <EditorField key={key} label={t(`tokenRouter.conditions.${key}`)}>
                      <Select
                        allowClear={false}
                        value={rule.match[key] === undefined ? 'any' : String(rule.match[key])}
                        options={conditionOptions(
                          t('tokenRouter.conditionValues.any'),
                          t('tokenRouter.conditionValues.true'),
                          t('tokenRouter.conditionValues.false')
                        )}
                        onChange={(selected) =>
                          updateRuleMatch(
                            sourceIndex,
                            key,
                            selected === 'any' ? undefined : selected === 'true'
                          )
                        }
                      />
                    </EditorField>
                  ))}
                  <EditorField label={t('tokenRouter.actionType')}>
                    <Select
                      allowClear={false}
                      value={rule.action.type}
                      options={[
                        { value: 'route', label: t('tokenRouter.actionTypes.route') },
                        { value: 'reject', label: t('tokenRouter.actionTypes.reject') },
                      ]}
                      onChange={(actionType) =>
                        updateRule(sourceIndex, {
                          action:
                            actionType === 'reject'
                              ? { type: 'reject', reason: 'request_rejected' }
                              : { type: 'route', backend_id: backendIds[0] || '' },
                        })
                      }
                    />
                  </EditorField>
                  {rule.action.type === 'route' ? (
                    <EditorField label={t('tokenRouter.targetBackend')}>
                      <Select
                        allowClear={false}
                        value={rule.action.backend_id}
                        options={backendIds.map((backendId) => ({
                          value: backendId,
                          label: backendId,
                        }))}
                        onChange={(backendId) =>
                          updateRule(sourceIndex, {
                            action: { type: 'route', backend_id: String(backendId || '') },
                          })
                        }
                      />
                    </EditorField>
                  ) : (
                    <EditorField label={t('tokenRouter.rejectReason')}>
                      <Input
                        value={rule.action.reason}
                        onChange={(event) =>
                          updateRule(sourceIndex, {
                            action: { type: 'reject', reason: event.target.value },
                          })
                        }
                      />
                    </EditorField>
                  )}
                </div>
              </div>
            ))}
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <h4 className="mb-3 text-sm font-semibold">{t('tokenRouter.defaultAction')}</h4>
        <div className="grid gap-3 md:grid-cols-2">
          <EditorField label={t('tokenRouter.actionType')}>
            <Select
              allowClear={false}
              value={value.defaultAction.type}
              options={[
                { value: 'route', label: t('tokenRouter.actionTypes.route') },
                { value: 'reject', label: t('tokenRouter.actionTypes.reject') },
              ]}
              onChange={(actionType) =>
                updateDefaultAction(
                  actionType === 'route'
                    ? { type: 'route', backend_id: backendIds[0] || '' }
                    : { type: 'reject', reason: 'context_length_exceeded' }
                )
              }
            />
          </EditorField>
          {value.defaultAction.type === 'route' ? (
            <EditorField label={t('tokenRouter.targetBackend')}>
              <Select
                allowClear={false}
                value={value.defaultAction.backend_id}
                options={backendIds.map((backendId) => ({ value: backendId, label: backendId }))}
                onChange={(backendId) =>
                  updateDefaultAction({ type: 'route', backend_id: String(backendId || '') })
                }
              />
            </EditorField>
          ) : (
            <EditorField label={t('tokenRouter.rejectReason')}>
              <Input
                value={value.defaultAction.reason}
                onChange={(event) =>
                  updateDefaultAction({ type: 'reject', reason: event.target.value })
                }
              />
            </EditorField>
          )}
        </div>
      </section>
    </div>
  );
}

function EditorField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="space-y-1.5 text-xs font-medium">
      <span>{label}</span>
      {children}
    </label>
  );
}

function ProfileItem({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={mono ? 'mt-1 break-all font-mono text-xs' : 'mt-1 text-sm font-medium'}>
        {value}
      </div>
    </div>
  );
}
