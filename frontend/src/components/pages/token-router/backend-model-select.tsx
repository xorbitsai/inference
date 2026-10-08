'use client';

import { Badge } from '@/components/ui/badge';
import { Select, type SelectOption } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import type { TokenRouterBackendCandidate } from '@/types/services';

interface BackendModelSelectProps {
  value?: string;
  onChange?: (value: string | undefined) => void;
  candidates: TokenRouterBackendCandidate[];
  loading?: boolean;
  loadFailed?: boolean;
  candidateErrors?: string[];
  tokenizerCompatibilityVerified?: boolean;
  excludedModelUids?: string[];
  disabled?: boolean;
  error?: boolean;
  placeholder?: string;
}

function compact(values: Array<string | undefined>) {
  return values.filter((value): value is string => Boolean(value?.trim()));
}

export function BackendModelSelect({
  value,
  onChange,
  candidates,
  loading = false,
  loadFailed = false,
  candidateErrors = [],
  tokenizerCompatibilityVerified = true,
  excludedModelUids = [],
  disabled = false,
  error = false,
  placeholder,
}: BackendModelSelectProps) {
  const { t } = useI18n();
  const excluded = new Set(excludedModelUids.filter(Boolean));
  const selectedCandidate = candidates.find((candidate) => candidate.model_uid === value);

  const options: SelectOption<string>[] = candidates.map((candidate) => {
    const alreadySelected = excluded.has(candidate.model_uid);
    const reasons = candidate.eligible
      ? [candidate.compatibility_reason]
      : [...candidate.ineligible_reasons];
    if (alreadySelected) reasons.push(t('tokenRouter.backendAlreadySelected'));

    const description = compact([
      candidate.model_name,
      candidate.model_format,
      candidate.model_ability.length ? candidate.model_ability.join(', ') : undefined,
      candidate.context_length
        ? t('tokenRouter.backendContextValue', { tokens: candidate.context_length })
        : undefined,
      ...reasons,
    ]).join(' · ');

    return {
      value: candidate.model_uid,
      label: `${candidate.model_uid}${candidate.model_engine ? ` · ${candidate.model_engine}` : ''}`,
      description,
      disabled: !candidate.eligible || alreadySelected,
      suffix: (
        <Badge variant={candidate.eligible && !alreadySelected ? 'secondary' : 'destructive'}>
          {candidate.eligible && !alreadySelected
            ? t('tokenRouter.backendCandidateEligible')
            : candidate.compatibility_status}
        </Badge>
      ),
    };
  });

  if (value && !options.some((option) => option.value === value)) {
    options.unshift({
      value,
      label: value,
      description: t('tokenRouter.historicalBackendUnavailable'),
      disabled: true,
      suffix: <Badge variant="destructive">{t('tokenRouter.backendUnavailable')}</Badge>,
    });
  }

  const eligibleCount = options.filter((option) => !option.disabled).length;
  let stateMessage = '';
  let stateClassName = 'text-muted-foreground';
  if (loading) {
    stateMessage = t('tokenRouter.backendCandidatesLoading');
  } else if (loadFailed || candidateErrors.length > 0) {
    stateMessage = candidateErrors.length
      ? `${t('tokenRouter.backendCandidatesLoadFailed')} ${candidateErrors.join('; ')}`
      : t('tokenRouter.backendCandidatesLoadFailed');
    stateClassName = 'text-destructive';
  } else if (candidates.length === 0) {
    stateMessage = t('tokenRouter.noRunningBackendModels');
  } else if (eligibleCount === 0) {
    stateMessage = t('tokenRouter.noCompatibleBackendModels');
  }

  let selectedWarning = '';
  if (value && !selectedCandidate) {
    selectedWarning = t('tokenRouter.historicalBackendUnavailable');
  } else if (selectedCandidate && excluded.has(selectedCandidate.model_uid)) {
    selectedWarning = t('tokenRouter.backendAlreadySelected');
  } else if (selectedCandidate && !selectedCandidate.eligible) {
    selectedWarning = selectedCandidate.ineligible_reasons.join('; ');
  }

  return (
    <div className="space-y-1.5">
      <Select<string>
        value={value}
        onChange={onChange}
        options={options}
        showSearch
        searchPlaceholder={t('tokenRouter.searchBackendModel')}
        allowClear={false}
        disabled={disabled || loading}
        error={error}
        placeholder={placeholder || t('tokenRouter.selectBackendCandidate')}
      />
      {stateMessage && <p className={`text-xs ${stateClassName}`}>{stateMessage}</p>}
      {selectedWarning && <p className="text-xs text-destructive">{selectedWarning}</p>}
      {!tokenizerCompatibilityVerified && (
        <p className="text-xs text-amber-700 dark:text-amber-300">
          {t('tokenRouter.tokenizerCompatibilityUnverified')}
        </p>
      )}
    </div>
  );
}
