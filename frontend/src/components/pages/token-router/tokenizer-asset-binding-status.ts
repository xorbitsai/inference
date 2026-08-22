import type { TFunc } from '@/contexts/i18n-context';
import type { TokenizerAssetBinding } from '@/types/services';

type BindingDesiredState = TokenizerAssetBinding['desired_state'];
type BadgeVariant = 'default' | 'secondary' | 'destructive' | 'outline';

function translateBindingState(t: TFunc, prefix: string, state: string): string {
  const key = `${prefix}.${state}`;
  const translated = t(key);
  return translated === key ? state : translated;
}

export function getBindingDesiredStateLabel(t: TFunc, state: BindingDesiredState): string {
  return translateBindingState(t, 'tokenRouter.bindingDesiredStates', state);
}

export function getBindingObservedStateLabel(t: TFunc, state: string): string {
  return translateBindingState(t, 'tokenRouter.bindingObservedStates', state);
}

export function getBindingStatusLabel(
  t: TFunc,
  desiredState: BindingDesiredState,
  observedState: string
): string {
  return `${getBindingDesiredStateLabel(t, desiredState)} / ${getBindingObservedStateLabel(
    t,
    observedState
  )}`;
}

export function getBindingStatusTitle(
  t: TFunc,
  desiredState: BindingDesiredState,
  observedState: string
): string {
  return `${t('tokenRouter.bindingDesiredState')}: ${getBindingDesiredStateLabel(
    t,
    desiredState
  )}\n${t('tokenRouter.bindingObservedState')}: ${getBindingObservedStateLabel(t, observedState)}`;
}

export function getBindingObservedStateBadgeVariant(state: string): BadgeVariant {
  if (state === 'ready') return 'default';
  if (state === 'error') return 'destructive';
  if (state === 'stale' || state === 'unavailable') return 'outline';
  return 'secondary';
}
