/* global describe, expect, it */

import { buildApiKeyListState, formatApiKeyDateTime } from './usageState'

const t = (key, values = {}) => {
  const templates = {
    'apikeyManagement.noLimit': 'No limit',
    'apikeyManagement.never': 'Never',
    'apikeyManagement.active': 'Active',
    'apikeyManagement.disabled': 'Disabled',
    'apikeyManagement.expired': 'Expired',
    'apikeyManagement.tokenExhausted': 'Budget exhausted',
    'apikeyManagement.rateLimited': 'Rate limited',
    'apikeyManagement.usedTokens': 'Used: {{used}} / {{total}}',
    'apikeyManagement.usedTokensUnlimited': 'Used: {{used}}',
    'apikeyManagement.remainingTokens': '{{remaining}} remaining',
    'apikeyManagement.renewsAt': 'Renews: {{time}}',
    'apikeyManagement.rateLimitUsage': '{{used}} / {{total}} used',
    'apikeyManagement.rateLimitReset': 'Reset: {{time}}',
    'apikeyManagement.tokenRenewalNone': 'No renewal',
    'apikeyManagement.tokenRenewalDaily': 'Daily',
  }
  return Object.entries(values).reduce(
    (text, [name, value]) => text.replace(`{{${name}}}`, value),
    templates[key] || key
  )
}

describe('api key list usage state', () => {
  it('formats unlimited token budget without marking it exhausted', () => {
    const state = buildApiKeyListState(
      { enabled: true, token_budget: null, token_usage: 12 },
      t
    )

    expect(state.status.label).toBe('Active')
    expect(state.status.color).toBe('success')
    expect(state.tokenUsage.primary).toBe('Used: 12')
    expect(state.tokenUsage.secondary).toBe('No limit')
  })

  it('marks exhausted token budgets distinctly', () => {
    const state = buildApiKeyListState(
      {
        enabled: true,
        token_budget: 100,
        token_usage: 100,
        token_remaining: 0,
      },
      t
    )

    expect(state.status.label).toBe('Budget exhausted')
    expect(state.status.color).toBe('error')
    expect(state.tokenUsage.secondary).toBe('0 remaining')
  })

  it('shows renewal next time when a key renews', () => {
    const state = buildApiKeyListState(
      {
        enabled: true,
        token_budget: 100,
        token_usage: 10,
        token_remaining: 90,
        token_renewal: 'daily',
        token_renewal_next_at: '2026-01-02T03:04:05',
      },
      t
    )

    expect(state.renewal.primary).toBe('Daily')
    expect(state.renewal.secondary).toBe('Renews: 2026-01-02 03:04:05')
  })

  it('marks expired and rate-limited keys distinctly', () => {
    const expired = buildApiKeyListState(
      {
        enabled: true,
        expires_at: '2026-01-01T00:00:00',
      },
      t,
      new Date('2026-01-02T00:00:00')
    )
    const limited = buildApiKeyListState(
      {
        enabled: true,
        request_rate_limit_enabled: true,
        request_rate_limit_requests: 5,
        request_rate_limit_count: 5,
        request_rate_limit_remaining: 0,
        request_rate_limit_reset_at: '2026-01-02T00:01:00',
      },
      t,
      new Date('2026-01-01T00:00:00')
    )

    expect(expired.status.label).toBe('Expired')
    expect(expired.status.color).toBe('warning')
    expect(limited.status.label).toBe('Rate limited')
    expect(limited.status.color).toBe('info')
    expect(limited.rateLimit.secondary).toBe('Reset: 2026-01-02 00:01:00')
  })

  it('formats rotation and empty date state for the list', () => {
    expect(formatApiKeyDateTime('2026-01-02T03:04:05')).toBe(
      '2026-01-02 03:04:05'
    )
    expect(buildApiKeyListState({ rotated_at: null }, t).rotation.primary).toBe(
      'Never'
    )
  })
})
