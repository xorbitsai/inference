const numberOrNull = (value) => {
  if (value === null || value === undefined || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const formatInteger = (value) => Number(value || 0).toLocaleString()

export const formatApiKeyDateTime = (value) => {
  if (!value) return ''
  const text = String(value).replace('T', ' ')
  return text.replace(/\.\d+(?=Z?$)/, '').replace(/Z$/, '')
}

const isExpired = (row, now) => {
  if (!row.expires_at) return false
  const expiresAt = new Date(row.expires_at)
  return !Number.isNaN(expiresAt.getTime()) && expiresAt < now
}

const isTokenExhausted = (row) => {
  if (row.token_budget_exhausted !== undefined) {
    return Boolean(row.token_budget_exhausted)
  }
  const budget = numberOrNull(row.token_budget)
  const usage = numberOrNull(row.token_usage) || 0
  return budget !== null && usage >= budget
}

const isRequestRateLimited = (row) =>
  Boolean(
    row.request_rate_limit_enabled &&
      numberOrNull(row.request_rate_limit_remaining) === 0
  )

const statusForRow = (row, t, now) => {
  if (!row.enabled) {
    return {
      key: 'disabled',
      label: t('apikeyManagement.disabled'),
      color: 'default',
    }
  }
  if (isExpired(row, now)) {
    return {
      key: 'expired',
      label: t('apikeyManagement.expired'),
      color: 'warning',
    }
  }
  if (isTokenExhausted(row)) {
    return {
      key: 'token_exhausted',
      label: t('apikeyManagement.tokenExhausted'),
      color: 'error',
    }
  }
  if (isRequestRateLimited(row)) {
    return {
      key: 'rate_limited',
      label: t('apikeyManagement.rateLimited'),
      color: 'info',
    }
  }
  return {
    key: 'active',
    label: t('apikeyManagement.active'),
    color: 'success',
  }
}

const tokenUsageForRow = (row, t) => {
  const budget = numberOrNull(row.token_budget)
  const usage = numberOrNull(row.token_usage) || 0
  if (budget === null) {
    return {
      primary: t('apikeyManagement.usedTokensUnlimited', {
        used: formatInteger(usage),
      }),
      secondary: t('apikeyManagement.noLimit'),
      color: 'default',
    }
  }
  const remaining =
    numberOrNull(row.token_remaining) === null
      ? Math.max(0, budget - usage)
      : numberOrNull(row.token_remaining)
  return {
    primary: t('apikeyManagement.usedTokens', {
      used: formatInteger(usage),
      total: formatInteger(budget),
    }),
    secondary: t('apikeyManagement.remainingTokens', {
      remaining: formatInteger(remaining),
    }),
    color: remaining === 0 ? 'error' : 'default',
  }
}

const renewalForRow = (row, t) => {
  const renewal = row.token_renewal || 'none'
  const labelKey =
    {
      none: 'tokenRenewalNone',
      daily: 'tokenRenewalDaily',
      monthly: 'tokenRenewalMonthly',
      custom: 'tokenRenewalCustom',
    }[renewal] || 'tokenRenewalNone'
  const secondary = row.token_renewal_next_at
    ? t('apikeyManagement.renewsAt', {
        time: formatApiKeyDateTime(row.token_renewal_next_at),
      })
    : ''
  return {
    primary: t(`apikeyManagement.${labelKey}`),
    secondary,
  }
}

const rateLimitForRow = (row, t) => {
  if (!row.request_rate_limit_enabled) {
    return {
      primary: t('apikeyManagement.noLimit'),
      secondary: '',
      color: 'default',
    }
  }
  const total = numberOrNull(row.request_rate_limit_requests) || 0
  const used = numberOrNull(row.request_rate_limit_count) || 0
  const remaining = numberOrNull(row.request_rate_limit_remaining)
  const secondary = row.request_rate_limit_reset_at
    ? t('apikeyManagement.rateLimitReset', {
        time: formatApiKeyDateTime(row.request_rate_limit_reset_at),
      })
    : remaining === null
    ? ''
    : t('apikeyManagement.remainingRequests', {
        remaining: formatInteger(remaining),
      })
  return {
    primary: t('apikeyManagement.rateLimitUsage', {
      used: formatInteger(used),
      total: formatInteger(total),
    }),
    secondary,
    color: remaining === 0 ? 'warning' : 'default',
  }
}

export const buildApiKeyListState = (row, t, now = new Date()) => ({
  status: statusForRow(row, t, now),
  expiration: {
    primary: row.expires_at
      ? formatApiKeyDateTime(row.expires_at)
      : t('apikeyManagement.never'),
  },
  rotation: {
    primary: row.rotated_at
      ? formatApiKeyDateTime(row.rotated_at)
      : t('apikeyManagement.never'),
  },
  tokenUsage: tokenUsageForRow(row, t),
  renewal: renewalForRow(row, t),
  rateLimit: rateLimitForRow(row, t),
})
