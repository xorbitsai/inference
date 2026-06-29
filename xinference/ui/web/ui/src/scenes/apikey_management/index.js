import AutorenewIcon from '@mui/icons-material/Autorenew'
import BlockIcon from '@mui/icons-material/Block'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import DeleteIcon from '@mui/icons-material/Delete'
import EditIcon from '@mui/icons-material/Edit'
import {
  Alert,
  Autocomplete,
  Box,
  Checkbox,
  Chip,
  FormControlLabel,
  FormGroup,
  IconButton,
  InputAdornment,
  Snackbar,
  Tooltip,
} from '@mui/material'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid } from '@mui/x-data-grid'
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'
import { DatePicker } from '@mui/x-date-pickers/DatePicker'
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider'
import dayjs from 'dayjs'
import * as React from 'react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import fetchWrapper from '../../components/fetchWrapper'
import Title from '../../components/Title'
import { buildApiKeyListState } from './usageState'

const MODEL_TYPES = ['LLM', 'embedding', 'rerank', 'image', 'audio', 'video']
const TOKEN_RENEWAL_OPTIONS = ['none', 'daily', 'monthly', 'custom']
const TOKEN_RENEWAL_LABEL_KEYS = {
  none: 'tokenRenewalNone',
  daily: 'tokenRenewalDaily',
  monthly: 'tokenRenewalMonthly',
  custom: 'tokenRenewalCustom',
}

const optionalNumber = (value) => {
  const trimmed = String(value || '').trim()
  return trimmed === '' ? undefined : Number(trimmed)
}

const numberText = (value) =>
  value === null || value === undefined ? '' : String(value)

const buildUsageControlPayload = ({
  tokenBudget,
  tokenRenewal,
  tokenRenewalIntervalDays,
  requestRateLimitEnabled,
  requestRateLimitRequests,
  requestRateLimitWindowSeconds,
}) => {
  const payload = {
    token_budget: optionalNumber(tokenBudget),
    token_renewal: tokenRenewal || 'none',
    request_rate_limit_enabled: requestRateLimitEnabled,
  }

  if (tokenRenewal === 'custom') {
    payload.token_renewal_interval_days = optionalNumber(
      tokenRenewalIntervalDays
    )
  }
  if (requestRateLimitEnabled) {
    payload.request_rate_limit_requests = optionalNumber(
      requestRateLimitRequests
    )
    payload.request_rate_limit_window_seconds = optionalNumber(
      requestRateLimitWindowSeconds
    )
  }

  return payload
}

function ApiKeyManagement() {
  const { t } = useTranslation()
  const [keys, setKeys] = useState([])
  const [createOpen, setCreateOpen] = useState(false)
  const [newKeyName, setNewKeyName] = useState('')
  const [newKeyDescription, setNewKeyDescription] = useState('')
  const [newKeyExpires, setNewKeyExpires] = useState(null)
  const [newTokenBudget, setNewTokenBudget] = useState('')
  const [newTokenRenewal, setNewTokenRenewal] = useState('none')
  const [newTokenRenewalIntervalDays, setNewTokenRenewalIntervalDays] =
    useState('')
  const [newRequestRateLimitEnabled, setNewRequestRateLimitEnabled] =
    useState(false)
  const [newRequestRateLimitRequests, setNewRequestRateLimitRequests] =
    useState('')
  const [
    newRequestRateLimitWindowSeconds,
    setNewRequestRateLimitWindowSeconds,
  ] = useState('')
  const [newKeyPermType, setNewKeyPermType] = useState('all')
  const [selectedModelTypes, setSelectedModelTypes] = useState([])
  const [selectedModelIds, setSelectedModelIds] = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [createdKey, setCreatedKey] = useState(null)
  const [snackError, setSnackError] = useState('')
  const [snackSuccess, setSnackSuccess] = useState('')
  const [users, setUsers] = useState([])
  const [selectedOwner, setSelectedOwner] = useState(null)
  const [isAdmin, setIsAdmin] = useState(false)
  const [editOpen, setEditOpen] = useState(false)
  const [editingKey, setEditingKey] = useState(null)
  const [editPermType, setEditPermType] = useState('all')
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editKeyExpires, setEditKeyExpires] = useState(null)
  const [editTokenBudget, setEditTokenBudget] = useState('')
  const [editTokenRenewal, setEditTokenRenewal] = useState('none')
  const [editTokenRenewalIntervalDays, setEditTokenRenewalIntervalDays] =
    useState('')
  const [editRequestRateLimitEnabled, setEditRequestRateLimitEnabled] =
    useState(false)
  const [editRequestRateLimitRequests, setEditRequestRateLimitRequests] =
    useState('')
  const [
    editRequestRateLimitWindowSeconds,
    setEditRequestRateLimitWindowSeconds,
  ] = useState('')
  const [editModelTypes, setEditModelTypes] = useState([])
  const [editModelIds, setEditModelIds] = useState([])
  const [bansDialogOpen, setBansDialogOpen] = useState(false)
  const [bansDialogKeyId, setBansDialogKeyId] = useState(null)
  const [bansData, setBansData] = useState([])
  const [rotateOpen, setRotateOpen] = useState(false)
  const [rotatingKey, setRotatingKey] = useState(null)
  const [rotatedKey, setRotatedKey] = useState('')

  const loadKeys = async () => {
    try {
      const data = await fetchWrapper.get('/v1/admin/keys')
      setKeys(data)
    } catch (e) {
      console.error(e)
    }
  }

  const handleShowBans = async (keyId) => {
    try {
      const data = await fetchWrapper.get(`/v1/admin/keys/${keyId}/banned`)
      setBansData(Array.isArray(data) ? data : [])
      setBansDialogKeyId(keyId)
      setBansDialogOpen(true)
    } catch (e) {
      setSnackError(t('apikeyManagement.failedLoadBans'))
    }
  }

  const handleUnbanFromDialog = async (ip) => {
    try {
      await fetchWrapper.post(`/v1/admin/keys/${bansDialogKeyId}/unban`, { ip })
      const data = await fetchWrapper.get(
        `/v1/admin/keys/${bansDialogKeyId}/banned`
      )
      setBansData(Array.isArray(data) ? data : [])
    } catch (e) {
      setSnackError(t('apikeyManagement.failedUnban'))
    }
  }

  useEffect(() => {
    loadKeys()
    fetchWrapper
      .get('/v1/models')
      .then((response) => {
        const models = (response.data || []).map((m) => ({
          id: m.id,
          label: `${m.model_name || m.id} (${m.model_type || 'unknown'}) — ${
            m.id
          }`,
        }))
        setAvailableModels(models)
      })
      .catch(() => {})

    const token = sessionStorage.getItem('token')
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]))
        const scopes = payload.scopes || []
        if (scopes.includes('admin')) {
          setIsAdmin(true)
          fetchWrapper
            .get('/v1/admin/users')
            .then((data) => setUsers(data))
            .catch(() => {})
        }
      } catch (_) {
        /* token parse failed, not admin */
      }
    }
  }, [])

  const resetCreateUsageControls = () => {
    setNewTokenBudget('')
    setNewTokenRenewal('none')
    setNewTokenRenewalIntervalDays('')
    setNewRequestRateLimitEnabled(false)
    setNewRequestRateLimitRequests('')
    setNewRequestRateLimitWindowSeconds('')
  }

  const renderUsageStateText = (state) => (
    <Box sx={{ py: 0.75, minWidth: 0 }}>
      <Typography variant="body2" sx={{ lineHeight: 1.35 }}>
        {state.primary}
      </Typography>
      {state.secondary && (
        <Typography
          variant="caption"
          color={
            state.color === 'error' || state.color === 'warning'
              ? `${state.color}.main`
              : 'text.secondary'
          }
          sx={{ display: 'block', lineHeight: 1.3 }}
        >
          {state.secondary}
        </Typography>
      )}
    </Box>
  )

  const renderUsageControlFields = ({
    tokenBudget,
    setTokenBudget,
    tokenRenewal,
    setTokenRenewal,
    tokenRenewalIntervalDays,
    setTokenRenewalIntervalDays,
    requestRateLimitEnabled,
    setRequestRateLimitEnabled,
    requestRateLimitRequests,
    setRequestRateLimitRequests,
    requestRateLimitWindowSeconds,
    setRequestRateLimitWindowSeconds,
  }) => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
        {t('apikeyManagement.advancedSettings')}
      </Typography>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
          gap: 1.5,
        }}
      >
        <TextField
          label={t('apikeyManagement.tokenBudgetLabel')}
          type="number"
          fullWidth
          value={tokenBudget}
          onChange={(e) => setTokenBudget(e.target.value)}
          inputProps={{ min: 1 }}
        />
        <TextField
          label={t('apikeyManagement.tokenRenewalLabel')}
          fullWidth
          select
          SelectProps={{ native: true }}
          value={tokenRenewal}
          onChange={(e) => setTokenRenewal(e.target.value)}
        >
          {TOKEN_RENEWAL_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {t(`apikeyManagement.${TOKEN_RENEWAL_LABEL_KEYS[option]}`)}
            </option>
          ))}
        </TextField>
        {tokenRenewal === 'custom' && (
          <TextField
            label={t('apikeyManagement.tokenRenewalDaysLabel')}
            type="number"
            fullWidth
            value={tokenRenewalIntervalDays}
            onChange={(e) => setTokenRenewalIntervalDays(e.target.value)}
            inputProps={{ min: 1 }}
          />
        )}
      </Box>
      <FormControlLabel
        sx={{ mt: 1 }}
        control={
          <Checkbox
            checked={requestRateLimitEnabled}
            onChange={(e) => setRequestRateLimitEnabled(e.target.checked)}
          />
        }
        label={t('apikeyManagement.requestRateLimitEnabled')}
      />
      {requestRateLimitEnabled && (
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
            gap: 1.5,
          }}
        >
          <TextField
            label={t('apikeyManagement.rateLimitRequestsLabel')}
            type="number"
            fullWidth
            value={requestRateLimitRequests}
            onChange={(e) => setRequestRateLimitRequests(e.target.value)}
            inputProps={{ min: 1 }}
          />
          <TextField
            label={t('apikeyManagement.rateLimitWindowLabel')}
            type="number"
            fullWidth
            value={requestRateLimitWindowSeconds}
            onChange={(e) => setRequestRateLimitWindowSeconds(e.target.value)}
            inputProps={{ min: 1 }}
          />
        </Box>
      )}
    </Box>
  )

  const handleCreate = async () => {
    try {
      const model_permissions = []
      if (newKeyPermType === 'all') {
        model_permissions.push({
          permission_type: 'all',
          permission_value: null,
        })
      } else if (newKeyPermType === 'model_type') {
        selectedModelTypes.forEach((v) => {
          model_permissions.push({
            permission_type: 'model_type',
            permission_value: v,
          })
        })
      } else if (newKeyPermType === 'model_id') {
        selectedModelIds.forEach((v) => {
          model_permissions.push({
            permission_type: 'model_id',
            permission_value: v.id || v,
          })
        })
      }
      const result = await fetchWrapper.post('/v1/admin/keys', {
        name: newKeyName || undefined,
        description: newKeyDescription || undefined,
        owner: selectedOwner ? selectedOwner.id : undefined,
        expires_at: newKeyExpires
          ? newKeyExpires.endOf('day').format('YYYY-MM-DDTHH:mm:ss')
          : undefined,
        model_permissions,
        ...buildUsageControlPayload({
          tokenBudget: newTokenBudget,
          tokenRenewal: newTokenRenewal,
          tokenRenewalIntervalDays: newTokenRenewalIntervalDays,
          requestRateLimitEnabled: newRequestRateLimitEnabled,
          requestRateLimitRequests: newRequestRateLimitRequests,
          requestRateLimitWindowSeconds: newRequestRateLimitWindowSeconds,
        }),
      })
      setCreatedKey(result.key)
      setNewKeyName('')
      setNewKeyDescription('')
      setNewKeyExpires(null)
      resetCreateUsageControls()
      setNewKeyPermType('all')
      setSelectedModelTypes([])
      setSelectedModelIds([])
      setSelectedOwner(null)
      loadKeys()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handleToggleEnabled = async (key) => {
    try {
      await fetchWrapper.put(`/v1/admin/keys/${key.id}`, {
        enabled: !key.enabled,
      })
      loadKeys()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handleDelete = async (key) => {
    if (
      !window.confirm(
        `${t('apikeyManagement.confirmDelete')} "${
          key.name || key.key_prefix
        }"?`
      )
    )
      return
    try {
      await fetchWrapper.delete(`/v1/admin/keys/${key.id}`)
      loadKeys()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handleOpenRotate = (key) => {
    setRotatingKey(key)
    setRotatedKey('')
    setRotateOpen(true)
  }

  const handleCloseRotate = () => {
    setRotateOpen(false)
    setRotatingKey(null)
    setRotatedKey('')
  }

  const handleRotate = async () => {
    if (!rotatingKey) return
    try {
      const data = await fetchWrapper.post(
        `/v1/admin/keys/${rotatingKey.id}/rotate`,
        {}
      )
      setRotatedKey(data.key)
      setRotatingKey({
        ...rotatingKey,
        key_prefix: data.key_prefix,
        rotated_at: data.rotated_at,
      })
      setSnackSuccess(t('apikeyManagement.keyRotated'))
      loadKeys()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handleEditPermissions = (key) => {
    const perms = key.model_permissions || []
    if (perms.some((p) => p.permission_type === 'all')) {
      setEditPermType('all')
      setEditModelTypes([])
      setEditModelIds([])
    } else {
      const types = perms
        .filter((p) => p.permission_type === 'model_type')
        .map((p) => p.permission_value)
      const ids = perms
        .filter((p) => p.permission_type === 'model_id')
        .map((p) => p.permission_value)
      if (types.length > 0 && ids.length > 0) {
        setEditPermType('mixed')
      } else if (ids.length > 0) {
        setEditPermType('model_id')
      } else {
        setEditPermType('model_type')
      }
      setEditModelTypes(types)
      setEditModelIds(ids)
    }
    setEditName(key.name || '')
    setEditDescription(key.description || '')
    setEditKeyExpires(key.expires_at ? dayjs(key.expires_at) : null)
    setEditTokenBudget(numberText(key.token_budget))
    setEditTokenRenewal(key.token_renewal || 'none')
    setEditTokenRenewalIntervalDays(numberText(key.token_renewal_interval_days))
    setEditRequestRateLimitEnabled(Boolean(key.request_rate_limit_enabled))
    setEditRequestRateLimitRequests(numberText(key.request_rate_limit_requests))
    setEditRequestRateLimitWindowSeconds(
      numberText(key.request_rate_limit_window_seconds)
    )
    setEditingKey(key)
    setEditOpen(true)
  }

  const handleSavePermissions = async () => {
    if (!editingKey) return
    try {
      const model_permissions = []
      if (editPermType === 'all') {
        model_permissions.push({
          permission_type: 'all',
          permission_value: null,
        })
      } else if (editPermType === 'model_type') {
        editModelTypes.forEach((v) => {
          model_permissions.push({
            permission_type: 'model_type',
            permission_value: v,
          })
        })
      } else if (editPermType === 'model_id') {
        editModelIds.forEach((v) => {
          model_permissions.push({
            permission_type: 'model_id',
            permission_value: v.id || v,
          })
        })
      } else {
        editModelTypes.forEach((v) => {
          model_permissions.push({
            permission_type: 'model_type',
            permission_value: v,
          })
        })
        editModelIds.forEach((v) => {
          model_permissions.push({
            permission_type: 'model_id',
            permission_value: v.id || v,
          })
        })
      }
      await fetchWrapper.put(`/v1/admin/keys/${editingKey.id}`, {
        name: editName === '' ? null : editName,
        description: editDescription === '' ? null : editDescription,
        expires_at: editKeyExpires
          ? editKeyExpires.endOf('day').format('YYYY-MM-DDTHH:mm:ss')
          : null,
        model_permissions,
        ...buildUsageControlPayload({
          tokenBudget: editTokenBudget,
          tokenRenewal: editTokenRenewal,
          tokenRenewalIntervalDays: editTokenRenewalIntervalDays,
          requestRateLimitEnabled: editRequestRateLimitEnabled,
          requestRateLimitRequests: editRequestRateLimitRequests,
          requestRateLimitWindowSeconds: editRequestRateLimitWindowSeconds,
        }),
      })
      setEditOpen(false)
      setEditingKey(null)
      setSnackSuccess(t('apikeyManagement.permissionsSaved'))
      loadKeys()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const columns = [
    { field: 'id', headerName: t('apikeyManagement.id'), width: 60 },
    {
      field: 'name',
      headerName: t('apikeyManagement.name'),
      flex: 1,
      minWidth: 120,
    },
    {
      field: 'description',
      headerName: t('apikeyManagement.description'),
      flex: 1,
      minWidth: 120,
    },
    {
      field: 'user_id',
      headerName: t('apikeyManagement.owner'),
      flex: 0.8,
      minWidth: 100,
      renderCell: (params) => {
        const user = users.find((u) => u.id === params.value)
        return user ? user.username : `#${params.value}`
      },
    },
    {
      field: 'key_prefix',
      headerName: t('apikeyManagement.prefix'),
      width: 100,
    },
    {
      field: 'enabled',
      headerName: t('apikeyManagement.status'),
      width: 80,
      renderCell: (params) =>
        (() => {
          const status = buildApiKeyListState(params.row, t).status
          return (
            <Chip
              label={status.label}
              color={status.color}
              size="small"
              variant={status.key === 'disabled' ? 'outlined' : 'filled'}
            />
          )
        })(),
    },
    {
      field: 'expires_at',
      headerName: t('apikeyManagement.expires'),
      flex: 1,
      minWidth: 140,
      renderCell: (params) =>
        renderUsageStateText(buildApiKeyListState(params.row, t).expiration),
    },
    {
      field: 'rotated_at',
      headerName: t('apikeyManagement.lastRotated'),
      flex: 1,
      minWidth: 140,
      renderCell: (params) =>
        renderUsageStateText(buildApiKeyListState(params.row, t).rotation),
    },
    {
      field: 'token_usage',
      headerName: t('apikeyManagement.tokenUsage'),
      flex: 1.2,
      minWidth: 160,
      renderCell: (params) =>
        renderUsageStateText(buildApiKeyListState(params.row, t).tokenUsage),
    },
    {
      field: 'token_renewal',
      headerName: t('apikeyManagement.tokenRenewal'),
      flex: 1,
      minWidth: 150,
      renderCell: (params) =>
        renderUsageStateText(buildApiKeyListState(params.row, t).renewal),
    },
    {
      field: 'request_rate_limit_enabled',
      headerName: t('apikeyManagement.requestRateLimit'),
      flex: 1.15,
      minWidth: 170,
      renderCell: (params) =>
        renderUsageStateText(buildApiKeyListState(params.row, t).rateLimit),
    },
    {
      field: 'model_permissions',
      headerName: t('apikeyManagement.modelPermissions'),
      flex: 1.5,
      minWidth: 180,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', py: 0.5 }}>
          {(params.value || []).map((mp, i) => (
            <Chip
              key={i}
              label={
                mp.permission_type === 'all'
                  ? 'ALL'
                  : `${mp.permission_type}:${mp.permission_value}`
              }
              size="small"
              variant="outlined"
              sx={{
                'maxWidth': '100%',
                'height': 'auto',
                '& .MuiChip-label': {
                  whiteSpace: 'normal',
                  wordBreak: 'break-all',
                },
              }}
            />
          ))}
        </Box>
      ),
    },
    ...(isAdmin
      ? [
          {
            field: 'bans',
            headerName: t('apikeyManagement.bans'),
            width: 70,
            renderCell: (params) => (
              <Chip
                label={t('apikeyManagement.viewBans')}
                size="small"
                variant="outlined"
                onClick={() => handleShowBans(params.row.id)}
                sx={{ cursor: 'pointer' }}
              />
            ),
          },
        ]
      : []),
    {
      field: 'actions',
      headerName: t('apikeyManagement.actions'),
      width: 190,
      renderCell: (params) => (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.25,
            flexWrap: 'nowrap',
          }}
        >
          <Tooltip title={t('apikeyManagement.editPermTooltip')}>
            <IconButton
              size="small"
              onClick={() => handleEditPermissions(params.row)}
            >
              <EditIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title={t('apikeyManagement.rotateTooltip')}>
            <IconButton
              size="small"
              onClick={() => handleOpenRotate(params.row)}
            >
              <AutorenewIcon />
            </IconButton>
          </Tooltip>
          <Tooltip
            title={
              params.row.enabled
                ? t('apikeyManagement.disableTooltip')
                : t('apikeyManagement.enableTooltip')
            }
          >
            <IconButton
              size="small"
              onClick={() => handleToggleEnabled(params.row)}
            >
              {params.row.enabled ? <BlockIcon /> : <CheckCircleIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title={t('apikeyManagement.deleteTooltip')}>
            <IconButton size="small" onClick={() => handleDelete(params.row)}>
              <DeleteIcon />
            </IconButton>
          </Tooltip>
        </Box>
      ),
    },
  ]

  return (
    <Box sx={{ p: 3 }}>
      <Title title={t('apikeyManagement.title')} />
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Button variant="contained" onClick={() => setCreateOpen(true)}>
          {t('apikeyManagement.createKey')}
        </Button>
      </Box>
      <DataGrid
        rows={keys}
        columns={columns}
        autoHeight
        pageSizeOptions={[10, 25, 50]}
        initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
        getRowHeight={() => 'auto'}
      />

      {/* Create Key Dialog */}
      <Dialog
        open={createOpen}
        onClose={() => {
          setCreateOpen(false)
          setCreatedKey(null)
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {createdKey
            ? t('apikeyManagement.keyCreatedTitle')
            : t('apikeyManagement.createKeyTitle')}
        </DialogTitle>
        <DialogContent>
          {createdKey ? (
            <Box>
              <Typography variant="body2" color="warning.main" sx={{ mb: 1 }}>
                {t('apikeyManagement.saveKeyWarning')}
              </Typography>
              <TextField
                fullWidth
                value={createdKey}
                InputProps={{
                  readOnly: true,
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title={t('apikeyManagement.copy')}>
                        <IconButton
                          size="small"
                          onClick={() => {
                            navigator.clipboard.writeText(createdKey)
                            setSnackSuccess(t('apikeyManagement.copied'))
                          }}
                        >
                          <ContentCopyIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </InputAdornment>
                  ),
                }}
                sx={{ fontFamily: 'monospace' }}
              />
            </Box>
          ) : (
            <Box>
              <TextField
                margin="dense"
                label={t('apikeyManagement.nameLabel')}
                fullWidth
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
              />
              <TextField
                margin="dense"
                label={t('apikeyManagement.descriptionLabel')}
                fullWidth
                multiline
                minRows={2}
                maxRows={4}
                value={newKeyDescription}
                onChange={(e) => setNewKeyDescription(e.target.value)}
              />
              {isAdmin && (
                <Autocomplete
                  options={users}
                  getOptionLabel={(option) => option.username}
                  value={selectedOwner}
                  onChange={(_, val) => setSelectedOwner(val)}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      margin="dense"
                      label={t('apikeyManagement.ownerLabel')}
                      helperText={t('apikeyManagement.ownerHelperText')}
                    />
                  )}
                />
              )}
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DatePicker
                  label={t('apikeyManagement.expiresLabel')}
                  value={newKeyExpires}
                  onChange={(val) => setNewKeyExpires(val)}
                  slotProps={{
                    textField: {
                      fullWidth: true,
                      margin: 'dense',
                      helperText: t('apikeyManagement.expiresHelperText'),
                    },
                    field: { clearable: true },
                  }}
                  minDate={dayjs().add(1, 'day')}
                />
              </LocalizationProvider>
              {renderUsageControlFields({
                tokenBudget: newTokenBudget,
                setTokenBudget: setNewTokenBudget,
                tokenRenewal: newTokenRenewal,
                setTokenRenewal: setNewTokenRenewal,
                tokenRenewalIntervalDays: newTokenRenewalIntervalDays,
                setTokenRenewalIntervalDays: setNewTokenRenewalIntervalDays,
                requestRateLimitEnabled: newRequestRateLimitEnabled,
                setRequestRateLimitEnabled: setNewRequestRateLimitEnabled,
                requestRateLimitRequests: newRequestRateLimitRequests,
                setRequestRateLimitRequests: setNewRequestRateLimitRequests,
                requestRateLimitWindowSeconds: newRequestRateLimitWindowSeconds,
                setRequestRateLimitWindowSeconds:
                  setNewRequestRateLimitWindowSeconds,
              })}
              <TextField
                margin="dense"
                label={t('apikeyManagement.permissionType')}
                fullWidth
                select
                SelectProps={{ native: true }}
                value={newKeyPermType}
                onChange={(e) => setNewKeyPermType(e.target.value)}
              >
                <option value="all">{t('apikeyManagement.allModels')}</option>
                <option value="model_type">
                  {t('apikeyManagement.byModelType')}
                </option>
                <option value="model_id">
                  {t('apikeyManagement.byModelId')}
                </option>
              </TextField>
              {newKeyPermType === 'model_type' && (
                <FormGroup row sx={{ mt: 1 }}>
                  {MODEL_TYPES.map((type) => (
                    <FormControlLabel
                      key={type}
                      control={
                        <Checkbox
                          size="small"
                          checked={selectedModelTypes.includes(type)}
                          onChange={() =>
                            setSelectedModelTypes((prev) =>
                              prev.includes(type)
                                ? prev.filter((t) => t !== type)
                                : [...prev, type]
                            )
                          }
                        />
                      }
                      label={type}
                    />
                  ))}
                </FormGroup>
              )}
              {newKeyPermType === 'model_id' && (
                <Autocomplete
                  multiple
                  options={availableModels}
                  getOptionLabel={(option) => option.label || option}
                  isOptionEqualToValue={(option, value) =>
                    option.id === (value.id || value)
                  }
                  value={selectedModelIds}
                  onChange={(_, val) => setSelectedModelIds(val)}
                  renderTags={(value, getTagProps) =>
                    value.map((option, index) => (
                      <Chip
                        size="small"
                        label={option.label || option}
                        {...getTagProps({ index })}
                        key={option.id || option}
                      />
                    ))
                  }
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      margin="dense"
                      label={t('apikeyManagement.permissionValues')}
                      helperText={t('apikeyManagement.modelIdHelper')}
                    />
                  )}
                  sx={{ mt: 1 }}
                />
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setCreateOpen(false)
              setCreatedKey(null)
            }}
          >
            {createdKey
              ? t('apikeyManagement.close')
              : t('apikeyManagement.cancel')}
          </Button>
          {!createdKey && (
            <Button variant="contained" onClick={handleCreate}>
              {t('apikeyManagement.create')}
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Rotate Key Dialog */}
      <Dialog
        open={rotateOpen}
        onClose={handleCloseRotate}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {rotatedKey
            ? t('apikeyManagement.keyRotatedTitle')
            : t('apikeyManagement.rotateTitle')}
        </DialogTitle>
        <DialogContent>
          {rotatedKey ? (
            <Box>
              <Typography variant="body2" color="warning.main" sx={{ mb: 1 }}>
                {t('apikeyManagement.rotateKeyWarning')}
              </Typography>
              <TextField
                fullWidth
                value={rotatedKey}
                InputProps={{
                  readOnly: true,
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title={t('apikeyManagement.copy')}>
                        <IconButton
                          size="small"
                          onClick={() => {
                            navigator.clipboard.writeText(rotatedKey)
                            setSnackSuccess(t('apikeyManagement.copied'))
                          }}
                        >
                          <ContentCopyIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </InputAdornment>
                  ),
                }}
                sx={{ fontFamily: 'monospace' }}
              />
            </Box>
          ) : (
            <Typography variant="body2">
              {t('apikeyManagement.rotateConfirm', {
                name: rotatingKey?.name || rotatingKey?.key_prefix || '',
              })}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          {rotatedKey ? (
            <Button onClick={handleCloseRotate}>
              {t('apikeyManagement.close')}
            </Button>
          ) : (
            <>
              <Button onClick={handleCloseRotate}>
                {t('apikeyManagement.cancel')}
              </Button>
              <Button
                color="warning"
                variant="contained"
                onClick={handleRotate}
              >
                {t('apikeyManagement.rotate')}
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>

      {/* Edit Permissions Dialog */}
      <Dialog
        open={editOpen}
        onClose={() => setEditOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {t('apikeyManagement.editKeyTitle')}{' '}
          {editingKey && (editingKey.name || editingKey.key_prefix)}
        </DialogTitle>
        <DialogContent>
          <TextField
            margin="dense"
            label={t('apikeyManagement.nameLabel')}
            fullWidth
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
          />
          <TextField
            margin="dense"
            label={t('apikeyManagement.descriptionLabel')}
            fullWidth
            multiline
            minRows={2}
            maxRows={4}
            value={editDescription}
            onChange={(e) => setEditDescription(e.target.value)}
          />
          <LocalizationProvider dateAdapter={AdapterDayjs}>
            <DatePicker
              label={t('apikeyManagement.expiresLabel')}
              value={editKeyExpires}
              onChange={(val) => setEditKeyExpires(val)}
              slotProps={{
                textField: {
                  fullWidth: true,
                  margin: 'dense',
                  helperText: t('apikeyManagement.expiresHelperText'),
                },
                field: { clearable: true },
              }}
              minDate={dayjs().add(1, 'day')}
            />
          </LocalizationProvider>
          {renderUsageControlFields({
            tokenBudget: editTokenBudget,
            setTokenBudget: setEditTokenBudget,
            tokenRenewal: editTokenRenewal,
            setTokenRenewal: setEditTokenRenewal,
            tokenRenewalIntervalDays: editTokenRenewalIntervalDays,
            setTokenRenewalIntervalDays: setEditTokenRenewalIntervalDays,
            requestRateLimitEnabled: editRequestRateLimitEnabled,
            setRequestRateLimitEnabled: setEditRequestRateLimitEnabled,
            requestRateLimitRequests: editRequestRateLimitRequests,
            setRequestRateLimitRequests: setEditRequestRateLimitRequests,
            requestRateLimitWindowSeconds: editRequestRateLimitWindowSeconds,
            setRequestRateLimitWindowSeconds:
              setEditRequestRateLimitWindowSeconds,
          })}
          <TextField
            margin="dense"
            label={t('apikeyManagement.permissionType')}
            fullWidth
            select
            SelectProps={{ native: true }}
            value={editPermType}
            onChange={(e) => {
              const newType = e.target.value
              if (newType === 'all') {
                setEditModelTypes([])
                setEditModelIds([])
              } else if (newType === 'model_type') {
                setEditModelIds([])
              } else if (newType === 'model_id') {
                setEditModelTypes([])
              }
              setEditPermType(newType)
            }}
          >
            <option value="all">{t('apikeyManagement.allModels')}</option>
            <option value="model_type">
              {t('apikeyManagement.byModelType')}
            </option>
            <option value="model_id">{t('apikeyManagement.byModelId')}</option>
            <option value="mixed">{t('apikeyManagement.mixedMode')}</option>
          </TextField>
          {(editPermType === 'model_type' || editPermType === 'mixed') && (
            <FormGroup row sx={{ mt: 1 }}>
              {MODEL_TYPES.map((type) => (
                <FormControlLabel
                  key={type}
                  control={
                    <Checkbox
                      size="small"
                      checked={editModelTypes.includes(type)}
                      onChange={() =>
                        setEditModelTypes((prev) =>
                          prev.includes(type)
                            ? prev.filter((t) => t !== type)
                            : [...prev, type]
                        )
                      }
                    />
                  }
                  label={type}
                />
              ))}
            </FormGroup>
          )}
          {(editPermType === 'model_id' || editPermType === 'mixed') && (
            <Autocomplete
              multiple
              options={availableModels}
              getOptionLabel={(option) => option.label || option}
              isOptionEqualToValue={(option, value) =>
                option.id === (value.id || value)
              }
              value={editModelIds.map(
                (id) =>
                  availableModels.find((m) => m.id === id) || { id, label: id }
              )}
              onChange={(_, val) => setEditModelIds(val.map((v) => v.id || v))}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    size="small"
                    label={option.label || option}
                    {...getTagProps({ index })}
                    key={option.id || option}
                  />
                ))
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  margin="dense"
                  label={t('apikeyManagement.modelInstances')}
                  helperText={t('apikeyManagement.modelIdHelper')}
                />
              )}
              sx={{ mt: 1 }}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditOpen(false)}>
            {t('apikeyManagement.cancel')}
          </Button>
          <Button variant="contained" onClick={handleSavePermissions}>
            {t('apikeyManagement.save')}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={!!snackError}
        autoHideDuration={5000}
        onClose={() => setSnackError('')}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          severity="error"
          onClose={() => setSnackError('')}
          variant="filled"
        >
          {snackError}
        </Alert>
      </Snackbar>
      <Snackbar
        open={!!snackSuccess}
        autoHideDuration={2000}
        onClose={() => setSnackSuccess('')}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          severity="success"
          onClose={() => setSnackSuccess('')}
          variant="filled"
        >
          {snackSuccess}
        </Alert>
      </Snackbar>

      {/* Bans Detail Dialog */}
      <Dialog
        open={bansDialogOpen}
        onClose={() => setBansDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {t('apikeyManagement.bannedIpsForKey', { keyId: bansDialogKeyId })}
        </DialogTitle>
        <DialogContent>
          {bansData.length === 0 ? (
            <Typography color="text.secondary">
              {t('apikeyManagement.noActiveBans')}
            </Typography>
          ) : (
            <Box>
              {bansData.map((ban, idx) => (
                <Box
                  key={idx}
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                  sx={{ py: 1, borderBottom: '1px solid #eee' }}
                >
                  <Box>
                    <Typography variant="body2">{ban.ip}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {t('apikeyManagement.remaining', {
                        seconds: ban.remaining_seconds,
                      })}
                    </Typography>
                  </Box>
                  <Button
                    size="small"
                    color="error"
                    onClick={() => handleUnbanFromDialog(ban.ip)}
                  >
                    {t('apikeyManagement.unban')}
                  </Button>
                </Box>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBansDialogOpen(false)}>
            {t('apikeyManagement.close')}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default ApiKeyManagement
