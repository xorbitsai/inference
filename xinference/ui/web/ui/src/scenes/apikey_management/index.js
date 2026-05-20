import BlockIcon from '@mui/icons-material/Block'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import DeleteIcon from '@mui/icons-material/Delete'
import EditIcon from '@mui/icons-material/Edit'
import VisibilityIcon from '@mui/icons-material/Visibility'
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

const MODEL_TYPES = ['LLM', 'embedding', 'rerank', 'image', 'audio', 'video']

function ApiKeyManagement() {
  const { t } = useTranslation()
  const [keys, setKeys] = useState([])
  const [createOpen, setCreateOpen] = useState(false)
  const [revealOpen, setRevealOpen] = useState(false)
  const [revealedKey, setRevealedKey] = useState('')
  const [newKeyName, setNewKeyName] = useState('')
  const [newKeyExpires, setNewKeyExpires] = useState(null)
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
  const [editModelTypes, setEditModelTypes] = useState([])
  const [editModelIds, setEditModelIds] = useState([])

  const loadKeys = async () => {
    try {
      const data = await fetchWrapper.get('/v1/admin/keys')
      setKeys(data)
    } catch (e) {
      console.error(e)
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
        owner: selectedOwner ? selectedOwner.id : undefined,
        expires_at: newKeyExpires
          ? newKeyExpires.endOf('day').format('YYYY-MM-DDTHH:mm:ss')
          : undefined,
        model_permissions,
      })
      setCreatedKey(result.key)
      setNewKeyName('')
      setNewKeyExpires(null)
      setNewKeyPermType('all')
      setSelectedModelTypes([])
      setSelectedModelIds([])
      setSelectedOwner(null)
      loadKeys()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handleReveal = async (keyId) => {
    try {
      const data = await fetchWrapper.get(`/v1/admin/keys/${keyId}/reveal`)
      setRevealedKey(data.key)
      setRevealOpen(true)
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
      await fetchWrapper.put(`/v1/admin/keys/${editingKey.id}/permissions`, {
        model_permissions,
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
    { field: 'name', headerName: t('apikeyManagement.name'), width: 150 },
    {
      field: 'user_id',
      headerName: t('apikeyManagement.owner'),
      width: 120,
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
      width: 100,
      renderCell: (params) =>
        params.value ? (
          <Chip
            label={t('apikeyManagement.active')}
            color="success"
            size="small"
          />
        ) : (
          <Chip
            label={t('apikeyManagement.disabled')}
            color="error"
            size="small"
          />
        ),
    },
    {
      field: 'expires_at',
      headerName: t('apikeyManagement.expires'),
      width: 180,
    },
    {
      field: 'model_permissions',
      headerName: t('apikeyManagement.modelPermissions'),
      width: 250,
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
    {
      field: 'actions',
      headerName: t('apikeyManagement.actions'),
      width: 190,
      renderCell: (params) => (
        <Box>
          <Tooltip title={t('apikeyManagement.revealTooltip')}>
            <IconButton
              size="small"
              onClick={() => handleReveal(params.row.id)}
            >
              <VisibilityIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title={t('apikeyManagement.editPermTooltip')}>
            <IconButton
              size="small"
              onClick={() => handleEditPermissions(params.row)}
            >
              <EditIcon />
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

      {/* Reveal Key Dialog */}
      <Dialog
        open={revealOpen}
        onClose={() => setRevealOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>{t('apikeyManagement.revealTitle')}</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            value={revealedKey}
            InputProps={{
              readOnly: true,
              endAdornment: (
                <InputAdornment position="end">
                  <Tooltip title={t('apikeyManagement.copy')}>
                    <IconButton
                      size="small"
                      onClick={() => {
                        navigator.clipboard.writeText(revealedKey)
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
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRevealOpen(false)}>
            {t('apikeyManagement.close')}
          </Button>
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
          {t('apikeyManagement.editPermTitle')}{' '}
          {editingKey && (editingKey.name || editingKey.key_prefix)}
        </DialogTitle>
        <DialogContent>
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
    </Box>
  )
}

export default ApiKeyManagement
