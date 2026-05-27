import BlockIcon from '@mui/icons-material/Block'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import DeleteIcon from '@mui/icons-material/Delete'
import EditIcon from '@mui/icons-material/Edit'
import {
  Alert,
  Box,
  Checkbox,
  Chip,
  Divider,
  FormControlLabel,
  FormGroup,
  IconButton,
  Snackbar,
  Tooltip,
  Typography,
} from '@mui/material'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import TextField from '@mui/material/TextField'
import { DataGrid } from '@mui/x-data-grid'
import * as React from 'react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import fetchWrapper from '../../components/fetchWrapper'
import Title from '../../components/Title'

const ALL_PERMISSIONS = [
  { group: 'admin', items: ['admin'] },
  { group: 'models', items: ['models:list', 'models:read', 'models:write'] },
  { group: 'keys', items: ['keys:create', 'keys:manage'] },
  { group: 'users', items: ['users:manage'] },
  { group: 'cache', items: ['cache:list', 'cache:delete'] },
  { group: 'virtualenv', items: ['virtualenv:list', 'virtualenv:delete'] },
]

const ALL_PERMISSION_VALUES = ALL_PERMISSIONS.flatMap((g) => g.items)

function UserManagement() {
  const { t } = useTranslation()
  const [users, setUsers] = useState([])
  const [createOpen, setCreateOpen] = useState(false)
  const [newUsername, setNewUsername] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [selectedPerms, setSelectedPerms] = useState([])
  const [snackError, setSnackError] = useState('')
  const [editPermOpen, setEditPermOpen] = useState(false)
  const [editingUser, setEditingUser] = useState(null)
  const [editPerms, setEditPerms] = useState([])

  const loadUsers = async () => {
    try {
      const data = await fetchWrapper.get('/v1/admin/users')
      setUsers(data)
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    loadUsers()
  }, [])

  const handleCreate = async () => {
    try {
      await fetchWrapper.post('/v1/admin/users', {
        username: newUsername,
        password: newPassword,
        permissions: selectedPerms,
      })
      setCreateOpen(false)
      setNewUsername('')
      setNewPassword('')
      setSelectedPerms([])
      loadUsers()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handleToggleEnabled = async (user) => {
    try {
      await fetchWrapper.put(`/v1/admin/users/${user.id}`, {
        enabled: !user.enabled,
      })
      loadUsers()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handleDelete = async (user) => {
    if (
      !window.confirm(
        `${t('userManagement.confirmDelete')} "${user.username}"?`
      )
    )
      return
    try {
      await fetchWrapper.delete(`/v1/admin/users/${user.id}`)
      loadUsers()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const handlePermToggle = (perm) => {
    setSelectedPerms((prev) =>
      prev.includes(perm) ? prev.filter((p) => p !== perm) : [...prev, perm]
    )
  }

  const handleSelectAll = () => {
    if (selectedPerms.length === ALL_PERMISSION_VALUES.length) {
      setSelectedPerms([])
    } else {
      setSelectedPerms([...ALL_PERMISSION_VALUES])
    }
  }

  const handleEditPermissions = (user) => {
    setEditingUser(user)
    setEditPerms(user.permissions || [])
    setEditPermOpen(true)
  }

  const handleEditPermToggle = (perm) => {
    setEditPerms((prev) =>
      prev.includes(perm) ? prev.filter((p) => p !== perm) : [...prev, perm]
    )
  }

  const handleEditSelectAll = () => {
    if (editPerms.length === ALL_PERMISSION_VALUES.length) {
      setEditPerms([])
    } else {
      setEditPerms([...ALL_PERMISSION_VALUES])
    }
  }

  const handleSavePermissions = async () => {
    try {
      await fetchWrapper.put(
        `/v1/admin/users/${editingUser.id}/permissions`,
        { permissions: editPerms }
      )
      setEditPermOpen(false)
      setEditingUser(null)
      loadUsers()
    } catch (e) {
      setSnackError(e.message || String(e))
    }
  }

  const columns = [
    { field: 'id', headerName: t('userManagement.id'), width: 60 },
    { field: 'username', headerName: t('userManagement.username'), width: 150 },
    { field: 'source', headerName: t('userManagement.source'), width: 100 },
    {
      field: 'enabled',
      headerName: t('userManagement.status'),
      width: 100,
      renderCell: (params) =>
        params.value ? (
          <Chip
            label={t('userManagement.active')}
            color="success"
            size="small"
          />
        ) : (
          <Chip
            label={t('userManagement.disabled')}
            color="error"
            size="small"
          />
        ),
    },
    {
      field: 'permissions',
      headerName: t('userManagement.permissions'),
      width: 300,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
          {(params.value || []).map((p) => (
            <Chip
              key={p}
              label={t(`permissions.${p.replace(/:/g, '_')}`, p)}
              size="small"
              variant="outlined"
            />
          ))}
        </Box>
      ),
    },
    {
      field: 'actions',
      headerName: t('userManagement.actions'),
      width: 150,
      renderCell: (params) => (
        <Box>
          <Tooltip title={t('userManagement.editPermissions') || 'Edit Permissions'}>
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
      <Title title={t('userManagement.title')} />
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Button variant="contained" onClick={() => setCreateOpen(true)}>
          {t('userManagement.createUser')}
        </Button>
      </Box>
      <DataGrid
        rows={users}
        columns={columns}
        autoHeight
        pageSizeOptions={[10, 25, 50]}
        initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
        getRowHeight={() => 'auto'}
      />

      <Dialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>{t('userManagement.dialogTitle')}</DialogTitle>
        <DialogContent>
          <TextField
            margin="dense"
            label={t('userManagement.usernameLabel')}
            fullWidth
            value={newUsername}
            onChange={(e) => setNewUsername(e.target.value)}
          />
          <TextField
            margin="dense"
            label={t('userManagement.passwordLabel')}
            type="password"
            fullWidth
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
          />
          <Box sx={{ mt: 2 }}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <Typography variant="subtitle2">
                {t('userManagement.permissions')}
              </Typography>
              <FormControlLabel
                control={
                  <Checkbox
                    size="small"
                    checked={
                      selectedPerms.length === ALL_PERMISSION_VALUES.length
                    }
                    indeterminate={
                      selectedPerms.length > 0 &&
                      selectedPerms.length < ALL_PERMISSION_VALUES.length
                    }
                    onChange={handleSelectAll}
                  />
                }
                label={t('userManagement.selectAll')}
              />
            </Box>
            <Divider sx={{ mb: 1 }} />
            {ALL_PERMISSIONS.map((group) => (
              <Box key={group.group} sx={{ mb: 1 }}>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontWeight: 'bold' }}
                >
                  {group.group}
                </Typography>
                <FormGroup row>
                  {group.items.map((perm) => (
                    <FormControlLabel
                      key={perm}
                      control={
                        <Checkbox
                          size="small"
                          checked={selectedPerms.includes(perm)}
                          onChange={() => handlePermToggle(perm)}
                        />
                      }
                      label={t(`permissions.${perm.replace(/:/g, '_')}`, perm)}
                      sx={{ mr: 2 }}
                    />
                  ))}
                </FormGroup>
              </Box>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>
            {t('userManagement.cancel')}
          </Button>
          <Button variant="contained" onClick={handleCreate}>
            {t('userManagement.create')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Permissions Dialog */}
      <Dialog
        open={editPermOpen}
        onClose={() => setEditPermOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {t('userManagement.editPermissionsTitle') || 'Edit Permissions'}{' '}
          {editingUser && `- ${editingUser.username}`}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 1 }}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <Typography variant="subtitle2">
                {t('userManagement.permissions')}
              </Typography>
              <FormControlLabel
                control={
                  <Checkbox
                    size="small"
                    checked={
                      editPerms.length === ALL_PERMISSION_VALUES.length
                    }
                    indeterminate={
                      editPerms.length > 0 &&
                      editPerms.length < ALL_PERMISSION_VALUES.length
                    }
                    onChange={handleEditSelectAll}
                  />
                }
                label={t('userManagement.selectAll')}
              />
            </Box>
            <Divider sx={{ mb: 1 }} />
            {ALL_PERMISSIONS.map((group) => (
              <Box key={group.group} sx={{ mb: 1 }}>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontWeight: 'bold' }}
                >
                  {group.group}
                </Typography>
                <FormGroup row>
                  {group.items.map((perm) => (
                    <FormControlLabel
                      key={perm}
                      control={
                        <Checkbox
                          size="small"
                          checked={editPerms.includes(perm)}
                          onChange={() => handleEditPermToggle(perm)}
                        />
                      }
                      label={t(`permissions.${perm.replace(/:/g, '_')}`, perm)}
                      sx={{ mr: 2 }}
                    />
                  ))}
                </FormGroup>
              </Box>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditPermOpen(false)}>
            {t('userManagement.cancel')}
          </Button>
          <Button variant="contained" onClick={handleSavePermissions}>
            {t('userManagement.save') || 'Save'}
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
    </Box>
  )
}

export default UserManagement
