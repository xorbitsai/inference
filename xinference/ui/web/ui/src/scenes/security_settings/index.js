import DeleteIcon from '@mui/icons-material/Delete'
import {
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  Grid,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import Title from '../../components/Title'
import { getEndpoint } from '../../components/utils'

function SecuritySettings() {
  const { t } = useTranslation()
  const endpoint = getEndpoint()
  const { setErrorMsg } = useContext(ApiContext)
  const [config, setConfig] = useState({ ip: {}, key: {} })
  const [bannedIps, setBannedIps] = useState([])
  const [bannedKeys, setBannedKeys] = useState([])

  const token = sessionStorage.getItem('token')
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  }

  const fetchConfig = () => {
    fetch(endpoint + '/v1/admin/security/rate-limit', { headers }).then((res) => {
      if (res.ok) res.json().then(setConfig)
    })
  }

  const fetchBannedIps = () => {
    fetch(endpoint + '/v1/admin/security/banned-ips', { headers }).then((res) => {
      if (res.ok) res.json().then(setBannedIps)
    })
  }

  const fetchBannedKeys = () => {
    fetch(endpoint + '/v1/admin/security/banned-keys', { headers }).then((res) => {
      if (res.ok) res.json().then(setBannedKeys)
    })
  }

  useEffect(() => {
    fetchConfig()
    fetchBannedIps()
    fetchBannedKeys()
  }, [])

  const handleSaveConfig = () => {
    fetch(endpoint + '/v1/admin/security/rate-limit', {
      method: 'PUT',
      headers,
      body: JSON.stringify(config),
    }).then((res) => {
      if (res.ok) setErrorMsg('')
      else setErrorMsg('Failed to update config')
    })
  }

  const handleUnbanIp = (ip) => {
    fetch(endpoint + '/v1/admin/security/unban-ip', {
      method: 'POST',
      headers,
      body: JSON.stringify({ ip }),
    }).then(() => fetchBannedIps())
  }

  const handleUnbanKey = (ip, key_id) => {
    fetch(endpoint + '/v1/admin/security/unban-key', {
      method: 'POST',
      headers,
      body: JSON.stringify({ ip, key_id }),
    }).then(() => fetchBannedKeys())
  }

  const handleUnbanAllIps = () => {
    fetch(endpoint + '/v1/admin/security/unban-all-ips', {
      method: 'POST',
      headers,
    }).then(() => fetchBannedIps())
  }

  const handleUnbanAllKeys = () => {
    fetch(endpoint + '/v1/admin/security/unban-all-keys', {
      method: 'POST',
      headers,
    }).then(() => fetchBannedKeys())
  }

  return (
    <Box m="20px">
      <ErrorMessageSnackBar />
      <Title title={t('securitySettings.title')} />

      {/* Rate Limit Config */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" mb={2}>
            {t('securitySettings.rateLimitConfig')}
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="subtitle1">{t('securitySettings.ipLayer')}</Typography>
            </Grid>
            <Grid item xs={4}>
              <TextField
                label={t('securitySettings.maxFailures')}
                type="number"
                size="small"
                fullWidth
                value={config.ip.max_failures || ''}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    ip: { ...config.ip, max_failures: parseInt(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label={t('securitySettings.windowSeconds')}
                type="number"
                size="small"
                fullWidth
                value={config.ip.window_seconds || ''}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    ip: { ...config.ip, window_seconds: parseInt(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label={t('securitySettings.banSeconds')}
                type="number"
                size="small"
                fullWidth
                value={config.ip.ban_seconds || ''}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    ip: { ...config.ip, ban_seconds: parseInt(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={12}>
              <Divider sx={{ my: 1 }} />
              <Typography variant="subtitle1">{t('securitySettings.keyLayer')}</Typography>
            </Grid>
            <Grid item xs={4}>
              <TextField
                label={t('securitySettings.maxFailures')}
                type="number"
                size="small"
                fullWidth
                value={config.key.max_failures || ''}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    key: { ...config.key, max_failures: parseInt(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label={t('securitySettings.windowSeconds')}
                type="number"
                size="small"
                fullWidth
                value={config.key.window_seconds || ''}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    key: { ...config.key, window_seconds: parseInt(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                label={t('securitySettings.banSeconds')}
                type="number"
                size="small"
                fullWidth
                value={config.key.ban_seconds || ''}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    key: { ...config.key, ban_seconds: parseInt(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={12}>
              <Button variant="contained" onClick={handleSaveConfig}>
                {t('securitySettings.saveConfig')}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Banned IPs */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="h6">{t('securitySettings.bannedIps')}</Typography>
            {bannedIps.length > 0 && (
              <Button size="small" color="error" onClick={handleUnbanAllIps}>
                {t('securitySettings.unbanAll')}
              </Button>
            )}
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {t('securitySettings.bannedIpsDesc')}
          </Typography>
          {bannedIps.length === 0 ? (
            <Typography color="text.secondary">{t('securitySettings.noBannedIps')}</Typography>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>{t('securitySettings.ip')}</TableCell>
                    <TableCell>{t('securitySettings.remaining')}</TableCell>
                    <TableCell>{t('securitySettings.action')}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {bannedIps.map((row) => (
                    <TableRow key={row.ip}>
                      <TableCell>{row.ip}</TableCell>
                      <TableCell>{row.remaining_seconds}</TableCell>
                      <TableCell>
                        <IconButton size="small" onClick={() => handleUnbanIp(row.ip)}>
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Banned (IP, Key) Pairs */}
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="h6">{t('securitySettings.bannedKeys')}</Typography>
            {bannedKeys.length > 0 && (
              <Button size="small" color="error" onClick={handleUnbanAllKeys}>
                {t('securitySettings.unbanAll')}
              </Button>
            )}
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {t('securitySettings.bannedKeysDesc')}
          </Typography>
          {bannedKeys.length === 0 ? (
            <Typography color="text.secondary">{t('securitySettings.noBannedKeys')}</Typography>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>{t('securitySettings.ip')}</TableCell>
                    <TableCell>{t('securitySettings.keyId')}</TableCell>
                    <TableCell>{t('securitySettings.remaining')}</TableCell>
                    <TableCell>{t('securitySettings.action')}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {bannedKeys.map((row, idx) => (
                    <TableRow key={idx}>
                      <TableCell>{row.ip}</TableCell>
                      <TableCell>{row.key_id}</TableCell>
                      <TableCell>{row.remaining_seconds}</TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => handleUnbanKey(row.ip, row.key_id)}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    </Box>
  )
}

export default SecuritySettings
