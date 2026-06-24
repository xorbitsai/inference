import {
  CloudCircle as CloudCircleIcon,
  ErrorOutline as AlertCircleIcon,
  Storage as StorageIcon,
} from '@mui/icons-material'
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Snackbar,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../components/apiContext'

const DASHBOARD_FIELDS = [
  { key: 'overview', labelKey: 'monitoring.config.dashboardOverview' },
  { key: 'model_load', labelKey: 'monitoring.config.dashboardModelLoad' },
  { key: 'llm_slo', labelKey: 'monitoring.config.dashboardLlmSlo' },
  { key: 'gpu', labelKey: 'monitoring.config.dashboardGpu' },
  { key: 'host', labelKey: 'monitoring.config.dashboardHost' },
  { key: 'security', labelKey: 'monitoring.config.dashboardSecurity' },
]

const SOURCE_ICONS = {
  db: {
    icon: StorageIcon,
    color: 'success',
    labelKey: 'monitoring.config.sourceDb',
  },
  env: {
    icon: CloudCircleIcon,
    color: 'info',
    labelKey: 'monitoring.config.sourceEnv',
  },
  default: {
    icon: AlertCircleIcon,
    color: 'default',
    labelKey: 'monitoring.config.sourceDefault',
  },
}

const SourceChip = ({ source, t }) => {
  const cfg = SOURCE_ICONS[source] || SOURCE_ICONS.default
  const Icon = cfg.icon
  return (
    <Chip
      icon={<Icon fontSize="small" />}
      label={t(cfg.labelKey)}
      size="small"
      color={cfg.color}
      variant="outlined"
      sx={{ ml: 1, height: 24, fontSize: '0.75rem' }}
    />
  )
}

const MonitorConfigDialog = ({ open, onClose, onSaved }) => {
  const { endPoint } = useContext(ApiContext)
  const { t } = useTranslation()

  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [checking, setChecking] = useState(false)

  const [grafanaUrl, setGrafanaUrl] = useState('')
  const [grafanaDatasource, setGrafanaDatasource] = useState('')
  const [grafanaAlertDatasource, setGrafanaAlertDatasource] = useState('')
  const [clusterName, setClusterName] = useState('')
  const [dashboards, setDashboards] = useState({})
  const [sources, setSources] = useState({})

  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info',
  })

  useEffect(() => {
    if (!open) return
    setLoading(true)
    fetch(endPoint + '/v1/cluster/monitor_config', {
      headers: {
        Authorization: `Bearer ${sessionStorage.getItem('token') || ''}`,
      },
    })
      .then((res) => res.json())
      .then((data) => {
        setGrafanaUrl(data.grafana_url || '')
        setGrafanaDatasource(data.grafana_datasource || '')
        setGrafanaAlertDatasource(data.grafana_alert_datasource || '')
        setClusterName(data.cluster_name || '')
        setDashboards(data.grafana_dashboards || {})
        setSources(data.sources || {})
      })
      .catch(() => {
        setSnackbar({
          open: true,
          message: t('monitoring.config.loadFailed'),
          severity: 'error',
        })
      })
      .finally(() => setLoading(false))
  }, [open, endPoint, t])

  const handleSave = async () => {
    setSaving(true)

    // Check Grafana connectivity first
    if (grafanaUrl) {
      setChecking(true)
      try {
        const checkRes = await fetch(
          endPoint + '/v1/cluster/monitor_config/check-grafana',
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${
                sessionStorage.getItem('token') || ''
              }`,
            },
            body: JSON.stringify({ grafana_url: grafanaUrl }),
          }
        )
        const checkData = await checkRes.json()
        if (!checkData.ok) {
          setSnackbar({
            open: true,
            message: `${t('monitoring.config.healthCheckFailed')}: ${
              checkData.error
            }`,
            severity: 'warning',
          })
        }
      } catch (e) {
        // Ignore check errors, proceed with save
      }
      setChecking(false)
    }

    // Save config
    try {
      const body = {
        grafana_url: grafanaUrl,
        grafana_datasource: grafanaDatasource,
        grafana_alert_datasource: grafanaAlertDatasource,
        cluster_name: clusterName,
        grafana_dashboards: dashboards,
      }
      const res = await fetch(endPoint + '/v1/cluster/monitor_config', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify(body),
      })
      if (res.ok) {
        setSnackbar({
          open: true,
          message: t('monitoring.config.saved'),
          severity: 'success',
        })
        onSaved()
      } else {
        setSnackbar({
          open: true,
          message: t('monitoring.config.saveFailed'),
          severity: 'error',
        })
      }
    } catch (e) {
      setSnackbar({
        open: true,
        message: t('monitoring.config.saveFailed'),
        severity: 'error',
      })
    }
    setSaving(false)
  }

  const handleReset = async () => {
    try {
      const res = await fetch(endPoint + '/v1/cluster/monitor_config/reset', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${sessionStorage.getItem('token') || ''}`,
        },
      })
      if (res.ok) {
        setSnackbar({
          open: true,
          message: t('monitoring.config.resetDone'),
          severity: 'success',
        })
        onSaved()
      }
    } catch (e) {
      setSnackbar({
        open: true,
        message: t('monitoring.config.saveFailed'),
        severity: 'error',
      })
    }
  }

  const handleDashboardChange = (key, value) => {
    setDashboards((prev) => ({ ...prev, [key]: value }))
  }

  if (loading) {
    return (
      <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
        <DialogContent
          sx={{ display: 'flex', justifyContent: 'center', py: 4 }}
        >
          <CircularProgress />
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <>
      <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
        <DialogTitle>{t('monitoring.config.title')}</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2.5}>
            {/* Grafana URL */}
            <Box>
              <Box display="flex" alignItems="center" mb={0.5}>
                <Typography variant="subtitle2">
                  {t('monitoring.config.grafanaUrl')} *
                </Typography>
                <SourceChip source={sources.grafana_url} t={t} />
              </Box>
              <TextField
                fullWidth
                size="small"
                value={grafanaUrl}
                onChange={(e) => setGrafanaUrl(e.target.value)}
                placeholder="https://grafana.example.com"
              />
            </Box>

            {/* Datasource */}
            <Box>
              <Box display="flex" alignItems="center" mb={0.5}>
                <Typography variant="subtitle2">
                  {t('monitoring.config.datasource')}
                </Typography>
                <SourceChip source={sources.grafana_datasource} t={t} />
              </Box>
              <TextField
                fullWidth
                size="small"
                value={grafanaDatasource}
                onChange={(e) => setGrafanaDatasource(e.target.value)}
                placeholder="Prometheus"
              />
            </Box>

            {/* Alert Datasource */}
            <Box>
              <Box display="flex" alignItems="center" mb={0.5}>
                <Typography variant="subtitle2">
                  {t('monitoring.config.alertDatasource')}
                </Typography>
                <SourceChip source={sources.grafana_alert_datasource} t={t} />
              </Box>
              <TextField
                fullWidth
                size="small"
                value={grafanaAlertDatasource}
                onChange={(e) => setGrafanaAlertDatasource(e.target.value)}
                placeholder={t('monitoring.config.alertDatasourceHint')}
              />
            </Box>

            {/* Cluster Name */}
            <Box>
              <Box display="flex" alignItems="center" mb={0.5}>
                <Typography variant="subtitle2">
                  {t('monitoring.config.clusterName')}
                </Typography>
                <SourceChip source={sources.cluster_name} t={t} />
              </Box>
              <TextField
                fullWidth
                size="small"
                value={clusterName}
                onChange={(e) => setClusterName(e.target.value)}
              />
            </Box>

            <Divider />

            {/* Dashboard UIDs */}
            <Typography variant="subtitle2" color="text.secondary">
              {t('monitoring.config.dashboardUid')}
            </Typography>
            {DASHBOARD_FIELDS.map((field) => (
              <Box key={field.key}>
                <Box display="flex" alignItems="center" mb={0.5}>
                  <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
                    {t(field.labelKey)}
                  </Typography>
                  <SourceChip
                    source={sources[`dashboard_${field.key}`]}
                    t={t}
                  />
                </Box>
                <TextField
                  fullWidth
                  size="small"
                  value={dashboards[field.key] || ''}
                  onChange={(e) =>
                    handleDashboardChange(field.key, e.target.value)
                  }
                />
              </Box>
            ))}
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, py: 2 }}>
          <Button color="warning" onClick={handleReset}>
            {t('monitoring.config.reset')}
          </Button>
          <Box sx={{ flex: 1 }} />
          <Button onClick={onClose}>{t('monitoring.config.cancel')}</Button>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={saving || checking}
          >
            {checking
              ? t('monitoring.config.checking')
              : saving
              ? t('monitoring.config.saving')
              : t('monitoring.config.save')}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
        message={snackbar.message}
        severity={snackbar.severity}
      />
    </>
  )
}

export default MonitorConfigDialog
