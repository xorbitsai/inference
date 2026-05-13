import 'dayjs/locale/ja'
import 'dayjs/locale/ko'
import 'dayjs/locale/zh-cn'

import { AccessTime, Refresh as RefreshIcon } from '@mui/icons-material'
import {
  Box,
  Button,
  Divider,
  IconButton,
  Menu,
  MenuItem,
  MenuList,
  Popover,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material'
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker'
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider'
import React, { useContext, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../components/apiContext'
import { buildGrafanaUrl } from '../../components/grafanaUtils'
import { useThemeContext } from '../../components/themeContext'
import Title from '../../components/Title'

const FONT_SIZE = '0.813rem'

const DAYJS_LOCALE_MAP = { en: 'en', zh: 'zh-cn', ja: 'ja', ko: 'ko' }

const TIME_RANGES = [
  { labelKey: 'monitoring.time.5m', from: 'now-5m', to: 'now' },
  { labelKey: 'monitoring.time.15m', from: 'now-15m', to: 'now' },
  { labelKey: 'monitoring.time.30m', from: 'now-30m', to: 'now' },
  { labelKey: 'monitoring.time.1h', from: 'now-1h', to: 'now' },
  { labelKey: 'monitoring.time.3h', from: 'now-3h', to: 'now' },
  { labelKey: 'monitoring.time.6h', from: 'now-6h', to: 'now' },
  { labelKey: 'monitoring.time.12h', from: 'now-12h', to: 'now' },
  { labelKey: 'monitoring.time.24h', from: 'now-24h', to: 'now' },
  { labelKey: 'monitoring.time.2d', from: 'now-2d', to: 'now' },
  { labelKey: 'monitoring.time.7d', from: 'now-7d', to: 'now' },
]

const REFRESH_OPTIONS = [
  { labelKey: 'monitoring.refresh.off', value: 0, grafana: '' },
  { labelKey: 'monitoring.refresh.10s', value: 10000, grafana: '10s' },
  { labelKey: 'monitoring.refresh.30s', value: 30000, grafana: '30s' },
  { labelKey: 'monitoring.refresh.1m', value: 60000, grafana: '1m' },
  { labelKey: 'monitoring.refresh.5m', value: 300000, grafana: '5m' },
]

const DATE_TIME_SLOT_PROPS = {
  textField: {
    size: 'small',
    fullWidth: true,
    sx: {
      '& .MuiInputBase-input': { fontSize: FONT_SIZE },
      '& .MuiInputLabel-root': { fontSize: FONT_SIZE },
    },
  },
  popper: {
    sx: {
      '& .MuiDayCalendar-weekDayLabel': { fontSize: FONT_SIZE },
      '& .MuiPickersDay-root': { fontSize: FONT_SIZE },
    },
  },
}

const PICKER_LOCALE_TEXT = {
  en: { okButtonLabel: 'OK', cancelButtonLabel: 'Cancel' },
  zh: { okButtonLabel: '确定', cancelButtonLabel: '取消' },
  ja: { okButtonLabel: '確定', cancelButtonLabel: 'キャンセル' },
  ko: { okButtonLabel: '확인', cancelButtonLabel: '취소' },
}

const Monitoring = () => {
  const { endPoint } = useContext(ApiContext)
  const { themeMode } = useThemeContext()
  const [config, setConfig] = useState(null)
  const { t, i18n } = useTranslation()

  const dayjsLocale =
    DAYJS_LOCALE_MAP[(i18n.language || 'en').split('-')[0]] || 'en'

  const [timeRange, setTimeRange] = useState({ from: 'now-1h', to: 'now' })
  const [timeRangeLabel, setTimeRangeLabel] = useState('monitoring.time.1h')
  const [customDisplay, setCustomDisplay] = useState('')
  const [customFrom, setCustomFrom] = useState(null)
  const [customTo, setCustomTo] = useState(null)
  const [refreshInterval, setRefreshInterval] = useState(() => {
    const saved = sessionStorage.getItem('monitoring_refresh_interval')
    return saved ? Number(saved) : 60000
  })
  const [refreshLabel, setRefreshLabel] = useState(
    () =>
      sessionStorage.getItem('monitoring_refresh_label') ||
      'monitoring.refresh.1m'
  )
  const [grafanaRefresh, setGrafanaRefresh] = useState(
    () => sessionStorage.getItem('monitoring_grafana_refresh') || '1m'
  )
  const [refreshKey, setRefreshKey] = useState(0)

  const [timeAnchor, setTimeAnchor] = useState(null)
  const [refreshAnchor, setRefreshAnchor] = useState(null)

  const timerRef = useRef(null)

  useEffect(() => {
    fetch(endPoint + '/v1/cluster/ui_config')
      .then((res) => res.json())
      .then((data) => setConfig(data))
      .catch(() => setConfig(null))
  }, [endPoint])

  // Auto refresh
  useEffect(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (refreshInterval > 0) {
      timerRef.current = setInterval(
        () => setRefreshKey((k) => k + 1),
        refreshInterval
      )
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [refreshInterval])

  const handleQuickRange = (item) => {
    setTimeRange({ from: item.from, to: item.to })
    setTimeRangeLabel(item.labelKey)
    setTimeAnchor(null)
    setRefreshKey((k) => k + 1)
  }

  const handleApplyAbsolute = () => {
    if (
      customFrom &&
      customTo &&
      customFrom.isValid() &&
      customTo.isValid() &&
      customTo.isAfter(customFrom)
    ) {
      const fromMs = customFrom.valueOf()
      const toMs = customTo.valueOf()
      setTimeRange({ from: String(fromMs), to: String(toMs) })
      setTimeRangeLabel(null)
      setCustomDisplay(
        `${customFrom.format('YYYY-MM-DD HH:mm')} ~ ${customTo.format(
          'YYYY-MM-DD HH:mm'
        )}`
      )
      setTimeAnchor(null)
      setRefreshKey((k) => k + 1)
    }
  }

  if (!config || !config.grafana_url) {
    return (
      <Box
        sx={{
          width: '100%',
          height: 'calc(100vh - 64px)',
          padding: '20px 20px 0 20px',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Title title={t('menu.monitoring')} />
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          sx={{ flex: 1 }}
        >
          <Typography variant="h6" color="text.secondary">
            {t('monitoring.notConfigured')}
          </Typography>
        </Box>
      </Box>
    )
  }

  const src =
    buildGrafanaUrl(
      config,
      themeMode,
      timeRange.from,
      timeRange.to,
      grafanaRefresh
    ) + `&_t=${refreshKey}`

  return (
    <Box
      sx={{
        width: '100%',
        height: 'calc(100vh - 64px)',
        padding: '20px 20px 0 20px',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Title title={t('menu.monitoring')} />
      {/* Toolbar */}
      <Toolbar
        variant="dense"
        sx={{
          minHeight: 40,
          px: 2,
          justifyContent: 'flex-end',
          gap: 1,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        {/* Time Range */}
        <Button
          size="small"
          color="inherit"
          startIcon={<AccessTime fontSize="small" />}
          onClick={(e) => setTimeAnchor(e.currentTarget)}
          sx={{ textTransform: 'none', fontSize: FONT_SIZE }}
        >
          {timeRangeLabel ? t(timeRangeLabel) : customDisplay}
        </Button>
        <Popover
          anchorEl={timeAnchor}
          open={Boolean(timeAnchor)}
          onClose={() => setTimeAnchor(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Box sx={{ display: 'flex', width: 480, maxHeight: 400 }}>
            {/* Left: Absolute time range */}
            <Box
              sx={{
                width: 240,
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                gap: 1.5,
              }}
            >
              <Typography
                variant="subtitle2"
                color="text.secondary"
                sx={{ fontSize: FONT_SIZE }}
              >
                {t('monitoring.absoluteRange')}
              </Typography>
              <LocalizationProvider
                dateAdapter={AdapterDayjs}
                adapterLocale={dayjsLocale}
                localeText={
                  PICKER_LOCALE_TEXT[(i18n.language || 'en').split('-')[0]] ||
                  PICKER_LOCALE_TEXT.en
                }
              >
                <DateTimePicker
                  label={t('monitoring.from')}
                  value={customFrom}
                  onChange={(v) => setCustomFrom(v)}
                  ampm={false}
                  slotProps={DATE_TIME_SLOT_PROPS}
                />
                <DateTimePicker
                  label={t('monitoring.to')}
                  value={customTo}
                  onChange={(v) => setCustomTo(v)}
                  ampm={false}
                  slotProps={DATE_TIME_SLOT_PROPS}
                />
              </LocalizationProvider>
              <Button
                variant="contained"
                size="small"
                onClick={handleApplyAbsolute}
                disabled={!customFrom || !customTo}
                fullWidth
                sx={{ fontSize: FONT_SIZE }}
              >
                {t('monitoring.applyTimeRange')}
              </Button>
            </Box>

            <Divider orientation="vertical" flexItem />

            {/* Right: Quick ranges */}
            <Box sx={{ width: 240, overflow: 'auto' }}>
              <MenuList dense disablePadding>
                {TIME_RANGES.map((item) => (
                  <MenuItem
                    key={item.labelKey}
                    selected={item.labelKey === timeRangeLabel}
                    onClick={() => handleQuickRange(item)}
                    sx={{ fontSize: FONT_SIZE }}
                  >
                    {t(item.labelKey)}
                  </MenuItem>
                ))}
              </MenuList>
            </Box>
          </Box>
        </Popover>

        {/* Manual Refresh */}
        <Tooltip title={t('monitoring.manualRefresh')}>
          <IconButton
            size="small"
            color="inherit"
            onClick={() => setRefreshKey((k) => k + 1)}
          >
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {/* Refresh Interval */}
        <Button
          size="small"
          color="inherit"
          startIcon={<RefreshIcon fontSize="small" />}
          onClick={(e) => setRefreshAnchor(e.currentTarget)}
          sx={{ textTransform: 'none', fontSize: FONT_SIZE }}
        >
          {t(refreshLabel)}
        </Button>
        <Menu
          anchorEl={refreshAnchor}
          open={Boolean(refreshAnchor)}
          onClose={() => setRefreshAnchor(null)}
        >
          {REFRESH_OPTIONS.map((item) => (
            <MenuItem
              key={item.labelKey}
              selected={item.labelKey === refreshLabel}
              onClick={() => {
                setRefreshInterval(item.value)
                setRefreshLabel(item.labelKey)
                setGrafanaRefresh(item.grafana)
                sessionStorage.setItem(
                  'monitoring_refresh_interval',
                  item.value
                )
                sessionStorage.setItem(
                  'monitoring_refresh_label',
                  item.labelKey
                )
                sessionStorage.setItem(
                  'monitoring_grafana_refresh',
                  item.grafana
                )
                setRefreshAnchor(null)
              }}
            >
              {t(item.labelKey)}
            </MenuItem>
          ))}
        </Menu>
      </Toolbar>

      {/* Grafana iframe */}
      <Box sx={{ flex: 1 }}>
        <iframe
          src={src}
          width="100%"
          height="100%"
          frameBorder="0"
          title="Xinference Monitoring"
          style={{ border: 'none' }}
        />
      </Box>
    </Box>
  )
}

export default Monitoring
