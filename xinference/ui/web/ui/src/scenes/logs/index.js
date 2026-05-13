import 'dayjs/locale/ja'
import 'dayjs/locale/ko'
import 'dayjs/locale/zh-cn'

import {
  AccessTime,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
} from '@mui/icons-material'
import {
  Box,
  Button,
  Chip,
  Collapse,
  Divider,
  FormControl,
  IconButton,
  InputAdornment,
  MenuItem,
  MenuList,
  Popover,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material'
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker'
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider'
import React, {
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../components/apiContext'
import Title from '../../components/Title'

const FONT_SIZE = '0.813rem'

const DAYJS_LOCALE_MAP = { en: 'en', zh: 'zh-cn', ja: 'ja', ko: 'ko' }

const LEVELS = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
const LOG_TYPES = ['worker', 'supervisor']

const TIME_RANGES = [
  { labelKey: 'monitoring.time.15m', from: 'now-15m', to: 'now' },
  { labelKey: 'monitoring.time.1h', from: 'now-1h', to: 'now' },
  { labelKey: 'monitoring.time.6h', from: 'now-6h', to: 'now' },
  { labelKey: 'monitoring.time.24h', from: 'now-24h', to: 'now' },
  { labelKey: 'monitoring.time.2d', from: 'now-2d', to: 'now' },
  { labelKey: 'monitoring.time.7d', from: 'now-7d', to: 'now' },
]

const PAGE_SIZE = 200

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

const LEVEL_COLORS = {
  ERROR: 'error.main',
  WARNING: 'warning.main',
  DEBUG: 'text.disabled',
}

function HighlightText({ text, keyword }) {
  if (!keyword || !text) return text || ''
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  return parts.map((part, i) =>
    part.toLowerCase() === keyword.toLowerCase() ? (
      <mark key={i} style={{ background: '#ffe082', padding: 0 }}>
        {part}
      </mark>
    ) : (
      part
    )
  )
}

function DetailRow({ row }) {
  const { t } = useTranslation()
  const fields = [
    ['module', row.module],
    ['pid', row.pid],
    ['request_id', row.request_id],
    ['address', row.address],
    ['log_type', row.log_type],
    ['rtf_avg', row.rtf_avg],
    ['time_speech', row.time_speech],
    ['time_escape', row.time_escape],
  ].filter(([, v]) => v !== undefined && v !== null && v !== '')

  return (
    <Box sx={{ p: 2, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
      {fields.map(([key, val]) => (
        <Box key={key} sx={{ minWidth: 180 }}>
          <Typography variant="caption" color="text.secondary">
            {t(`logs.detail.${key}`, key)}
          </Typography>
          <Typography
            variant="body2"
            sx={{ fontSize: FONT_SIZE, wordBreak: 'break-all' }}
          >
            {String(val)}
          </Typography>
        </Box>
      ))}
    </Box>
  )
}

const Logs = () => {
  const { endPoint } = useContext(ApiContext)
  const { t, i18n } = useTranslation()
  const dayjsLocale =
    DAYJS_LOCALE_MAP[(i18n.language || 'en').split('-')[0]] || 'en'

  const [esEnabled, setEsEnabled] = useState(null)
  const [logs, setLogs] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)

  const [searchText, setSearchText] = useState('')
  const [appliedSearch, setAppliedSearch] = useState('')
  const [selectedLevels, setSelectedLevels] = useState([])
  const [selectedLogType, setSelectedLogType] = useState('')
  const [selectedNode, setSelectedNode] = useState('')
  const [nodes, setNodes] = useState([])
  const [pageFrom, setPageFrom] = useState(0)
  const [expandedRow, setExpandedRow] = useState(null)

  const [timeRange, setTimeRange] = useState({ from: 'now-1h', to: 'now' })
  const [timeRangeLabel, setTimeRangeLabel] = useState('monitoring.time.1h')
  const [customDisplay, setCustomDisplay] = useState('')
  const [customFrom, setCustomFrom] = useState(null)
  const [customTo, setCustomTo] = useState(null)
  const [timeAnchor, setTimeAnchor] = useState(null)

  const debounceRef = useRef(null)

  useEffect(() => {
    fetch(endPoint + '/v1/cluster/ui_config')
      .then((res) => {
        if (!res.ok) throw new Error('Network response was not ok')
        return res.json()
      })
      .then((data) => setEsEnabled(data.es_enabled || false))
      .catch(() => setEsEnabled(false))
  }, [endPoint])

  useEffect(() => {
    if (!esEnabled) return
    const token = sessionStorage.getItem('token')
    const headers = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`
    fetch(endPoint + '/v1/cluster/info', { headers })
      .then((res) => res.json())
      .then((data) => {
        const nodeSet = new Set()
        if (Array.isArray(data)) {
          data.forEach((item) => {
            if (item.node) nodeSet.add(item.node)
          })
        }
        setNodes([...nodeSet])
      })
      .catch(() => setNodes([]))
  }, [endPoint, esEnabled])

  const fetchLogs = useCallback(() => {
    if (!esEnabled) return
    setLoading(true)
    const params = new URLSearchParams()
    if (appliedSearch) params.set('q', appliedSearch)
    if (selectedLevels.length) params.set('level', selectedLevels.join(','))
    if (selectedLogType) params.set('log_type', selectedLogType)
    if (selectedNode) params.set('node', selectedNode)
    params.set('time_from', timeRange.from)
    params.set('time_to', timeRange.to)
    params.set('size', String(PAGE_SIZE))
    params.set('page_from', String(pageFrom))

    const token = sessionStorage.getItem('token')
    const headers = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`

    fetch(endPoint + '/v1/cluster/logs?' + params.toString(), { headers })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => {
        setLogs(data.hits || [])
        setTotal(data.total || 0)
      })
      .catch((err) => {
        console.error('Failed to fetch logs:', err)
        setLogs([])
        setTotal(0)
      })
      .finally(() => setLoading(false))
  }, [
    endPoint,
    esEnabled,
    appliedSearch,
    selectedLevels,
    selectedLogType,
    selectedNode,
    timeRange,
    pageFrom,
  ])

  useEffect(() => {
    fetchLogs()
  }, [fetchLogs])

  const handleSearchChange = (e) => {
    const val = e.target.value
    setSearchText(val)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setAppliedSearch(val)
      setPageFrom(0)
    }, 500)
  }

  const handleSearchKeyDown = (e) => {
    if (e.key === 'Enter') {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      setAppliedSearch(searchText)
      setPageFrom(0)
    }
  }

  const toggleLevel = (level) => {
    setSelectedLevels((prev) =>
      prev.includes(level) ? prev.filter((l) => l !== level) : [...prev, level]
    )
    setPageFrom(0)
  }

  const handleQuickRange = (item) => {
    setTimeRange({ from: item.from, to: item.to })
    setTimeRangeLabel(item.labelKey)
    setCustomDisplay('')
    setTimeAnchor(null)
    setPageFrom(0)
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
      setPageFrom(0)
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)
  const currentPage = Math.floor(pageFrom / PAGE_SIZE) + 1

  if (esEnabled === null) return null

  if (!esEnabled) {
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
        <Title title={t('menu.logs')} />
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          sx={{ flex: 1 }}
        >
          <Typography variant="h6" color="text.secondary">
            {t('logs.notConfigured')}
          </Typography>
        </Box>
      </Box>
    )
  }

  const formatTime = (ts) => {
    if (!ts) return ''
    const d = new Date(ts)
    return d.toLocaleString([], {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
  }

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
      <Title title={t('menu.logs')} />
      {/* Toolbar */}
      <Toolbar
        variant="dense"
        sx={{
          minHeight: 48,
          px: 2,
          gap: 1,
          borderBottom: 1,
          borderColor: 'divider',
          flexWrap: 'wrap',
        }}
      >
        {/* Search */}
        <TextField
          size="small"
          placeholder={t('logs.searchPlaceholder')}
          value={searchText}
          onChange={handleSearchChange}
          onKeyDown={handleSearchKeyDown}
          sx={{
            'width': 240,
            '& .MuiInputBase-input': { fontSize: FONT_SIZE },
          }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
          }}
        />

        {/* Level Chips */}
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {LEVELS.map((level) => (
            <Chip
              key={level}
              label={level}
              size="small"
              variant={selectedLevels.includes(level) ? 'filled' : 'outlined'}
              color={
                level === 'ERROR'
                  ? 'error'
                  : level === 'WARNING'
                  ? 'warning'
                  : 'default'
              }
              onClick={() => toggleLevel(level)}
              sx={{ fontSize: FONT_SIZE }}
            />
          ))}
        </Box>

        {/* Log Type Chips */}
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {LOG_TYPES.map((lt) => (
            <Chip
              key={lt}
              label={lt}
              size="small"
              variant={selectedLogType === lt ? 'filled' : 'outlined'}
              onClick={() => {
                setSelectedLogType((prev) => (prev === lt ? '' : lt))
                setPageFrom(0)
              }}
              sx={{ fontSize: FONT_SIZE }}
            />
          ))}
        </Box>

        {/* Node Select */}
        {nodes.length > 0 && (
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <Select
              value={selectedNode}
              onChange={(e) => {
                setSelectedNode(e.target.value)
                setPageFrom(0)
              }}
              displayEmpty
              sx={{ fontSize: FONT_SIZE }}
            >
              <MenuItem value="" sx={{ fontSize: FONT_SIZE }}>
                {t('logs.allNodes')}
              </MenuItem>
              {nodes.map((n) => (
                <MenuItem key={n} value={n} sx={{ fontSize: FONT_SIZE }}>
                  {n}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        <Box sx={{ flex: 1 }} />

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

        {/* Refresh */}
        <Tooltip title={t('logs.refresh')}>
          <IconButton size="small" color="inherit" onClick={fetchLogs}>
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Toolbar>

      {/* Log Table */}
      <TableContainer sx={{ flex: 1, overflow: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ width: 32, fontSize: FONT_SIZE }} />
              <TableCell sx={{ width: 150, fontSize: FONT_SIZE }}>
                {t('logs.time')}
              </TableCell>
              <TableCell sx={{ width: 80, fontSize: FONT_SIZE }}>
                {t('logs.level')}
              </TableCell>
              <TableCell sx={{ width: 160, fontSize: FONT_SIZE }}>
                {t('logs.node')}
              </TableCell>
              <TableCell sx={{ fontSize: FONT_SIZE }}>
                {t('logs.message')}
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {logs.map((row, idx) => {
              const isExpanded = expandedRow === idx
              return (
                <React.Fragment key={idx}>
                  <TableRow
                    hover
                    sx={{
                      'cursor': 'pointer',
                      '& > *': {
                        borderBottom: isExpanded ? 'none' : undefined,
                      },
                    }}
                    onClick={() => setExpandedRow(isExpanded ? null : idx)}
                  >
                    <TableCell sx={{ fontSize: FONT_SIZE, p: 0.5 }}>
                      <ExpandMoreIcon
                        fontSize="small"
                        sx={{
                          transform: isExpanded ? 'rotate(180deg)' : 'none',
                          transition: 'transform 0.2s',
                        }}
                      />
                    </TableCell>
                    <TableCell
                      sx={{ fontSize: FONT_SIZE, whiteSpace: 'nowrap' }}
                    >
                      {formatTime(row['@timestamp'])}
                    </TableCell>
                    <TableCell sx={{ fontSize: FONT_SIZE }}>
                      <Typography
                        component="span"
                        sx={{
                          fontSize: FONT_SIZE,
                          fontWeight: 600,
                          color: LEVEL_COLORS[row.level] || 'text.primary',
                        }}
                      >
                        {row.level}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ fontSize: FONT_SIZE }}>
                      {row.node}
                    </TableCell>
                    <TableCell
                      sx={{
                        fontSize: FONT_SIZE,
                        maxWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      <HighlightText
                        text={row.message}
                        keyword={appliedSearch}
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ p: 0 }} colSpan={5}>
                      <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                        <DetailRow row={row} />
                        {row.message && (
                          <Box sx={{ px: 2, pb: 2 }}>
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              {t('logs.fullMessage')}
                            </Typography>
                            <Typography
                              variant="body2"
                              sx={{
                                fontSize: FONT_SIZE,
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-all',
                                mt: 0.5,
                                p: 1,
                                bgcolor: 'action.hover',
                                borderRadius: 1,
                              }}
                            >
                              <HighlightText
                                text={row.message}
                                keyword={appliedSearch}
                              />
                            </Typography>
                          </Box>
                        )}
                      </Collapse>
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              )
            })}
            {!loading && logs.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">
                    {t('logs.noLogs')}
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      <Toolbar
        variant="dense"
        sx={{
          minHeight: 40,
          px: 2,
          justifyContent: 'flex-end',
          gap: 1,
          borderTop: 1,
          borderColor: 'divider',
        }}
      >
        <Typography variant="body2" sx={{ fontSize: FONT_SIZE }}>
          {t('logs.totalHits', { count: total })}
        </Typography>
        <Button
          size="small"
          disabled={pageFrom === 0}
          onClick={() => setPageFrom(Math.max(0, pageFrom - PAGE_SIZE))}
          sx={{ fontSize: FONT_SIZE, textTransform: 'none' }}
        >
          {t('logs.prevPage')}
        </Button>
        <Typography variant="body2" sx={{ fontSize: FONT_SIZE }}>
          {currentPage} / {totalPages || 1}
        </Typography>
        <Button
          size="small"
          disabled={
            pageFrom + PAGE_SIZE >= total || pageFrom + PAGE_SIZE > 10000
          }
          onClick={() => setPageFrom(pageFrom + PAGE_SIZE)}
          sx={{ fontSize: FONT_SIZE, textTransform: 'none' }}
        >
          {t('logs.nextPage')}
        </Button>
      </Toolbar>
    </Box>
  )
}

export default Logs
