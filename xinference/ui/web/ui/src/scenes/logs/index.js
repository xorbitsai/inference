import 'dayjs/locale/ja'
import 'dayjs/locale/ko'
import 'dayjs/locale/zh-cn'

import {
  AccessTime,
  AddCircleOutline,
  ArticleOutlined,
  CalendarToday,
  Check as CheckIcon,
  Close as CloseIcon,
  ContentCopy as ContentCopyIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  RemoveCircleOutline,
  Search as SearchIcon,
  Tag as TagIcon,
  TextFields as TextFieldsIcon,
} from '@mui/icons-material'
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  IconButton,
  InputAdornment,
  Menu,
  MenuItem,
  MenuList,
  Popover,
  Select,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
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

const REFRESH_OPTIONS = [
  { labelKey: 'monitoring.refresh.off', value: 0 },
  { labelKey: 'monitoring.refresh.10s', value: 10000 },
  { labelKey: 'monitoring.refresh.30s', value: 30000 },
  { labelKey: 'monitoring.refresh.1m', value: 60000 },
  { labelKey: 'monitoring.refresh.5m', value: 300000 },
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

const LEVEL_COLORS = {
  ERROR: 'error.main',
  WARNING: 'warning.main',
  DEBUG: 'text.disabled',
}

function copyToClipboard(text) {
  if (navigator.clipboard) {
    return navigator.clipboard.writeText(text)
  }
  const textarea = document.createElement('textarea')
  textarea.value = text
  document.body.appendChild(textarea)
  textarea.select()
  document.execCommand('copy')
  document.body.removeChild(textarea)
  return Promise.resolve()
}

function HighlightText({ text, keywords }) {
  if (!text) return ''
  const validKeywords = (
    Array.isArray(keywords) ? keywords : [keywords]
  ).filter((k) => k && typeof k === 'string' && k.trim())
  if (!validKeywords.length) return text
  const escaped = validKeywords
    .sort((a, b) => b.length - a.length)
    .map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')
  const parts = String(text).split(new RegExp(`(${escaped})`, 'gi'))
  const lowerSet = new Set(validKeywords.map((k) => k.toLowerCase()))
  return parts.map((part, i) =>
    lowerSet.has(part.toLowerCase()) ? (
      <mark key={i} style={{ background: '#ffe082', padding: 0 }}>
        {part}
      </mark>
    ) : (
      part
    )
  )
}

function FieldTypeIcon({ fieldKey, value }) {
  if (fieldKey === '@timestamp')
    return <CalendarToday sx={{ fontSize: 14, color: 'text.disabled' }} />
  if (typeof value === 'number')
    return <TagIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
  return <TextFieldsIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
}

function formatFieldValue(value) {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function DetailRow({
  row,
  onFilter,
  fieldFilters,
  appliedSearch,
  selectedLevels,
  selectedLogType,
  endPoint,
  onViewContext,
}) {
  const [tab, setTab] = useState(0)
  const [copied, setCopied] = useState(false)
  const copyTimerRef = useRef(null)
  const { t } = useTranslation()

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current)
    }
  }, [])

  const handleCopyJson = () => {
    copyToClipboard(JSON.stringify(row, null, 2)).then(() => {
      setCopied(true)
      copyTimerRef.current = setTimeout(() => setCopied(false), 1500)
    })
  }

  const fields = Object.entries(row).filter(
    ([, v]) => v !== undefined && v !== null && v !== ''
  )

  const activeFilterMap = new Map()
  if (fieldFilters) {
    fieldFilters.forEach((f) => {
      if (f.op === '+') {
        if (!activeFilterMap.has(f.key)) activeFilterMap.set(f.key, new Set())
        activeFilterMap.get(f.key).add(f.value)
      }
    })
  }
  if (selectedLevels && selectedLevels.length) {
    if (!activeFilterMap.has('level')) activeFilterMap.set('level', new Set())
    selectedLevels.forEach((v) => activeFilterMap.get('level').add(v))
  }
  if (selectedLogType) {
    if (!activeFilterMap.has('log_type'))
      activeFilterMap.set('log_type', new Set())
    activeFilterMap.get('log_type').add(selectedLogType)
  }

  const [contextOpen, setContextOpen] = useState(false)

  const handleViewContext = (e) => {
    e.stopPropagation()
    if (onViewContext) {
      onViewContext(row)
    } else {
      setContextOpen(true)
    }
  }

  return (
    <Box sx={{ px: 2, pb: 2 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          sx={{
            'minHeight': 32,
            '& .MuiTab-root': {
              minHeight: 32,
              fontSize: FONT_SIZE,
              textTransform: 'none',
              py: 0.5,
            },
          }}
        >
          <Tab label={t('logs.detail.tableTab')} />
          <Tab label={t('logs.detail.jsonTab')} />
        </Tabs>
        <Button
          size="small"
          startIcon={<ArticleOutlined sx={{ fontSize: 16 }} />}
          onClick={handleViewContext}
          sx={{
            fontSize: FONT_SIZE,
            textTransform: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          {t('logs.detail.viewContext')}
        </Button>
      </Box>
      {tab === 0 ? (
        <Table size="small" sx={{ mt: 1 }}>
          <TableHead>
            <TableRow>
              <TableCell
                sx={{ fontSize: FONT_SIZE, width: 80, fontWeight: 600 }}
              >
                {t('logs.detail.action')}
              </TableCell>
              <TableCell
                sx={{ fontSize: FONT_SIZE, width: 200, fontWeight: 600 }}
              >
                {t('logs.detail.field')}
              </TableCell>
              <TableCell sx={{ fontSize: FONT_SIZE, fontWeight: 600 }}>
                {t('logs.detail.value')}
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {fields.map(([key, val]) => {
              const isFilterMatch =
                activeFilterMap.has(key) &&
                activeFilterMap.get(key).has(String(val))
              const valueKeywords = []
              if (appliedSearch) valueKeywords.push(appliedSearch)
              if (isFilterMatch) valueKeywords.push(String(val))

              return (
                <TableRow
                  key={key}
                  hover
                  sx={isFilterMatch ? { bgcolor: '#fff8e1' } : undefined}
                >
                  <TableCell sx={{ fontSize: FONT_SIZE, p: 0.5 }}>
                    <Tooltip title={t('logs.detail.filterFor')}>
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation()
                          if (onFilter) onFilter(key, val, '+')
                        }}
                      >
                        <AddCircleOutline sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title={t('logs.detail.filterOut')}>
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation()
                          if (onFilter) onFilter(key, val, '-')
                        }}
                      >
                        <RemoveCircleOutline sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                  <TableCell sx={{ fontSize: FONT_SIZE }}>
                    <Box
                      sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                    >
                      <FieldTypeIcon fieldKey={key} value={val} />
                      <HighlightText
                        text={key}
                        keywords={appliedSearch ? [appliedSearch] : []}
                      />
                    </Box>
                  </TableCell>
                  <TableCell
                    sx={{
                      fontSize: FONT_SIZE,
                      wordBreak: 'break-all',
                      fontFamily: 'monospace',
                    }}
                  >
                    <HighlightText
                      text={formatFieldValue(val)}
                      keywords={valueKeywords}
                    />
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      ) : (
        <Box
          sx={{
            mt: 1,
            p: 1.5,
            bgcolor: 'action.hover',
            borderRadius: 1,
            overflow: 'auto',
            maxHeight: 400,
            position: 'relative',
          }}
        >
          <Tooltip
            title={copied ? t('logs.detail.copied') : t('logs.detail.copyJson')}
          >
            <IconButton
              size="small"
              onClick={handleCopyJson}
              sx={{ position: 'absolute', top: 4, right: 4 }}
            >
              {copied ? (
                <CheckIcon sx={{ fontSize: 16 }} />
              ) : (
                <ContentCopyIcon sx={{ fontSize: 16 }} />
              )}
            </IconButton>
          </Tooltip>
          <pre
            style={{
              margin: 0,
              fontSize: FONT_SIZE,
              fontFamily: 'monospace',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}
          >
            {JSON.stringify(row, null, 2)}
          </pre>
        </Box>
      )}
      {!onViewContext && (
        <ContextDialog
          open={contextOpen}
          onClose={() => setContextOpen(false)}
          anchorRow={row}
          endPoint={endPoint}
        />
      )}
    </Box>
  )
}

function ContextDialog({ open, onClose, anchorRow, endPoint }) {
  const { t } = useTranslation()
  const [currentAnchor, setCurrentAnchor] = useState(anchorRow)
  const [olderSize, setOlderSize] = useState(5)
  const [newerSize, setNewerSize] = useState(5)
  const [olderLoadCount, setOlderLoadCount] = useState(5)
  const [newerLoadCount, setNewerLoadCount] = useState(5)
  const [older, setOlder] = useState([])
  const [newer, setNewer] = useState([])
  const [hasMoreOlder, setHasMoreOlder] = useState(false)
  const [hasMoreNewer, setHasMoreNewer] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedRow, setExpandedRow] = useState(null)
  const [localFieldFilters, setLocalFieldFilters] = useState([])
  const anchorRef = useRef(null)

  useEffect(() => {
    if (open && anchorRow) setCurrentAnchor(anchorRow)
  }, [open, anchorRow])

  const timestamp = currentAnchor ? currentAnchor['@timestamp'] : null

  useEffect(() => {
    if (!open || !timestamp) return
    setLoading(true)
    setError(null)

    const params = new URLSearchParams()
    params.set('timestamp', timestamp)
    params.set('size', String(Math.max(olderSize, newerSize)))
    if (currentAnchor.node) params.set('node', currentAnchor.node)

    const token = sessionStorage.getItem('token')
    const headers = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`

    fetch(endPoint + '/v1/cluster/logs/context?' + params.toString(), {
      headers,
    })
      .then((res) => {
        if (res.status === 404) {
          setError('notFound')
          return null
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => {
        if (!data) return
        setOlder((data.older || []).slice(0, olderSize))
        setNewer((data.newer || []).slice(0, newerSize))
        setHasMoreOlder(data.has_more_older || false)
        setHasMoreNewer(data.has_more_newer || false)
      })
      .catch(() => setError('fetchError'))
      .finally(() => setLoading(false))
  }, [open, timestamp, olderSize, newerSize, currentAnchor, endPoint])

  useEffect(() => {
    if (!loading && anchorRef.current) {
      anchorRef.current.scrollIntoView({ block: 'center', behavior: 'smooth' })
    }
  }, [loading, older, newer])

  const resetState = () => {
    setOlderSize(5)
    setNewerSize(5)
    setOlderLoadCount(5)
    setNewerLoadCount(5)
    setOlder([])
    setNewer([])
    setError(null)
    setExpandedRow(null)
    setLocalFieldFilters([])
  }

  const handleClose = () => {
    resetState()
    onClose()
  }

  const handleSwitchAnchor = (row) => {
    resetState()
    setCurrentAnchor(row)
  }

  const handleLocalFilter = useCallback((key, value, op) => {
    const valStr = String(value)
    setLocalFieldFilters((prev) => {
      const exists = prev.find(
        (f) => f.key === key && f.value === valStr && f.op === op
      )
      if (exists) return prev.filter((f) => f !== exists)
      return [...prev, { key, value: valStr, op }]
    })
  }, [])

  const clampCount = (val) => Math.max(1, Math.min(500, Number(val) || 1))

  const filterRows = (rows) => {
    if (!localFieldFilters.length) return rows
    const plusByKey = {}
    const minusFilters = []
    localFieldFilters.forEach((f) => {
      if (f.op === '+') {
        if (!plusByKey[f.key]) plusByKey[f.key] = []
        plusByKey[f.key].push(f.value)
      } else {
        minusFilters.push(f)
      }
    })
    return rows.filter((row) => {
      for (const [key, values] of Object.entries(plusByKey)) {
        if (!values.includes(String(row[key]))) return false
      }
      for (const f of minusFilters) {
        if (String(row[f.key]) === f.value) return false
      }
      return true
    })
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

  const colSpan = 5

  const renderLoadBar = (direction) => {
    const isNewer = direction === 'newer'
    const count = isNewer ? newerLoadCount : olderLoadCount
    const setCount = isNewer ? setNewerLoadCount : setOlderLoadCount
    const setSize = isNewer ? setNewerSize : setOlderSize
    const hasMore = isNewer ? hasMoreNewer : hasMoreOlder
    if (!hasMore) return null
    return (
      <TableRow>
        <TableCell colSpan={colSpan} sx={{ py: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Button
              size="small"
              onClick={() => setSize((s) => s + clampCount(count))}
              sx={{ fontSize: FONT_SIZE, textTransform: 'none' }}
            >
              {t(isNewer ? 'logs.detail.loadNewer' : 'logs.detail.loadOlder')}
            </Button>
            <TextField
              size="small"
              type="number"
              value={count}
              onChange={(e) => setCount(clampCount(e.target.value))}
              onBlur={(e) => setCount(clampCount(e.target.value))}
              onKeyDown={(e) => {
                if (e.key === 'Enter') setSize((s) => s + clampCount(count))
              }}
              inputProps={{
                min: 1,
                max: 500,
                style: {
                  fontSize: FONT_SIZE,
                  width: 48,
                  textAlign: 'center',
                  padding: '2px 4px',
                },
              }}
              sx={{ '& .MuiOutlinedInput-root': { height: 28 } }}
            />
            <Typography sx={{ fontSize: FONT_SIZE, color: 'text.secondary' }}>
              {t(isNewer ? 'logs.detail.newerDocs' : 'logs.detail.olderDocs')}
            </Typography>
          </Box>
        </TableCell>
      </TableRow>
    )
  }

  const renderContextRow = (row, rowKey, isAnchor) => {
    const isExpanded = expandedRow === rowKey
    return (
      <React.Fragment key={rowKey}>
        <TableRow
          ref={isAnchor ? anchorRef : undefined}
          sx={{
            ...(isAnchor ? { bgcolor: '#e3f2fd' } : undefined),
            'cursor': 'pointer',
            '& > *': { borderBottom: isExpanded ? 'none' : undefined },
          }}
          hover={!isAnchor}
          onClick={() => setExpandedRow(isExpanded ? null : rowKey)}
        >
          <TableCell sx={{ fontSize: FONT_SIZE, p: 0.5, width: 32 }}>
            <ExpandMoreIcon
              fontSize="small"
              sx={{
                transform: isExpanded ? 'rotate(180deg)' : 'none',
                transition: 'transform 0.2s',
              }}
            />
          </TableCell>
          <TableCell sx={{ fontSize: FONT_SIZE, whiteSpace: 'nowrap' }}>
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
          <TableCell sx={{ fontSize: FONT_SIZE }}>{row.node}</TableCell>
          <TableCell
            sx={{
              fontSize: FONT_SIZE,
              maxWidth: 0,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {row.message}
          </TableCell>
        </TableRow>
        <TableRow>
          <TableCell sx={{ p: 0 }} colSpan={colSpan}>
            <Collapse in={isExpanded} timeout="auto" unmountOnExit>
              <DetailRow
                row={row}
                onFilter={handleLocalFilter}
                fieldFilters={localFieldFilters}
                appliedSearch=""
                selectedLevels={[]}
                selectedLogType=""
                endPoint={endPoint}
                onViewContext={handleSwitchAnchor}
              />
            </Collapse>
          </TableCell>
        </TableRow>
      </React.Fragment>
    )
  }

  const newerDesc = filterRows([...newer].reverse())
  const filteredOlder = filterRows(older)

  return (
    <Dialog open={open} onClose={handleClose} fullWidth maxWidth="lg">
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: FONT_SIZE,
          py: 1.5,
        }}
      >
        {t('logs.detail.contextTitle')}
        <IconButton size="small" onClick={handleClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 0 }}>
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress size={28} />
          </Box>
        )}
        {error === 'notFound' && (
          <Box sx={{ py: 4, textAlign: 'center' }}>
            <Typography color="text.secondary">
              {t('logs.detail.contextError')}
            </Typography>
          </Box>
        )}
        {error === 'fetchError' && (
          <Box sx={{ py: 4, textAlign: 'center' }}>
            <Typography color="error">
              {t('logs.detail.contextFetchError')}
            </Typography>
          </Box>
        )}
        {!loading && !error && (
          <>
            {localFieldFilters.length > 0 && (
              <Box
                sx={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 0.5,
                  px: 2,
                  py: 0.5,
                  borderBottom: 1,
                  borderColor: 'divider',
                  alignItems: 'center',
                }}
              >
                {localFieldFilters.map((f, i) => (
                  <Chip
                    key={`${f.op}${f.key}:${f.value}-${i}`}
                    label={
                      f.op === '-'
                        ? `NOT ${f.key}: ${f.value}`
                        : `${f.key}: ${f.value}`
                    }
                    size="small"
                    color={f.op === '-' ? 'error' : 'default'}
                    variant={f.op === '-' ? 'outlined' : 'filled'}
                    onDelete={() =>
                      setLocalFieldFilters((prev) =>
                        prev.filter((_, idx) => idx !== i)
                      )
                    }
                    sx={{ fontSize: FONT_SIZE }}
                  />
                ))}
                <Chip
                  label={t('logs.clearFilters', 'Clear all')}
                  size="small"
                  variant="outlined"
                  onClick={() => setLocalFieldFilters([])}
                  sx={{ fontSize: FONT_SIZE }}
                />
              </Box>
            )}
            <TableContainer sx={{ maxHeight: '60vh' }}>
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
                  {renderLoadBar('newer')}
                  {newerDesc.map((row, idx) =>
                    renderContextRow(row, `newer-${idx}`, false)
                  )}
                  {currentAnchor &&
                    renderContextRow(currentAnchor, 'anchor', true)}
                  {filteredOlder.map((row, idx) =>
                    renderContextRow(row, `older-${idx}`, false)
                  )}
                  {renderLoadBar('older')}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}
      </DialogContent>
    </Dialog>
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
  const [fieldFilters, setFieldFilters] = useState([])

  const [timeRange, setTimeRange] = useState({ from: 'now-1h', to: 'now' })
  const [timeRangeLabel, setTimeRangeLabel] = useState('monitoring.time.1h')
  const [customDisplay, setCustomDisplay] = useState('')
  const [customFrom, setCustomFrom] = useState(null)
  const [customTo, setCustomTo] = useState(null)
  const [timeAnchor, setTimeAnchor] = useState(null)

  // Auto refresh
  const [refreshInterval, setRefreshInterval] = useState(0)
  const [refreshLabel, setRefreshLabel] = useState('monitoring.refresh.off')
  const [refreshAnchor, setRefreshAnchor] = useState(null)
  const refreshTimerRef = useRef(null)

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
    fieldFilters.forEach((f) => {
      params.append('filters', `${f.op}${f.key}:${String(f.value)}`)
    })

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
    fieldFilters,
  ])

  useEffect(() => {
    fetchLogs()
  }, [fetchLogs])

  // Auto refresh timer
  useEffect(() => {
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current)
      refreshTimerRef.current = null
    }
    if (refreshInterval > 0) {
      refreshTimerRef.current = setInterval(() => fetchLogs(), refreshInterval)
    }
    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current)
    }
  }, [refreshInterval, fetchLogs])

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

  const handleFieldFilter = useCallback((key, value, op) => {
    const valStr = String(value)
    setFieldFilters((prev) => {
      const exists = prev.find(
        (f) => f.key === key && f.value === valStr && f.op === op
      )
      if (exists) return prev.filter((f) => f !== exists)
      return [...prev, { key, value: valStr, op }]
    })
    setPageFrom(0)
  }, [])

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

        {/* Auto Refresh Interval */}
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
                setRefreshAnchor(null)
              }}
              sx={{ fontSize: FONT_SIZE }}
            >
              {t(item.labelKey)}
            </MenuItem>
          ))}
        </Menu>
      </Toolbar>

      {/* Active Field Filters */}
      {fieldFilters.length > 0 && (
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 0.5,
            px: 2,
            py: 0.5,
            borderBottom: 1,
            borderColor: 'divider',
            alignItems: 'center',
          }}
        >
          {fieldFilters.map((f, i) => (
            <Chip
              key={`${f.op}${f.key}:${f.value}-${i}`}
              label={
                f.op === '-'
                  ? `NOT ${f.key}: ${f.value}`
                  : `${f.key}: ${f.value}`
              }
              size="small"
              color={f.op === '-' ? 'error' : 'default'}
              variant={f.op === '-' ? 'outlined' : 'filled'}
              onDelete={() =>
                setFieldFilters((prev) => prev.filter((_, idx) => idx !== i))
              }
              sx={{ fontSize: FONT_SIZE }}
            />
          ))}
          <Chip
            label={t('logs.clearFilters', 'Clear all')}
            size="small"
            variant="outlined"
            onClick={() => setFieldFilters([])}
            sx={{ fontSize: FONT_SIZE }}
          />
        </Box>
      )}

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
            {(() => {
              const levelFilterValues = [
                ...fieldFilters
                  .filter((f) => f.op === '+' && f.key === 'level')
                  .map((f) => f.value),
                ...selectedLevels,
              ]
              const nodeFilterValues = fieldFilters
                .filter((f) => f.op === '+' && f.key === 'node')
                .map((f) => f.value)
              const messageFilterValues = fieldFilters
                .filter((f) => f.op === '+' && f.key === 'message')
                .map((f) => f.value)

              return logs.map((row, idx) => {
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
                          <HighlightText
                            text={row.level}
                            keywords={levelFilterValues}
                          />
                        </Typography>
                      </TableCell>
                      <TableCell sx={{ fontSize: FONT_SIZE }}>
                        <HighlightText
                          text={row.node}
                          keywords={nodeFilterValues}
                        />
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
                          keywords={[appliedSearch, ...messageFilterValues]}
                        />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell sx={{ p: 0 }} colSpan={5}>
                        <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                          <DetailRow
                            row={row}
                            onFilter={handleFieldFilter}
                            fieldFilters={fieldFilters}
                            appliedSearch={appliedSearch}
                            selectedLevels={selectedLevels}
                            selectedLogType={selectedLogType}
                            endPoint={endPoint}
                          />
                        </Collapse>
                      </TableCell>
                    </TableRow>
                  </React.Fragment>
                )
              })
            })()}
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
