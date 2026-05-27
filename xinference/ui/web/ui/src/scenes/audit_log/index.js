import AccessTime from '@mui/icons-material/AccessTime'
import CalendarToday from '@mui/icons-material/CalendarToday'
import CheckIcon from '@mui/icons-material/Check'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import RefreshIcon from '@mui/icons-material/Refresh'
import TagIcon from '@mui/icons-material/Tag'
import TextFieldsIcon from '@mui/icons-material/TextFields'
import {
  Box,
  Button,
  Checkbox,
  Chip,
  Collapse,
  Divider,
  IconButton,
  ListItemText,
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
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import fetchWrapper from '../../components/fetchWrapper'
import Title from '../../components/Title'

const FONT_SIZE = '0.813rem'
const PAGE_SIZE = 50

const TIME_RANGES = [
  { labelKey: 'monitoring.time.15m', from: 'now-15m', to: 'now' },
  { labelKey: 'monitoring.time.1h', from: 'now-1h', to: 'now' },
  { labelKey: 'monitoring.time.6h', from: 'now-6h', to: 'now' },
  { labelKey: 'monitoring.time.24h', from: 'now-24h', to: 'now' },
  { labelKey: 'monitoring.time.2d', from: 'now-2d', to: 'now' },
  { labelKey: 'monitoring.time.7d', from: 'now-7d', to: 'now' },
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
}

const MODEL_TYPES = ['LLM', 'embedding', 'rerank', 'audio', 'image', 'video']

const CATEGORIES = ['inference', 'admin', 'auth']

const REFRESH_OPTIONS = [
  { labelKey: 'monitoring.refresh.off', value: 0 },
  { labelKey: 'monitoring.refresh.10s', value: 10000 },
  { labelKey: 'monitoring.refresh.30s', value: 30000 },
  { labelKey: 'monitoring.refresh.1m', value: 60000 },
  { labelKey: 'monitoring.refresh.5m', value: 300000 },
]

const ALL_STATUSES = [
  'success',
  'denied',
  'error',
  'login_failed',
  'model_not_found',
  'no_credentials',
  'invalid_token',
  'invalid_key',
  'key_expired',
  'key_disabled',
  'ip_banned',
  'key_banned',
  'user_disabled',
  'insufficient_scope',
]

const STATUS_COLORS = {
  success: 'success.main',
  denied: 'error.main',
  error: 'error.main',
  login_failed: 'error.main',
  ip_banned: 'error.main',
  key_banned: 'error.main',
  user_disabled: 'error.main',
  insufficient_scope: 'warning.main',
  model_not_found: 'warning.main',
  no_credentials: 'warning.main',
  invalid_token: 'warning.main',
  invalid_key: 'warning.main',
  key_expired: 'warning.main',
  key_disabled: 'warning.main',
}

const DETAIL_FIELDS = [
  '@timestamp',
  'event_type',
  'category',
  'auth_type',
  'user',
  'api_key_name',
  'api_key_prefix',
  'model_id',
  'model_name',
  'model_type',
  'endpoint',
  'status',
  'latency_ms',
  'client_ip',
  'node',
  'address',
]

const NUMBER_FIELDS = new Set(['latency_ms'])
const TIME_FIELDS = new Set(['@timestamp'])

function FieldTypeIcon({ fieldKey }) {
  if (TIME_FIELDS.has(fieldKey))
    return <CalendarToday sx={{ fontSize: 14, color: 'text.disabled' }} />
  if (NUMBER_FIELDS.has(fieldKey))
    return <TagIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
  return <TextFieldsIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
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

function copyToClipboard(text) {
  if (navigator.clipboard) return navigator.clipboard.writeText(text)
  const ta = document.createElement('textarea')
  ta.value = text
  document.body.appendChild(ta)
  ta.select()
  document.execCommand('copy')
  document.body.removeChild(ta)
  return Promise.resolve()
}

function formatFieldValue(value) {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function buildJsonObj(hit) {
  const obj = {}
  DETAIL_FIELDS.forEach((k) => {
    if (hit[k] !== undefined) obj[k] = hit[k]
  })
  return obj
}

function DetailPanel({ hit, t }) {
  const [tab, setTab] = useState(0)
  const [copied, setCopied] = useState(false)
  const copyTimerRef = useRef(null)

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current)
    }
  }, [])

  const handleCopyJson = () => {
    copyToClipboard(JSON.stringify(buildJsonObj(hit), null, 2)).then(() => {
      setCopied(true)
      copyTimerRef.current = setTimeout(() => setCopied(false), 1500)
    })
  }

  const fields = DETAIL_FIELDS.filter(
    (k) => hit[k] !== undefined && hit[k] !== ''
  )

  return (
    <Box sx={{ px: 2, pb: 2 }}>
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
        <Tab label={t('auditLog.tableTab')} />
        <Tab label={t('auditLog.jsonTab')} />
      </Tabs>

      {tab === 0 ? (
        <Table size="small" sx={{ mt: 1 }}>
          <TableHead>
            <TableRow>
              <TableCell
                sx={{ fontSize: FONT_SIZE, width: 200, fontWeight: 600 }}
              >
                {t('auditLog.detailField')}
              </TableCell>
              <TableCell sx={{ fontSize: FONT_SIZE, fontWeight: 600 }}>
                {t('auditLog.detailValue')}
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {fields.map((key) => (
              <TableRow key={key} hover>
                <TableCell sx={{ fontSize: FONT_SIZE }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <FieldTypeIcon fieldKey={key} />
                    {key}
                  </Box>
                </TableCell>
                <TableCell
                  sx={{
                    fontSize: FONT_SIZE,
                    wordBreak: 'break-all',
                    fontFamily: 'monospace',
                  }}
                >
                  {formatFieldValue(hit[key])}
                </TableCell>
              </TableRow>
            ))}
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
            title={copied ? t('auditLog.copied') : t('auditLog.copyJson')}
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
            {JSON.stringify(buildJsonObj(hit), null, 2)}
          </pre>
        </Box>
      )}
    </Box>
  )
}

function AuditLog() {
  const { t } = useTranslation()
  const [hits, setHits] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [pageFrom, setPageFrom] = useState(0)
  const [expandedRow, setExpandedRow] = useState(null)

  // Time
  const [timeRange, setTimeRange] = useState({ from: 'now-1h', to: 'now' })
  const [timeRangeLabel, setTimeRangeLabel] = useState('monitoring.time.1h')
  const [timeAnchor, setTimeAnchor] = useState(null)
  const [customFrom, setCustomFrom] = useState(null)
  const [customTo, setCustomTo] = useState(null)
  const [customDisplay, setCustomDisplay] = useState('')

  // Filters
  const [selectedTypes, setSelectedTypes] = useState([])
  const [selectedCategories, setSelectedCategories] = useState([])
  const [selectedStatuses, setSelectedStatuses] = useState([])
  const [filterUser, setFilterUser] = useState('')
  const [filterModel, setFilterModel] = useState('')
  const [filterModelName, setFilterModelName] = useState('')
  const [filterKey, setFilterKey] = useState('')
  const [filterClientIp, setFilterClientIp] = useState('')

  // Auto refresh
  const [refreshInterval, setRefreshInterval] = useState(0)
  const [refreshLabel, setRefreshLabel] = useState('monitoring.refresh.off')
  const [refreshAnchor, setRefreshAnchor] = useState(null)
  const refreshTimerRef = useRef(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const params = new URLSearchParams({
        time_from: timeRange.from,
        time_to: timeRange.to,
        page_from: String(pageFrom),
        size: String(PAGE_SIZE),
      })
      if (filterUser) params.set('user', filterUser)
      if (filterKey) params.set('api_key_name', filterKey)
      if (filterModel) params.set('model_id', filterModel)
      if (filterModelName) params.set('model_name', filterModelName)
      if (filterClientIp) params.set('client_ip', filterClientIp)
      if (selectedTypes.length)
        params.set('model_type', selectedTypes.join(','))
      if (selectedCategories.length)
        params.set('category', selectedCategories.join(','))
      if (selectedStatuses.length)
        params.set('status', selectedStatuses.join(','))

      const data = await fetchWrapper.get(
        `/v1/audit/search?${params.toString()}`
      )
      setHits(data.hits || [])
      setTotal(data.total || 0)
    } catch (e) {
      setError(e.message || String(e))
      setHits([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [
    timeRange,
    pageFrom,
    filterUser,
    filterKey,
    filterModel,
    filterModelName,
    filterClientIp,
    selectedTypes,
    selectedCategories,
    selectedStatuses,
  ])

  useEffect(() => {
    fetchData()
  }, [pageFrom, timeRange, selectedTypes, selectedCategories, selectedStatuses])

  // Auto refresh timer
  useEffect(() => {
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current)
      refreshTimerRef.current = null
    }
    if (refreshInterval > 0) {
      refreshTimerRef.current = setInterval(() => fetchData(), refreshInterval)
    }
    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current)
    }
  }, [refreshInterval, fetchData])

  const handleSearch = () => {
    setPageFrom(0)
    setExpandedRow(null)
    fetchData()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch()
  }

  const toggleType = (type) => {
    setSelectedTypes((prev) =>
      prev.includes(type) ? prev.filter((t2) => t2 !== type) : [...prev, type]
    )
    setPageFrom(0)
  }

  const toggleCategory = (cat) => {
    setSelectedCategories((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
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

  const totalPages = Math.ceil(total / PAGE_SIZE)
  const currentPage = Math.floor(pageFrom / PAGE_SIZE) + 1
  const searchKeywords = [
    filterUser,
    filterModel,
    filterModelName,
    filterKey,
    filterClientIp,
  ].filter(Boolean)

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
      <Title title={t('auditLog.title')} />

      {/* Toolbar */}
      <Toolbar
        variant="dense"
        sx={{
          minHeight: 48,
          px: 2,
          gap: 1,
          flexWrap: 'wrap',
        }}
      >
        {/* Status Multi-Select */}
        <Select
          multiple
          displayEmpty
          value={selectedStatuses}
          onChange={(e) => setSelectedStatuses(e.target.value)}
          renderValue={(selected) =>
            selected.length === 0 ? (
              <Typography sx={{ fontSize: FONT_SIZE, color: 'text.disabled' }}>
                {t('auditLog.status')}
              </Typography>
            ) : (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.3 }}>
                {selected.map((v) => (
                  <Chip
                    key={v}
                    label={v}
                    size="small"
                    onDelete={() =>
                      setSelectedStatuses((prev) => prev.filter((s) => s !== v))
                    }
                    onMouseDown={(e) => e.stopPropagation()}
                    sx={{ fontSize: '0.7rem', height: 20 }}
                  />
                ))}
              </Box>
            )
          }
          size="small"
          sx={{ minWidth: 160, fontSize: FONT_SIZE }}
        >
          {ALL_STATUSES.map((s) => (
            <MenuItem key={s} value={s} dense>
              <Checkbox checked={selectedStatuses.includes(s)} size="small" />
              <ListItemText
                primary={s}
                primaryTypographyProps={{ fontSize: FONT_SIZE }}
              />
            </MenuItem>
          ))}
        </Select>

        {/* Text Filters */}
        <TextField
          size="small"
          placeholder={t('auditLog.user')}
          value={filterUser}
          onChange={(e) => setFilterUser(e.target.value)}
          onKeyDown={handleKeyDown}
          sx={{
            'width': 100,
            '& .MuiInputBase-input': { fontSize: FONT_SIZE },
          }}
        />
        <TextField
          size="small"
          placeholder={t('auditLog.modelId')}
          value={filterModel}
          onChange={(e) => setFilterModel(e.target.value)}
          onKeyDown={handleKeyDown}
          sx={{
            'width': 120,
            '& .MuiInputBase-input': { fontSize: FONT_SIZE },
          }}
        />
        <TextField
          size="small"
          placeholder={t('auditLog.modelName')}
          value={filterModelName}
          onChange={(e) => setFilterModelName(e.target.value)}
          onKeyDown={handleKeyDown}
          sx={{
            'width': 120,
            '& .MuiInputBase-input': { fontSize: FONT_SIZE },
          }}
        />
        <TextField
          size="small"
          placeholder={t('auditLog.apiKey')}
          value={filterKey}
          onChange={(e) => setFilterKey(e.target.value)}
          onKeyDown={handleKeyDown}
          sx={{
            'width': 100,
            '& .MuiInputBase-input': { fontSize: FONT_SIZE },
          }}
        />
        <TextField
          size="small"
          placeholder={t('auditLog.clientIp')}
          value={filterClientIp}
          onChange={(e) => setFilterClientIp(e.target.value)}
          onKeyDown={handleKeyDown}
          sx={{
            'width': 120,
            '& .MuiInputBase-input': { fontSize: FONT_SIZE },
          }}
        />
        <Button
          variant="contained"
          size="small"
          onClick={handleSearch}
          sx={{ fontSize: FONT_SIZE, textTransform: 'none' }}
        >
          {t('auditLog.search')}
        </Button>

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
                {t('auditLog.timeFrom')} / {t('auditLog.timeTo')}
              </Typography>
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DateTimePicker
                  label={t('auditLog.timeFrom')}
                  value={customFrom}
                  onChange={(v) => setCustomFrom(v)}
                  ampm={false}
                  slotProps={DATE_TIME_SLOT_PROPS}
                />
                <DateTimePicker
                  label={t('auditLog.timeTo')}
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
                OK
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
        <Tooltip title={t('auditLog.search')}>
          <IconButton size="small" color="inherit" onClick={handleSearch}>
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

      {/* Model Type Chips */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          px: 2,
          py: 0.5,
        }}
      >
        <Typography
          sx={{ fontSize: FONT_SIZE, color: 'text.secondary', mr: 0.5 }}
        >
          {t('auditLog.modelType')}
        </Typography>
        {MODEL_TYPES.map((type) => (
          <Chip
            key={type}
            label={type}
            size="small"
            variant={selectedTypes.includes(type) ? 'filled' : 'outlined'}
            color={selectedTypes.includes(type) ? 'primary' : 'default'}
            onClick={() => toggleType(type)}
            sx={{ fontSize: FONT_SIZE }}
          />
        ))}
      </Box>

      {/* Category Chips */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          px: 2,
          py: 0.5,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Typography
          sx={{ fontSize: FONT_SIZE, color: 'text.secondary', mr: 0.5 }}
        >
          {t('auditLog.category')}
        </Typography>
        {CATEGORIES.map((cat) => (
          <Chip
            key={cat}
            label={t(`auditLog.category_${cat}`)}
            size="small"
            variant={selectedCategories.includes(cat) ? 'filled' : 'outlined'}
            color={selectedCategories.includes(cat) ? 'primary' : 'default'}
            onClick={() => toggleCategory(cat)}
            sx={{ fontSize: FONT_SIZE }}
          />
        ))}
      </Box>

      {/* Error */}
      {error && (
        <Typography color="error" sx={{ px: 2, py: 1, fontSize: FONT_SIZE }}>
          {error}
        </Typography>
      )}

      {/* Table */}
      <TableContainer sx={{ flex: 1, overflow: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ width: 32, fontSize: FONT_SIZE }} />
              <TableCell sx={{ width: 150, fontSize: FONT_SIZE }}>
                {t('auditLog.time')}
              </TableCell>
              <TableCell sx={{ width: 120, fontSize: FONT_SIZE }}>
                {t('auditLog.status')}
              </TableCell>
              <TableCell sx={{ width: 80, fontSize: FONT_SIZE }}>
                {t('auditLog.user')}
              </TableCell>
              <TableCell sx={{ width: 110, fontSize: FONT_SIZE }}>
                {t('auditLog.apiKey')}
              </TableCell>
              <TableCell sx={{ width: 160, fontSize: FONT_SIZE }}>
                {t('auditLog.modelName')}
              </TableCell>
              <TableCell sx={{ width: 160, fontSize: FONT_SIZE }}>
                {t('auditLog.modelId')}
              </TableCell>
              <TableCell sx={{ width: 120, fontSize: FONT_SIZE }}>
                {t('auditLog.clientIp')}
              </TableCell>
              <TableCell sx={{ fontSize: FONT_SIZE }}>
                {t('auditLog.endpoint')}
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {hits.map((hit, idx) => {
              const isExpanded = expandedRow === idx
              const statusColor = STATUS_COLORS[hit.status] || 'text.primary'

              return (
                <React.Fragment key={`${hit['@timestamp']}-${idx}`}>
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
                      {formatTime(hit['@timestamp'])}
                    </TableCell>
                    <TableCell sx={{ fontSize: FONT_SIZE }}>
                      <Typography
                        component="span"
                        sx={{
                          fontSize: FONT_SIZE,
                          fontWeight: 600,
                          color: statusColor,
                        }}
                      >
                        {hit.status}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ fontSize: FONT_SIZE }}>
                      <HighlightText
                        text={hit.user || ''}
                        keywords={searchKeywords}
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
                        text={hit.api_key_name || ''}
                        keywords={searchKeywords}
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
                        text={hit.model_name || ''}
                        keywords={searchKeywords}
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
                        text={hit.model_id || ''}
                        keywords={searchKeywords}
                      />
                    </TableCell>
                    <TableCell
                      sx={{
                        fontSize: FONT_SIZE,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      <HighlightText
                        text={hit.client_ip || ''}
                        keywords={searchKeywords}
                      />
                    </TableCell>
                    <TableCell
                      sx={{
                        fontSize: FONT_SIZE,
                        maxWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        color: 'text.secondary',
                      }}
                    >
                      {hit.endpoint || ''}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ p: 0 }} colSpan={9}>
                      <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                        <DetailPanel hit={hit} t={t} />
                      </Collapse>
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              )
            })}
            {!loading && hits.length === 0 && (
              <TableRow>
                <TableCell colSpan={9} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">
                    {t('auditLog.noData')}
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
          {t('auditLog.totalRecords', { count: total })}
        </Typography>
        <Button
          size="small"
          disabled={pageFrom === 0}
          onClick={() => setPageFrom(Math.max(0, pageFrom - PAGE_SIZE))}
          sx={{ fontSize: FONT_SIZE, textTransform: 'none' }}
        >
          {t('auditLog.prevPage')}
        </Button>
        <Typography variant="body2" sx={{ fontSize: FONT_SIZE }}>
          {currentPage} / {totalPages || 1}
        </Typography>
        <Button
          size="small"
          disabled={pageFrom + PAGE_SIZE >= total}
          onClick={() => setPageFrom(pageFrom + PAGE_SIZE)}
          sx={{ fontSize: FONT_SIZE, textTransform: 'none' }}
        >
          {t('auditLog.nextPage')}
        </Button>
      </Toolbar>
    </Box>
  )
}

export default AuditLog
