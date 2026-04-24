import ContentCopyOutlinedIcon from '@mui/icons-material/ContentCopyOutlined'
import DeleteOutlineOutlinedIcon from '@mui/icons-material/DeleteOutlineOutlined'
import OpenInBrowserOutlinedIcon from '@mui/icons-material/OpenInBrowserOutlined'
import { TabContext, TabList, TabPanel } from '@mui/lab'
import {
  Badge,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  Paper,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material'
import { DataGrid } from '@mui/x-data-grid'
import React, { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'
import { useLocation, useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import fetcher from '../../components/fetcher'
import fetchWrapper from '../../components/fetchWrapper'
import Title from '../../components/Title'
import { isValidBearerToken } from '../../components/utils'

const tabArr = [
  {
    label: 'model.languageModels',
    value: '/running_models/LLM',
    showPrompt: false,
  },
  {
    label: 'model.embeddingModels',
    value: '/running_models/embedding',
    showPrompt: false,
  },
  {
    label: 'model.rerankModels',
    value: '/running_models/rerank',
    showPrompt: false,
  },
  {
    label: 'model.imageModels',
    value: '/running_models/image',
    showPrompt: false,
  },
  {
    label: 'model.audioModels',
    value: '/running_models/audio',
    showPrompt: false,
  },
  {
    label: 'model.videoModels',
    value: '/running_models/video',
    showPrompt: false,
  },
  {
    label: 'model.flexibleModels',
    value: '/running_models/flexible',
    showPrompt: false,
  },
]

const RunningModels = () => {
  const [tabValue, setTabValue] = React.useState(
    sessionStorage.getItem('runningModelType')
  )
  const [tabList, setTabList] = useState(tabArr)
  const [llmData, setLlmData] = useState([])
  const [embeddingModelData, setEmbeddingModelData] = useState([])
  const [imageModelData, setImageModelData] = useState([])
  const [audioModelData, setAudioModelData] = useState([])
  const [videoModelData, setVideoModelData] = useState([])
  const [rerankModelData, setRerankModelData] = useState([])
  const [flexibleModelData, setFlexibleModelData] = useState([])
  const [replicaDialogOpen, setReplicaDialogOpen] = useState(false)
  const [selectedModelReplicas, setSelectedModelReplicas] = useState([])
  const [selectedModelUid, setSelectedModelUid] = useState('')
  const [removingReplicaId, setRemovingReplicaId] = useState(null)
  const [terminatingModelUids, setTerminatingModelUids] = useState([])
  const [terminateDialog, setTerminateDialog] = useState({
    open: false,
    row: null,
  })
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel, setIsUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()
  const endPoint = useContext(ApiContext).endPoint
  const { t } = useTranslation()
  const location = useLocation()

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue)
    navigate(newValue)
    sessionStorage.setItem('runningModelType', newValue)
  }

  const update = (isCallingApi) => {
    if (
      sessionStorage.getItem('auth') === 'true' &&
      !isValidBearerToken(sessionStorage.getItem('token')) &&
      !isValidBearerToken(cookie.token)
    ) {
      navigate('/login', { replace: true })
      return
    }
    if (isCallingApi) {
      setLlmData([{ id: 'Loading, do not refresh page...', url: 'IS_LOADING' }])
      setEmbeddingModelData([
        { id: 'Loading, do not refresh page...', url: 'IS_LOADING' },
      ])
      setAudioModelData([
        { id: 'Loading, do not refresh page...', url: 'IS_LOADING' },
      ])
      setVideoModelData([
        { id: 'Loading, do not refresh page...', url: 'IS_LOADING' },
      ])
      setImageModelData([
        { id: 'Loading, do not refresh page...', url: 'IS_LOADING' },
      ])
      setRerankModelData([
        { id: 'Loading, do not refresh page...', url: 'IS_LOADING' },
      ])
      setFlexibleModelData([
        { id: 'Loading, do not refresh page...', url: 'IS_LOADING' },
      ])
    } else {
      setIsUpdatingModel(true)

      fetchWrapper
        .get('/v1/models')
        .then((response) => {
          const modelMap = {
            LLM: [],
            embedding: [],
            image: [],
            audio: [],
            video: [],
            rerank: [],
            flexible: [],
          }

          response.data.forEach((model) => {
            const newValue = {
              ...model,
              id: model.id,
              url: model.id,
            }
            if (modelMap[newValue.model_type]) {
              modelMap[newValue.model_type].push(newValue)
            }
          })

          setLlmData(modelMap.LLM)
          setEmbeddingModelData(modelMap.embedding)
          setImageModelData(modelMap.image)
          setAudioModelData(modelMap.audio)
          setVideoModelData(modelMap.video)
          setRerankModelData(modelMap.rerank)
          setFlexibleModelData(modelMap.flexible)

          if (location.pathname.endsWith('/LLM')) {
            const modelOrder = [
              'LLM',
              'embedding',
              'image',
              'audio',
              'video',
              'rerank',
              'flexible',
            ]
            for (const type of modelOrder) {
              if (modelMap[type] && modelMap[type].length > 0) {
                navigate(`/running_models/${type}`)
                setTabValue(`/running_models/${type}`)
                break
              }
            }
          }

          setIsUpdatingModel(false)
        })
        .catch((error) => {
          console.error('Error:', error)
          setIsUpdatingModel(false)
          if (error.response.status !== 403 && error.response.status !== 401) {
            setErrorMsg(error.message)
          }
        })
    }
  }

  const pollUntilModelRemoved = async (modelUid) => {
    const delays = [2000, 4000, 8000]
    const maxAttempts = 48
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await fetchWrapper.get('/v1/models')
        const exists = response.data.some((m) => m.id === modelUid)
        if (!exists) return
      } catch (e) {
        console.error(e)
        break
      }
      const ms = delays[Math.min(attempt, delays.length - 1)]
      await new Promise((resolve) => setTimeout(resolve, ms))
    }
  }

  const handleTerminateClick = (row) => {
    if (isUpdatingModel || row.url === 'IS_LOADING') return
    if (terminatingModelUids.includes(row.id)) return
    setTerminateDialog({ open: true, row })
  }

  const handleCloseTerminateDialog = () => {
    setTerminateDialog({ open: false, row: null })
  }

  const handleConfirmTerminate = async () => {
    const row = terminateDialog.row
    if (!row) return
    handleCloseTerminateDialog()
    const modelUid = row.id
    const closeUrl = `${endPoint}/v1/models/${modelUid}`
    setTerminatingModelUids((prev) =>
      prev.includes(modelUid) ? prev : [...prev, modelUid]
    )
    try {
      const res = await fetcher(closeUrl, { method: 'DELETE' })
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || res.statusText || String(res.status))
      }
      await res.json().catch(() => ({}))
      await pollUntilModelRemoved(modelUid)
    } catch (e) {
      console.error(e)
      setErrorMsg(t('runningModels.terminateErrorHint'))
    } finally {
      setTerminatingModelUids((prev) => prev.filter((id) => id !== modelUid))
      update(false)
    }
  }

  const renderTerminateButton = (row) => {
    const busy = terminatingModelUids.includes(row.id)
    return (
      <IconButton
        title={
          busy
            ? t('runningModels.terminateInProgress')
            : t('runningModels.terminateModel')
        }
        disabled={busy || isUpdatingModel || row.url === 'IS_LOADING'}
        style={{
          borderWidth: '0px',
          backgroundColor: 'transparent',
          paddingLeft: '0px',
          paddingRight: '10px',
        }}
        onClick={() => handleTerminateClick(row)}
      >
        <Box
          width="40px"
          m="0 auto"
          p="5px"
          display="flex"
          justifyContent="center"
          alignItems="center"
          borderRadius="4px"
          style={{
            border: '1px solid #e5e7eb',
            borderWidth: '1px',
            borderColor: '#e5e7eb',
            minHeight: 32,
          }}
        >
          {busy ? (
            <CircularProgress size={18} thickness={5} />
          ) : (
            <DeleteOutlineOutlinedIcon />
          )}
        </Box>
      </IconButton>
    )
  }

  const loadReplicaDetails = async (modelUid) => {
    const replicas = await fetchWrapper.get(`/v1/models/${modelUid}/replicas`)
    setSelectedModelReplicas(replicas)
    return replicas
  }

  const handleViewReplicas = async (modelUid) => {
    try {
      await loadReplicaDetails(modelUid)
      setSelectedModelUid(modelUid)
      setReplicaDialogOpen(true)
    } catch (error) {
      console.error('Error fetching replica details:', error)
      setErrorMsg('Failed to load replica details: ' + error.message)
    }
  }

  const handleRemoveReplica = async (replica) => {
    const confirmed = window.confirm(
      t('modelReplicaDetails.removeConfirm', {
        replicaId: replica.replica_id,
        modelUid: selectedModelUid,
      })
    )
    if (!confirmed) {
      return
    }

    try {
      setRemovingReplicaId(replica.replica_id)
      const response = await fetchWrapper.delete(
        `/v1/models/${selectedModelUid}/replicas/${replica.replica_id}`
      )

      if (response.remaining_replicas > 0) {
        await loadReplicaDetails(selectedModelUid)
      } else {
        setSelectedModelReplicas([])
        setReplicaDialogOpen(false)
      }

      update(false)
    } catch (error) {
      console.error('Error removing replica:', error)
      setErrorMsg('Failed to remove replica: ' + error.message)
    } finally {
      setRemovingReplicaId(null)
    }
  }

  useEffect(() => {
    update(isCallingApi)
    // eslint-disable-next-line
  }, [isCallingApi, cookie.token])

  const replicaColumn = {
    field: 'replica',
    headerName: t('runningModels.replica'),
    flex: 1,
    renderCell: ({ row }) => {
      return (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '100%',
            paddingRight: 2,
          }}
        >
          <span>{row.replica}</span>
          <Button
            size="small"
            variant="text"
            onClick={(e) => {
              e.stopPropagation()
              handleViewReplicas(row.id)
            }}
            sx={{ minWidth: 'auto', padding: '4px 8px' }}
          >
            {t('runningModels.viewDetails')}
          </Button>
        </Box>
      )
    },
  }

  const llmColumns = [
    {
      field: 'id',
      headerName: 'ID',
      flex: 1,
      minWidth: 250,
      renderCell: ({ row }) => {
        return renderWithCopy(row.id)
      },
    },
    {
      field: 'model_name',
      headerName: t('runningModels.name'),
      flex: 1,
      renderCell: ({ row }) => {
        return renderWithCopy(row.model_name)
      },
    },
    {
      field: 'address',
      headerName: t('runningModels.address'),
      flex: 1,
    },
    {
      field: 'accelerators',
      headerName: t('runningModels.gpuIndexes'),
      flex: 1,
    },
    {
      field: 'model_size_in_billions',
      headerName: t('runningModels.size'),
      flex: 1,
    },
    {
      field: 'quantization',
      headerName: t('runningModels.quantization'),
      flex: 1,
    },
    replicaColumn,
    {
      field: 'url',
      headerName: t('runningModels.actions'),
      flex: 1,
      minWidth: 200,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
      renderCell: ({ row }) => {
        const url = row.url
        const openUrl = `${endPoint}/` + url
        const gradioUrl = `${endPoint}/v1/ui/` + url

        if (url === 'IS_LOADING') {
          return <div></div>
        }

        return (
          <Box
            style={{
              width: '100%',
              display: 'flex',
              justifyContent: 'left',
              alignItems: 'left',
            }}
          >
            <IconButton
              title="Launch Web UI"
              style={{
                borderWidth: '0px',
                backgroundColor: 'transparent',
                paddingLeft: '0px',
                paddingRight: '10px',
              }}
              onClick={() => {
                if (isCallingApi || isUpdatingModel) {
                  // Make sure no ongoing call
                  return
                }

                setIsCallingApi(true)

                fetcher(openUrl, {
                  method: 'HEAD',
                })
                  .then((response) => {
                    if (response.status === 404) {
                      // If web UI doesn't exist (404 Not Found)
                      console.log('UI does not exist, creating new...')
                      return fetcher(gradioUrl, {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                          model_type: row.model_type,
                          model_name: row.model_family,
                          model_size_in_billions: row.model_size_in_billions,
                          model_format: row.model_format,
                          quantization: row.quantization,
                          context_length: row.context_length,
                          model_ability: row.model_ability,
                          model_description: row.model_description,
                          model_lang: row.model_lang,
                        }),
                      })
                        .then((response) => response.json())
                        .then(() =>
                          window.open(openUrl, '_blank', 'noopener noreferrer')
                        )
                        .finally(() => setIsCallingApi(false))
                    } else if (response.ok) {
                      // If web UI does exist
                      console.log('UI exists, opening...')
                      window.open(openUrl, '_blank', 'noopener noreferrer')
                      setIsCallingApi(false)
                    } else {
                      // Other HTTP errors
                      console.error(
                        `Unexpected response status: ${response.status}`
                      )
                      setIsCallingApi(false)
                    }
                  })
                  .catch((error) => {
                    console.error('Error:', error)
                    setIsCallingApi(false)
                  })
              }}
            >
              <Box
                width="40px"
                m="0 auto"
                p="5px"
                display="flex"
                justifyContent="center"
                borderRadius="4px"
                style={{
                  border: '1px solid #e5e7eb',
                  borderWidth: '1px',
                  borderColor: '#e5e7eb',
                }}
              >
                <OpenInBrowserOutlinedIcon />
              </Box>
            </IconButton>
            {renderTerminateButton(row)}
          </Box>
        )
      },
    },
  ]
  const embeddingModelColumns = [
    {
      field: 'id',
      headerName: 'ID',
      flex: 1,
      minWidth: 250,
      renderCell: ({ row }) => {
        return renderWithCopy(row.id)
      },
    },
    {
      field: 'model_name',
      headerName: t('runningModels.name'),
      flex: 1,
      renderCell: ({ row }) => {
        return renderWithCopy(row.model_name)
      },
    },
    {
      field: 'address',
      headerName: t('runningModels.address'),
      flex: 1,
    },
    {
      field: 'accelerators',
      headerName: t('runningModels.gpuIndexes'),
      flex: 1,
    },
    replicaColumn,
    {
      field: 'url',
      headerName: t('runningModels.actions'),
      flex: 1,
      minWidth: 200,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
      renderCell: ({ row }) => {
        const url = row.url

        if (url === 'IS_LOADING') {
          return <div></div>
        }

        return (
          <Box
            style={{
              width: '100%',
              display: 'flex',
              justifyContent: 'left',
              alignItems: 'left',
            }}
          >
            {renderTerminateButton(row)}
          </Box>
        )
      },
    },
  ]
  const imageModelColumns = [
    {
      field: 'id',
      headerName: 'ID',
      flex: 1,
      minWidth: 250,
      renderCell: ({ row }) => {
        return renderWithCopy(row.id)
      },
    },
    {
      field: 'model_name',
      headerName: t('runningModels.name'),
      flex: 1,
      renderCell: ({ row }) => {
        return renderWithCopy(row.model_name)
      },
    },
    {
      field: 'address',
      headerName: t('runningModels.address'),
      flex: 1,
    },
    {
      field: 'accelerators',
      headerName: t('runningModels.gpuIndexes'),
      flex: 1,
    },
    replicaColumn,
    {
      field: 'url',
      headerName: t('runningModels.actions'),
      flex: 1,
      minWidth: 200,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
      renderCell: ({ row }) => {
        // this URL means model_uid
        const url = row.url
        const openUrl = `${endPoint}/` + url
        let pathType
        if (row.model_type === 'video') {
          pathType = 'videos'
        } else if (row.model_type === 'audio') {
          pathType = 'audios'
        } else {
          pathType = 'images' // default
        }
        const gradioUrl = `${endPoint}/v1/ui/${pathType}/` + url

        if (url === 'IS_LOADING') {
          return <div></div>
        }

        return (
          <Box
            style={{
              width: '100%',
              display: 'flex',
              justifyContent: 'left',
              alignItems: 'left',
            }}
          >
            <IconButton
              title="Launch Web UI"
              style={{
                borderWidth: '0px',
                backgroundColor: 'transparent',
                paddingLeft: '0px',
                paddingRight: '10px',
              }}
              onClick={() => {
                if (isCallingApi || isUpdatingModel) {
                  // Make sure no ongoing call
                  return
                }

                setIsCallingApi(true)

                fetcher(openUrl, {
                  method: 'HEAD',
                })
                  .then((response) => {
                    if (response.status === 404) {
                      // If web UI doesn't exist (404 Not Found)
                      console.log('UI does not exist, creating new...')
                      return fetcher(gradioUrl, {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                          model_type: row.model_type,
                          model_family: row.model_family,
                          model_id: row.id,
                          controlnet: row.controlnet,
                          model_revision: row.model_revision,
                          model_name: row.model_name,
                          model_ability: row.model_ability,
                        }),
                      })
                        .then((response) => response.json())
                        .then(() =>
                          window.open(openUrl, '_blank', 'noopener noreferrer')
                        )
                        .finally(() => setIsCallingApi(false))
                    } else if (response.ok) {
                      // If web UI does exist
                      console.log('UI exists, opening...')
                      window.open(openUrl, '_blank', 'noopener noreferrer')
                      setIsCallingApi(false)
                    } else {
                      // Other HTTP errors
                      console.error(
                        `Unexpected response status: ${response.status}`
                      )
                      setIsCallingApi(false)
                    }
                  })
                  .catch((error) => {
                    console.error('Error:', error)
                    setIsCallingApi(false)
                  })
              }}
            >
              <Box
                width="40px"
                m="0 auto"
                p="5px"
                display="flex"
                justifyContent="center"
                borderRadius="4px"
                style={{
                  border: '1px solid #e5e7eb',
                  borderWidth: '1px',
                  borderColor: '#e5e7eb',
                }}
              >
                <OpenInBrowserOutlinedIcon />
              </Box>
            </IconButton>
            {renderTerminateButton(row)}
          </Box>
        )
      },
    },
  ]
  const audioModelColumns = imageModelColumns
  const videoModelColumns = imageModelColumns
  const rerankModelColumns = embeddingModelColumns
  const flexibleModelColumns = embeddingModelColumns

  const dataGridStyle = {
    '& .MuiDataGrid-cell': {
      borderBottom: 'none',
    },
    '& .MuiDataGrid-columnHeaders': {
      borderBottom: 'none',
    },
    '& .MuiDataGrid-columnHeaderTitle': {
      fontWeight: 'bold',
    },
    '& .MuiDataGrid-virtualScroller': {
      overflowX: 'visible !important',
      overflow: 'visible',
    },
    '& .MuiDataGrid-footerContainer': {
      borderTop: 'none',
    },
    'border-width': '0px',
  }

  const noRowsOverlay = () => {
    return (
      <Stack height="100%" alignItems="center" justifyContent="center">
        {t('runningModels.noRunningModels')}
      </Stack>
    )
  }

  const noResultsOverlay = () => {
    return (
      <Stack height="100%" alignItems="center" justifyContent="center">
        {t('runningModels.noRunningModelsMatches')}
      </Stack>
    )
  }

  const renderWithCopy = (display) => {
    const [tooltipOpen, setTooltipOpen] = useState(false)
    const [tooltipText, setTooltipText] = useState(t('runningModels.copy'))

    const handleCopy = (event) => {
      event.stopPropagation()

      if (navigator.clipboard && window.isSecureContext) {
        // for HTTPS
        navigator.clipboard
          .writeText(display)
          .then(() => {
            setTooltipText(t('runningModels.copied'))
          })
          .catch(() => {
            setTooltipText(t('runningModels.copyFailed'))
          })
          .finally(() => {
            setTooltipOpen(true)
            setTimeout(() => {
              setTooltipOpen(false)
              setTooltipText(t('runningModels.copy'))
            }, 1500)
          })
      } else {
        // for HTTP
        const textArea = document.createElement('textarea')
        textArea.value = display
        textArea.style.position = 'absolute'
        textArea.style.left = '-9999px'
        document.body.appendChild(textArea)
        textArea.select()

        try {
          const success = document.execCommand('copy')
          if (success) {
            setTooltipText(t('runningModels.copied'))
          } else {
            setTooltipText(t('runningModels.copyFailed'))
          }
        } catch (err) {
          setTooltipText(t('runningModels.copyFailed'))
        }

        document.body.removeChild(textArea)

        setTooltipOpen(true)
        setTimeout(() => {
          setTooltipOpen(false)
          setTooltipText(t('runningModels.copy'))
        }, 1500)
      }
    }

    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          maxWidth: '100%',
          overflow: 'hidden',
        }}
      >
        <span
          style={{
            flex: 1,
            minWidth: 0,
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
            marginRight: 8,
          }}
        >
          {display}
        </span>
        <Tooltip title={tooltipText} open={tooltipOpen}>
          <IconButton size="small" onClick={handleCopy}>
            <ContentCopyOutlinedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </div>
    )
  }

  useEffect(() => {
    const dataMap = {
      'model.languageModels': llmData,
      'model.embeddingModels': embeddingModelData,
      'model.rerankModels': rerankModelData,
      'model.imageModels': imageModelData,
      'model.audioModels': audioModelData,
      'model.videoModels': videoModelData,
      'model.flexibleModels': flexibleModelData,
    }

    setTabList(
      tabList.map((item) => {
        if (dataMap[item.label]?.length && dataMap[item.label][0].model_type) {
          return {
            ...item,
            showPrompt: true,
          }
        }
        return {
          ...item,
          showPrompt: false,
        }
      })
    )
  }, [
    llmData,
    embeddingModelData,
    rerankModelData,
    imageModelData,
    audioModelData,
    videoModelData,
    flexibleModelData,
  ])

  return (
    <Box
      sx={{
        height: '100%',
        width: '100%',
        padding: '20px 20px 0 20px',
      }}
    >
      <Title title={t('menu.runningModels')} />
      <ErrorMessageSnackBar />
      <TabContext value={tabValue}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList
            value={tabValue}
            onChange={handleTabChange}
            aria-label="tabs"
          >
            {tabList.map((item) => (
              <Tab
                key={item.value}
                label={
                  <Badge
                    color="secondary"
                    variant="dot"
                    invisible={!item.showPrompt}
                  >
                    {t(item.label)}
                  </Badge>
                }
                value={item.value}
              />
            ))}
          </TabList>
        </Box>
        <TabPanel value="/running_models/LLM" sx={{ padding: 0 }}>
          <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
              rows={llmData}
              columns={llmColumns}
              autoHeight={true}
              sx={dataGridStyle}
              slots={{
                noRowsOverlay: noRowsOverlay,
                noResultsOverlay: noResultsOverlay,
              }}
            />
          </Box>
        </TabPanel>
        <TabPanel value="/running_models/embedding" sx={{ padding: 0 }}>
          <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
              rows={embeddingModelData}
              columns={embeddingModelColumns}
              autoHeight={true}
              sx={dataGridStyle}
              slots={{
                noRowsOverlay: noRowsOverlay,
                noResultsOverlay: noResultsOverlay,
              }}
            />
          </Box>
        </TabPanel>
        <TabPanel value="/running_models/rerank" sx={{ padding: 0 }}>
          <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
              rows={rerankModelData}
              columns={rerankModelColumns}
              autoHeight={true}
              sx={dataGridStyle}
              slots={{
                noRowsOverlay: noRowsOverlay,
                noResultsOverlay: noResultsOverlay,
              }}
            />
          </Box>
        </TabPanel>
        <TabPanel value="/running_models/image" sx={{ padding: 0 }}>
          <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
              rows={imageModelData}
              columns={imageModelColumns}
              autoHeight={true}
              sx={dataGridStyle}
              slots={{
                noRowsOverlay: noRowsOverlay,
                noResultsOverlay: noResultsOverlay,
              }}
            />
          </Box>
        </TabPanel>
        <TabPanel value="/running_models/audio" sx={{ padding: 0 }}>
          <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
              rows={audioModelData}
              columns={audioModelColumns}
              autoHeight={true}
              sx={dataGridStyle}
              slots={{
                noRowsOverlay: noRowsOverlay,
                noResultsOverlay: noResultsOverlay,
              }}
            />
          </Box>
        </TabPanel>
        <TabPanel value="/running_models/video" sx={{ padding: 0 }}>
          <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
              rows={videoModelData}
              columns={videoModelColumns}
              autoHeight={true}
              sx={dataGridStyle}
              slots={{
                noRowsOverlay: noRowsOverlay,
                noResultsOverlay: noResultsOverlay,
              }}
            />
          </Box>
        </TabPanel>
        <TabPanel value="/running_models/flexible" sx={{ padding: 0 }}>
          <Box sx={{ height: '100%', width: '100%' }}>
            <DataGrid
              rows={flexibleModelData}
              columns={flexibleModelColumns}
              autoHeight={true}
              sx={dataGridStyle}
              slots={{
                noRowsOverlay: noRowsOverlay,
                noResultsOverlay: noResultsOverlay,
              }}
            />
          </Box>
        </TabPanel>
      </TabContext>

      {/* Replica Details Dialog */}
      <Dialog
        open={replicaDialogOpen}
        onClose={() => setReplicaDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {t('modelReplicaDetails.title')}
          <Typography variant="caption" display="block" color="text.secondary">
            {t('modelReplicaDetails.modelUid')}: {selectedModelUid}
          </Typography>
        </DialogTitle>
        <DialogContent>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>{t('modelReplicaDetails.replicaId')}</TableCell>
                  <TableCell>{t('modelReplicaDetails.modelUid')}</TableCell>
                  <TableCell>
                    {t('modelReplicaDetails.workerAddress')}
                  </TableCell>
                  <TableCell>{t('modelReplicaDetails.status')}</TableCell>
                  <TableCell>{t('modelReplicaDetails.createdTime')}</TableCell>
                  <TableCell align="right">
                    {t('modelReplicaDetails.actions')}
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {selectedModelReplicas.map((replica) => (
                  <TableRow key={replica.replica_id}>
                    <TableCell>{replica.replica_id}</TableCell>
                    <TableCell>
                      <Typography
                        variant="caption"
                        sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}
                      >
                        {replica.replica_model_uid}
                      </Typography>
                    </TableCell>
                    <TableCell>{replica.worker_address}</TableCell>
                    <TableCell>
                      <Chip
                        label={replica.status}
                        color={
                          replica.status === 'READY'
                            ? 'success'
                            : replica.status === 'ERROR'
                            ? 'error'
                            : 'default'
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(replica.created_ts * 1000).toLocaleString()}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title={t('modelReplicaDetails.remove')}>
                        <span>
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleRemoveReplica(replica)}
                            disabled={removingReplicaId === replica.replica_id}
                          >
                            <DeleteOutlineOutlinedIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
                {selectedModelReplicas.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <Typography variant="body2" color="text.secondary">
                        {t('modelReplicaDetails.noReplicaInfo')}
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          {selectedModelReplicas.some((r) => r.error_message) && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" color="error">
                {t('modelReplicaDetails.errors')}:
              </Typography>
              {selectedModelReplicas
                .filter((r) => r.error_message)
                .map((replica, idx) => (
                  <Typography
                    key={idx}
                    variant="caption"
                    display="block"
                    color="error"
                  >
                    {t('modelReplicaDetails.replica')} {replica.replica_id}:{' '}
                    {replica.error_message}
                  </Typography>
                ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReplicaDialogOpen(false)}>
            {t('modelReplicaDetails.close')}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={terminateDialog.open}
        onClose={handleCloseTerminateDialog}
        aria-labelledby="terminate-dialog-title"
      >
        <DialogTitle id="terminate-dialog-title">
          {t('runningModels.terminateConfirmTitle')}
        </DialogTitle>
        <DialogContent>
          <DialogContentText component="div">
            <Typography variant="body2" paragraph>
              {t('runningModels.terminateConfirmBody', {
                replica: terminateDialog.row?.replica ?? 1,
              })}
            </Typography>
            {terminateDialog.row?.model_name && (
              <Typography
                variant="caption"
                color="text.secondary"
                display="block"
              >
                {t('runningModels.name')}: {terminateDialog.row.model_name}
              </Typography>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseTerminateDialog}>
            {t('components.cancel')}
          </Button>
          <Button
            onClick={handleConfirmTerminate}
            color="error"
            variant="contained"
            autoFocus
          >
            {t('runningModels.terminateConfirmOk')}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default RunningModels
