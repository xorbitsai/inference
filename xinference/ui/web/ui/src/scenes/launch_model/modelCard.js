import './styles/modelCardStyle.css'

import {
  ChatOutlined,
  Computer,
  Delete,
  EditNote,
  EditNoteOutlined,
  Grade,
  HelpCenterOutlined,
  StarBorder,
} from '@mui/icons-material'
import {
  Box,
  Chip,
  IconButton,
  Paper,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import DeleteDialog from '../../components/deleteDialog'
import fetchWrapper from '../../components/fetchWrapper'
import TitleTypography from '../../components/titleTypography'
import CachedListDialog from './components/cachedListDialog'
import EditCustomModel from './components/editCustomModelDialog'
import VirtualEnvListDialog from './components/virtualenvListDialog'

const modelAbilityIcons = {
  chat: <ChatOutlined />,
  generate: <EditNoteOutlined />,
  default: <HelpCenterOutlined />,
}

const ModelCard = ({
  modelData,
  modelType,
  is_custom = false,
  onGetCollectionArr,
  onUpdate,
  onClick,
  virtualEnvs = [],
}) => {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { setErrorMsg } = useContext(ApiContext)
  const [isHovered, setIsHovered] = useState(false)
  const [isDeleteCustomModel, setIsDeleteCustomModel] = useState(false)
  const [isJsonShow, setIsJsonShow] = useState(false)
  const [isOpenCachedList, setIsOpenCachedList] = useState(false)
  const [isOpenVirtualenvList, setIsOpenVirtualenvList] = useState(false)
  const [hasVirtualEnv, setHasVirtualEnv] = useState(null) // null = not checked yet

  const isCached = (spec) => {
    if (Array.isArray(spec.cache_status)) {
      return spec.cache_status.some((cs) => cs)
    } else {
      return spec.cache_status === true
    }
  }

  // Check if model has virtual environment using virtualEnvs data from parent
  useEffect(() => {
    if (modelData?.model_name) {
      const hasEnv = virtualEnvs.some(
        (env) => env.model_name === modelData.model_name
      )
      setHasVirtualEnv(hasEnv)
    }
  }, [modelData?.model_name, virtualEnvs])

  // Handle favorite feature
  const handleCollection = (shouldAdd) => {
    const collectionArr =
      JSON.parse(localStorage.getItem('collectionArr')) || []
    const updatedCollection = shouldAdd
      ? [...collectionArr, modelData.model_name]
      : collectionArr.filter((item) => item !== modelData.model_name)

    localStorage.setItem('collectionArr', JSON.stringify(updatedCollection))
    onGetCollectionArr?.(updatedCollection)
  }

  // Retrieve model language array
  const getModelLanguages = () => {
    // Check possible language field names
    const lang = modelData?.model_lang || modelData?.language

    if (!lang) return []
    return Array.isArray(lang) ? lang : [lang]
  }

  // Render generic chip component
  const renderChips = ({ items, variant = 'default', onClick }) => {
    if (!items || items.length === 0) return null

    return items.map((item) => (
      <Chip
        key={item}
        label={item}
        size="small"
        variant={variant}
        onClick={(e) => {
          e?.stopPropagation()
          onClick?.(e)
        }}
      />
    ))
  }

  const renderAbilityChips = () => {
    const abilities = modelData?.model_ability
    if (!abilities) return null

    const abilityArray = Array.isArray(abilities) ? abilities : [abilities]
    return renderChips({ items: abilityArray })
  }

  const renderLanguageChips = () => {
    const languages = getModelLanguages()
    return renderChips({
      items: languages,
      variant: 'outlined',
    })
  }

  const renderCacheChip = () => {
    const hasCachedSpecs = modelData?.model_specs?.some((spec) =>
      isCached(spec)
    )
    if (!hasCachedSpecs) return null

    return (
      <Chip
        label={t('launchModel.manageCachedModels')}
        variant="outlined"
        color="primary"
        size="small"
        deleteIcon={<EditNote />}
        onDelete={handleOpenCachedList}
        onClick={(e) => {
          e?.stopPropagation()
          handleOpenCachedList()
        }}
      />
    )
  }

  const renderVirtualEnvChip = () => {
    // Only show if we've determined the model might have a virtual environment
    if (!hasVirtualEnv) return null

    return (
      <Chip
        label={t('launchModel.manageVirtualEnvironments')}
        variant="outlined"
        color="secondary"
        size="small"
        deleteIcon={<Computer />}
        onDelete={() => setIsOpenVirtualenvList(true)}
        onClick={(e) => {
          e?.stopPropagation()
          setIsOpenVirtualenvList(true)
        }}
      />
    )
  }

  // Check if already favorited
  const isFavorite = () => {
    const collectionArr =
      JSON.parse(localStorage.getItem('collectionArr')) || []
    return collectionArr.includes(modelData.model_name)
  }

  // Render favorite/unfavorite button
  const renderFavoriteButton = () => {
    const favorite = isFavorite()
    const icon = favorite ? (
      <Grade style={{ color: 'rgb(255, 206, 0)' }} />
    ) : (
      <StarBorder />
    )

    const tooltipTitle = favorite
      ? t('launchModel.unfavorite')
      : t('launchModel.favorite')

    return (
      <Tooltip title={tooltipTitle} placement="top">
        <IconButton
          aria-label={favorite ? 'collection' : 'cancellation-of-collections'}
          onClick={(e) => {
            e?.stopPropagation()
            handleCollection(!favorite)
          }}
        >
          {icon}
        </IconButton>
      </Tooltip>
    )
  }

  // Render custom model action buttons
  const renderCustomModelActions = () => (
    <>
      <Tooltip title={t('launchModel.edit')} placement="top">
        <IconButton
          aria-label="show"
          onClick={(e) => {
            e?.stopPropagation()
            setIsJsonShow(true)
          }}
        >
          <EditNote />
        </IconButton>
      </Tooltip>
      <Tooltip title={t('launchModel.delete')} placement="top">
        <IconButton
          aria-label="delete"
          onClick={(e) => {
            e?.stopPropagation()
            setIsDeleteCustomModel(true)
          }}
        >
          <Delete />
        </IconButton>
      </Tooltip>
    </>
  )

  const MetricItem = ({ value, label, icon }) => (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        minWidth: 80,
      }}
    >
      {icon && React.cloneElement(icon, { sx: { fontSize: 20, mb: 0.5 } })}
      {value && (
        <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '1.2em' }}>
          {value}
        </Typography>
      )}
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ fontWeight: 600, fontSize: '0.8em' }}
      >
        {label}
      </Typography>
    </Box>
  )

  // Delete custom model
  const handeCustomDelete = (e) => {
    e.stopPropagation()

    const subType = sessionStorage.getItem('subType').split('/')
    if (subType) {
      subType[3]
      fetchWrapper
        .delete(
          `/v1/model_registrations/${
            subType[3] === 'llm' ? 'LLM' : subType[3]
          }/${modelData.model_name}`
        )
        .then(() => {
          onUpdate(subType[3] === 'llm' ? 'LLM' : subType[3])
          setIsDeleteCustomModel(false)
        })
        .catch((error) => {
          console.error(error)
          if (error.response.status !== 403) {
            setErrorMsg(error.message)
          }
        })
    }
  }

  const handleJsonDataPresentation = () => {
    const arr = sessionStorage.getItem('subType').split('/')
    sessionStorage.setItem(
      'registerModelType',
      `/register_model/${arr[arr.length - 1]}`
    )
    sessionStorage.setItem('customJsonData', JSON.stringify(modelData))
    navigate(`/register_model/${arr[arr.length - 1]}/${modelData.model_name}`)
  }

  const handleOpenCachedList = () => {
    setIsOpenCachedList(true)
  }

  return (
    <>
      <Tooltip title={modelData.model_name} placement="top">
        <Paper
          className="container"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          elevation={isHovered ? 24 : 4}
          onClick={onClick}
        >
          <Box className="descriptionCard">
            <Box>
              <Box className="cardTitle">
                <TitleTypography value={modelData.model_name} />
                <Box className="iconButtonBox">
                  {is_custom
                    ? renderCustomModelActions()
                    : renderFavoriteButton()}
                </Box>
              </Box>

              <Stack
                spacing={1}
                direction="row"
                useFlexGap
                flexWrap="wrap"
                sx={{ marginLeft: 1 }}
              >
                {renderAbilityChips()}
                {renderLanguageChips()}
                {renderCacheChip()}
                {renderVirtualEnvChip()}
              </Stack>
            </Box>

            <Box
              sx={{
                flex: 1,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: 24,
                color: 'text.secondary',
                fontSize: '0.875rem',
                lineHeight: 1.5,
              }}
            >
              {modelData.model_description ? (
                <Box
                  component="p"
                  sx={{
                    m: 0,
                    display: '-webkit-box',
                    WebkitLineClamp: 4,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    fontSize: '14px',
                    padding: '0px 10px',
                  }}
                  title={modelData.model_description}
                >
                  {modelData.model_description}
                </Box>
              ) : (
                isHovered && (
                  <Box
                    component="p"
                    sx={{
                      m: 0,
                      textAlign: 'center',
                      fontStyle: 'italic',
                    }}
                  >
                    {t('launchModel.clickToLaunchModel')}
                  </Box>
                )
              )}
            </Box>

            {modelType === 'LLM' && (
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <MetricItem
                  value={`${Math.floor(modelData.context_length / 1000)}K`}
                  label={t('launchModel.contextLength')}
                />

                <MetricItem
                  icon={
                    modelData.model_ability?.includes('chat')
                      ? modelAbilityIcons.chat
                      : modelData.model_ability?.includes('generate')
                      ? modelAbilityIcons.generate
                      : modelAbilityIcons.default
                  }
                  label={
                    modelData.model_ability?.includes('chat')
                      ? t('launchModel.chatModel')
                      : modelData.model_ability?.includes('generate')
                      ? t('launchModel.generateModel')
                      : t('launchModel.otherModel')
                  }
                />
              </Box>
            )}

            {(modelData.dimensions || modelData.max_tokens) && (
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <Box>
                  {modelData.dimensions && (
                    <MetricItem
                      value={modelData.dimensions}
                      label={t('launchModel.dimensions')}
                    />
                  )}
                </Box>

                <Box>
                  {modelData.max_tokens && (
                    <MetricItem
                      value={modelData.max_tokens}
                      label={t('launchModel.maxTokens')}
                    />
                  )}
                </Box>
              </Box>
            )}
          </Box>
        </Paper>
      </Tooltip>

      <DeleteDialog
        text={t('launchModel.confirmDeleteCustomModel')}
        isDelete={isDeleteCustomModel}
        onHandleIsDelete={() => setIsDeleteCustomModel(false)}
        onHandleDelete={handeCustomDelete}
      />

      <CachedListDialog
        open={isOpenCachedList}
        modelData={modelData}
        modelType={modelType}
        onClose={() => setIsOpenCachedList(false)}
        onUpdate={onUpdate}
      />

      <VirtualEnvListDialog
        open={isOpenVirtualenvList}
        onClose={() => setIsOpenVirtualenvList(false)}
        onUpdate={onUpdate}
        modelData={modelData}
      />

      <EditCustomModel
        open={isJsonShow}
        modelData={modelData}
        onClose={() => setIsJsonShow(false)}
        handleJsonDataPresentation={handleJsonDataPresentation}
      />
    </>
  )
}

export default ModelCard
