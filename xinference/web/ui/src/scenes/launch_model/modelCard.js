import {
  ChatOutlined,
  EditNoteOutlined,
  HelpCenterOutlined,
  RocketLaunchOutlined,
  UndoOutlined,
} from '@mui/icons-material'
import DeleteIcon from '@mui/icons-material/Delete'
import {
  Box,
  Chip,
  CircularProgress,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
} from '@mui/material'
import IconButton from '@mui/material/IconButton'
import Typography from '@mui/material/Typography'
import React, { useContext, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'

const CARD_HEIGHT = 380
const CARD_WIDTH = 300

const ModelCard = ({ url, modelData, gpuAvailable, is_custom = false }) => {
  const [hover, setHover] = useState(false)
  const [selected, setSelected] = useState(false)
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)
  const navigate = useNavigate()

  // Model parameter selections
  const [modelUID, setModelUID] = useState('')
  const [modelFormat, setModelFormat] = useState('')
  const [modelSize, setModelSize] = useState('')
  const [quantization, setQuantization] = useState('')
  const [nGPU, setNGPU] = useState('auto')

  const [formatOptions, setFormatOptions] = useState([])
  const [sizeOptions, setSizeOptions] = useState([])
  const [quantizationOptions, setQuantizationOptions] = useState([])
  const [customDeleted, setCustomDeleted] = useState(false)

  const range = (start, end) => {
    return new Array(end - start + 1).fill(undefined).map((_, i) => i + start)
  }

  const isCached = (spec) => {
    if (spec.model_format === 'pytorch') {
      return spec.cache_status && spec.cache_status === true
    } else {
      return spec.cache_status && spec.cache_status.some((cs) => cs)
    }
  }

  // model size can be int or string. For string style, "1_8" means 1.8 as an example.
  const convertModelSize = (size) => {
    return size.toString().includes('_') ? size : parseInt(size, 10)
  }

  // UseEffects for parameter selection, change options based on previous selections
  useEffect(() => {
    if (modelData) {
      const modelFamily = modelData.model_specs
      const formats = [...new Set(modelFamily.map((spec) => spec.model_format))]
      setFormatOptions(formats)
    }
  }, [modelData])

  useEffect(() => {
    if (modelFormat && modelData) {
      const modelFamily = modelData.model_specs
      const sizes = [
        ...new Set(
          modelFamily
            .filter((spec) => spec.model_format === modelFormat)
            .map((spec) => spec.model_size_in_billions)
        ),
      ]
      setSizeOptions(sizes)
    }
  }, [modelFormat, modelData])

  useEffect(() => {
    if (modelFormat && modelSize && modelData) {
      const modelFamily = modelData.model_specs
      const quants = [
        ...new Set(
          modelFamily
            .filter(
              (spec) =>
                spec.model_format === modelFormat &&
                spec.model_size_in_billions === convertModelSize(modelSize)
            )
            .flatMap((spec) => spec.quantizations)
        ),
      ]
      setQuantizationOptions(quants)
    }
  }, [modelFormat, modelSize, modelData])

  const launchModel = (url) => {
    if (isCallingApi || isUpdatingModel) {
      return
    }

    setIsCallingApi(true)

    const modelDataWithID = {
      // If user does not fill model_uid, pass null (None) to server and server generates it.
      model_uid: modelUID.trim() === '' ? null : modelUID.trim(),
      model_name: modelData.model_name,
      model_format: modelFormat,
      model_size_in_billions: convertModelSize(modelSize),
      quantization: quantization,
      n_gpu:
        nGPU === '0' ? null : nGPU === 'auto' ? 'auto' : parseInt(nGPU, 10),
    }

    // First fetcher request to initiate the model
    fetcher(url + '/v1/models', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modelDataWithID),
    })
      .then((response) => {
        if (!response.ok) {
          // Assuming the server returns error details in JSON format
          response.json().then((errorData) => {
            setErrorMsg(
              `Server error: ${response.status} - ${
                errorData.detail || 'Unknown error'
              }`
            )
          })
        } else {
          navigate('/running_models')
        }
        setIsCallingApi(false)
      })
      .catch((error) => {
        console.error('Error:', error)
        setIsCallingApi(false)
      })
  }

  const styles = {
    container: {
      display: 'block',
      position: 'relative',
      width: `${CARD_WIDTH}px`,
      height: `${CARD_HEIGHT}px`,
      border: '1px solid #ddd',
      borderRadius: '20px',
      background: 'white',
      overflow: 'hidden',
    },
    containerSelected: {
      display: 'block',
      position: 'relative',
      width: `${CARD_WIDTH}px`,
      height: `${CARD_HEIGHT}px`,
      border: '1px solid #ddd',
      borderRadius: '20px',
      background: 'white',
      overflow: 'hidden',
      boxShadow: '0 0 2px #00000099',
    },
    descriptionCard: {
      position: 'relative',
      top: '-1px',
      left: '-1px',
      width: `${CARD_WIDTH}px`,
      height: `${CARD_HEIGHT}px`,
      border: '1px solid #ddd',
      padding: '20px',
      borderRadius: '20px',
      background: 'white',
    },
    parameterCard: {
      position: 'relative',
      top: `-${CARD_HEIGHT + 1}px`,
      left: '-1px',
      width: `${CARD_WIDTH}px`,
      height: `${CARD_HEIGHT}px`,
      border: '1px solid #ddd',
      padding: '20px',
      borderRadius: '20px',
      background: 'white',
    },
    img: {
      display: 'block',
      margin: '0 auto',
      width: '180px',
      height: '180px',
      objectFit: 'cover',
      borderRadius: '10px',
    },
    h2: {
      margin: '10px 10px',
      fontSize: '20px',
    },
    p: {
      minHeight: '140px',
      fontSize: '14px',
      padding: '0px 10px 15px 10px',
    },
    buttonsContainer: {
      display: 'flex',
      margin: '0 auto',
      marginTop: '15px',
      border: 'none',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    buttonContainer: {
      width: '45%',
      borderWidth: '0px',
      backgroundColor: 'transparent',
      paddingLeft: '0px',
      paddingRight: '0px',
    },
    buttonItem: {
      width: '100%',
      margin: '0 auto',
      padding: '5px',
      display: 'flex',
      justifyContent: 'center',
      borderRadius: '4px',
      border: '1px solid #e5e7eb',
      borderWidth: '1px',
      borderColor: '#e5e7eb',
    },
    instructionText: {
      fontSize: '12px',
      color: '#666666',
      fontStyle: 'italic',
      margin: '10px 0',
      textAlign: 'center',
    },
    slideIn: {
      transform: 'translateX(0%)',
      transition: 'transform 0.2s ease-in-out',
    },
    slideOut: {
      transform: 'translateX(100%)',
      transition: 'transform 0.2s ease-in-out',
    },
    iconRow: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    iconItem: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      margin: '20px',
    },
    boldIconText: {
      fontWeight: 'bold',
      fontSize: '1.2em',
    },
    muiIcon: {
      fontSize: '1.5em',
    },
    smallText: {
      fontSize: '0.8em',
    },
    tagRow: {
      margin: '2px 5px',
    },
  }

  const handeCustomDelete = (e) => {
    e.stopPropagation()
    fetcher(url + `/v1/model_registrations/LLM/${modelData.model_name}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    })
      .then(() => setCustomDeleted(true))
      .catch(console.error)
  }

  // Set two different states based on mouse hover
  return (
    <Box
      style={hover ? styles.containerSelected : styles.container}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => {
        if (!selected && !customDeleted) {
          setSelected(true)
        }
      }}
    >
      {/* First state: show description page */}
      <Box style={styles.descriptionCard}>
        {is_custom && (
          <Stack
            direction="row"
            justifyContent="space-evenly"
            alignItems="center"
            spacing={1}
          >
            <Typography variant="h4" gutterBottom noWrap>
              {modelData.model_name}
            </Typography>
            <IconButton
              aria-label="delete"
              onClick={handeCustomDelete}
              disabled={customDeleted}
            >
              <DeleteIcon />
            </IconButton>
          </Stack>
        )}
        {!is_custom && <h2 style={styles.h2}>{modelData.model_name}</h2>}
        <Stack
          spacing={1}
          direction="row"
          useFlexGap
          flexWrap="wrap"
          sx={{ marginLeft: 1 }}
        >
          {(() => {
            return modelData.model_lang.map((v) => {
              return <Chip label={v} variant="outlined" size="small" />
            })
          })()}
          {(() => {
            if (modelData.model_specs.some((spec) => isCached(spec))) {
              return <Chip label="Cached" variant="outlined" size="small" />
            }
          })()}
          {(() => {
            if (is_custom && customDeleted) {
              return <Chip label="Deleted" variant="outlined" size="small" />
            }
          })()}
        </Stack>
        <p style={styles.p}>{modelData.model_description}</p>

        <div style={styles.iconRow}>
          <div style={styles.iconItem}>
            <span style={styles.boldIconText}>
              {Math.floor(modelData.context_length / 1000)}K
            </span>
            <small style={styles.smallText}>context length</small>
          </div>
          {(() => {
            if (modelData.model_ability.includes('chat')) {
              return (
                <div style={styles.iconItem}>
                  <ChatOutlined style={styles.muiIcon} />
                  <small style={styles.smallText}>chat model</small>
                </div>
              )
            } else if (modelData.model_ability.includes('generate')) {
              return (
                <div style={styles.iconItem}>
                  <EditNoteOutlined style={styles.muiIcon} />
                  <small style={styles.smallText}>generate model</small>
                </div>
              )
            } else {
              return (
                <div style={styles.iconItem}>
                  <HelpCenterOutlined style={styles.muiIcon} />
                  <small style={styles.smallText}>other model</small>
                </div>
              )
            }
          })()}
        </div>
      </Box>
      {/* Second state: show parameter selection page */}
      <Box
        style={
          selected
            ? { ...styles.parameterCard, ...styles.slideIn }
            : { ...styles.parameterCard, ...styles.slideOut }
        }
      >
        <h2 style={styles.h2}>{modelData.model_name}</h2>
        <Box display="flex" flexDirection="column" width="100%" mx="auto">
          <Grid container rowSpacing={0} columnSpacing={1}>
            <Grid item xs={6}>
              <FormControl variant="outlined" margin="normal" fullWidth>
                <InputLabel id="modelFormat-label">Model Format</InputLabel>
                <Select
                  labelId="modelFormat-label"
                  value={modelFormat}
                  onChange={(e) => setModelFormat(e.target.value)}
                  label="Model Format"
                >
                  {formatOptions.map((format) => {
                    const specs = modelData.model_specs.filter(
                      (spec) => spec.model_format === format
                    )
                    const cached = specs.some((spec) => isCached(spec))
                    const displayedFormat = cached
                      ? format + ' (cached)'
                      : format

                    return (
                      <MenuItem key={format} value={format}>
                        {displayedFormat}
                      </MenuItem>
                    )
                  })}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <FormControl
                variant="outlined"
                margin="normal"
                fullWidth
                disabled={!modelFormat}
              >
                <InputLabel id="modelSize-label">Model Size</InputLabel>
                <Select
                  labelId="modelSize-label"
                  value={modelSize}
                  onChange={(e) => setModelSize(e.target.value)}
                  label="Model Size"
                >
                  {sizeOptions.map((size) => {
                    const specs = modelData.model_specs
                      .filter((spec) => spec.model_format === modelFormat)
                      .filter((spec) => spec.model_size_in_billions === size)
                    const cached = specs.some((spec) => isCached(spec))
                    const displayedSize = cached ? size + ' (cached)' : size

                    return (
                      <MenuItem key={size} value={size}>
                        {displayedSize}
                      </MenuItem>
                    )
                  })}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <FormControl
                variant="outlined"
                margin="normal"
                fullWidth
                disabled={!modelFormat || !modelSize}
              >
                <InputLabel id="quantization-label">Quantization</InputLabel>
                <Select
                  labelId="quantization-label"
                  value={quantization}
                  onChange={(e) => setQuantization(e.target.value)}
                  label="Quantization"
                >
                  {quantizationOptions.map((quant, index) => {
                    const specs = modelData.model_specs
                      .filter((spec) => spec.model_format === modelFormat)
                      .filter(
                        (spec) =>
                          spec.model_size_in_billions ===
                          convertModelSize(modelSize)
                      )

                    const cached =
                      modelFormat === 'pytorch'
                        ? specs[0]?.cache_status ?? false === true
                        : specs[0]?.cache_status?.[index] ?? false === true
                    const displayedQuant = cached ? quant + ' (cached)' : quant

                    return (
                      <MenuItem key={quant} value={quant}>
                        {displayedQuant}
                      </MenuItem>
                    )
                  })}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <FormControl
                variant="outlined"
                margin="normal"
                fullWidth
                disabled={!modelFormat || !modelSize || !quantization}
              >
                <InputLabel id="n-gpu-label">N-GPU</InputLabel>
                <Select
                  labelId="n-gpu-label"
                  value={nGPU}
                  onChange={(e) => setNGPU(e.target.value)}
                  label="N-GPU"
                >
                  {['auto']
                    .concat(
                      range(
                        0,
                        modelFormat !== 'pytorch' && modelFormat !== 'gptq' && modelFormat !== 'awq'
                          ? 1
                          : gpuAvailable
                      )
                    )
                    .map((v) => {
                      return (
                        <MenuItem key={v} value={v}>
                          {v}
                        </MenuItem>
                      )
                    })}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl variant="outlined" margin="normal" fullWidth>
                <TextField
                  variant="outlined"
                  value={modelUID}
                  label="(Optional) Model UID, model name by default"
                  onChange={(e) => setModelUID(e.target.value)}
                />
              </FormControl>
            </Grid>
          </Grid>
        </Box>
        <Box style={styles.buttonsContainer}>
          <button
            title="Launch"
            style={styles.buttonContainer}
            onClick={() => launchModel(url, modelData)}
            disabled={
              isCallingApi ||
              isUpdatingModel ||
              !(
                modelFormat &&
                modelSize &&
                modelData &&
                (quantization ||
                  (!modelData.is_builtin && modelFormat !== 'pytorch'))
              )
            }
          >
            {(() => {
              if (isCallingApi || isUpdatingModel) {
                return (
                  <Box
                    style={{ ...styles.buttonItem, backgroundColor: '#f2f2f2' }}
                  >
                    <CircularProgress
                      size="20px"
                      sx={{
                        color: '#000000',
                      }}
                    />
                  </Box>
                )
              } else if (
                !(
                  modelFormat &&
                  modelSize &&
                  modelData &&
                  (quantization ||
                    (!modelData.is_builtin && modelFormat !== 'pytorch'))
                )
              ) {
                return (
                  <Box
                    style={{ ...styles.buttonItem, backgroundColor: '#f2f2f2' }}
                  >
                    <RocketLaunchOutlined size="20px" />
                  </Box>
                )
              } else {
                return (
                  <Box style={styles.buttonItem}>
                    <RocketLaunchOutlined color="#000000" size="20px" />
                  </Box>
                )
              }
            })()}
          </button>
          <button
            title="Go Back"
            style={styles.buttonContainer}
            onClick={() => setSelected(false)}
          >
            <Box style={styles.buttonItem}>
              <UndoOutlined color="#000000" size="20px" />
            </Box>
          </button>
        </Box>
      </Box>
    </Box>
  )
}

export default ModelCard
