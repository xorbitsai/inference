import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import {
  Alert,
  Box,
  Button,
  FormControlLabel,
  Radio,
  RadioGroup,
  TextField,
  Tooltip,
} from '@mui/material'
import React, { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

const modelFormatData = [
  {
    type: 'LLM',
    options: [
      { value: 'pytorch', label: 'PyTorch' },
      { value: 'ggufv2', label: 'GGUF' },
      { value: 'gptq', label: 'GPTQ' },
      { value: 'awq', label: 'AWQ' },
      { value: 'fp8', label: 'FP8' },
      { value: 'mlx', label: 'MLX' },
    ],
  },
  {
    type: 'embedding',
    options: [
      { value: 'pytorch', label: 'PyTorch' },
      { value: 'ggufv2', label: 'GGUF' },
    ],
  },
  {
    type: 'rerank',
    options: [
      { value: 'pytorch', label: 'PyTorch' },
      { value: 'ggufv2', label: 'GGUF' },
    ],
  },
]

const modelUriDefault = {
  LLM: '/path/to/llama',
  embedding: '/path/to/embedding',
  rerank: '/path/to/rerank',
}

const AddModelSpecs = ({
  isJump,
  formData,
  specsDataArr,
  onGetArr,
  scrollRef,
  modelType,
}) => {
  const [count, setCount] = useState(0)
  const [specsArr, setSpecsArr] = useState([])
  const [pathArr, setPathArr] = useState([])
  const [modelSizeAlertId, setModelSizeAlertId] = useState([])
  const [quantizationAlertId, setQuantizationAlertId] = useState([])
  const [isError, setIsError] = useState(false)
  const [isAdd, setIsAdd] = useState(false)
  const { t } = useTranslation()

  useEffect(() => {
    if (isJump) {
      const dataArr = specsDataArr.map((item, index) => {
        const {
          model_uri,
          model_size_in_billions,
          model_format,
          quantization,
          model_file_name_template,
        } = item
        let size = model_size_in_billions
        if (typeof size !== 'number') size = size?.split('_').join('.')

        return {
          id: index,
          model_uri,
          model_size_in_billions: size,
          model_format,
          quantization,
          model_file_name_template,
        }
      })
      setCount(dataArr.length)
      setSpecsArr(dataArr)

      const subPathArr = []
      specsDataArr.forEach((item) => {
        if (item.model_format !== 'ggufv2') {
          subPathArr.push(item.model_uri)
        } else {
          subPathArr.push(item.model_uri + '/' + item.model_file_name_template)
        }
      })
      setPathArr(subPathArr)
    } else {
      setSpecsArr([
        {
          id: count,
          ...formData,
        },
      ])
      setCount(count + 1)
      setPathArr([formData.model_uri])
    }
  }, [specsDataArr])

  useEffect(() => {
    const arr = specsArr.map((item) => {
      const {
        model_uri: uri,
        model_size_in_billions: size,
        model_format: modelFormat,
        quantization,
        model_file_name_template,
      } = item

      let handleSize
      if (modelType === 'LLM') {
        handleSize =
          parseInt(size) === parseFloat(size)
            ? Number(size)
            : size.split('.').join('_')
      }

      let handleQuantization = quantization
      if (modelFormat === 'pytorch') {
        handleQuantization = 'none'
      } else if (handleQuantization === '' && modelFormat === 'ggufv2') {
        handleQuantization = 'default'
      }

      return {
        model_uri: uri,
        ...(modelType === 'LLM' && {
          model_size_in_billions: handleSize,
        }),
        model_format: modelFormat,
        quantization: handleQuantization,
        model_file_name_template,
      }
    })
    setIsError(true)
    if (modelSizeAlertId.length === 0 && quantizationAlertId.length === 0) {
      setIsError(false)
    }
    onGetArr(arr, isError)
    isAdd && handleScrollBottom()
    setIsAdd(false)
  }, [specsArr, isError])

  const handleAddSpecs = () => {
    setCount(count + 1)
    const item = {
      id: count,
      model_uri: modelUriDefault[modelType],
      model_size_in_billions: 7,
      model_format: 'pytorch',
      quantization: '',
    }
    setSpecsArr([...specsArr, item])
    setIsAdd(true)
    setPathArr([...pathArr, modelUriDefault[modelType]])
  }

  const handleUpdateSpecsArr = (index, type, newValue) => {
    if (type === 'model_format') {
      const subPathArr = [...pathArr]
      if (specsArr[index].model_format !== 'ggufv2') {
        pathArr[index] = specsArr[index].model_uri
      } else {
        pathArr[index] =
          specsArr[index].model_uri +
          '/' +
          specsArr[index].model_file_name_template
      }
      setPathArr(subPathArr)
    }

    setSpecsArr(
      specsArr.map((item, subIndex) => {
        if (subIndex === index) {
          if (type === '') {
            return { ...item, [type]: [newValue] }
          } else if (type === 'model_format') {
            if (newValue === 'ggufv2') {
              const { baseDir, filename } = getPathComponents(pathArr[index])
              const obj = {
                ...item,
                model_format: newValue,
                quantization: '',
                model_uri: baseDir,
                model_file_name_template: filename,
              }
              return obj
            } else {
              const { id, model_size_in_billions, model_format } = item
              return {
                id,
                model_uri: pathArr[index],
                model_size_in_billions,
                model_format,
                [type]: newValue,
                quantization: '',
              }
            }
          } else if (type === 'model_uri') {
            const subPathArr = [...pathArr]
            subPathArr[index] = newValue
            setPathArr(subPathArr)
            if (item.model_format === 'ggufv2') {
              const { baseDir, filename } = getPathComponents(newValue)
              const obj = {
                ...item,
                model_uri: baseDir,
                model_file_name_template: filename,
              }
              return obj
            } else {
              return { ...item, [type]: newValue }
            }
          } else {
            return { ...item, [type]: newValue }
          }
        }
        return item
      })
    )
  }

  const handleDeleteSpecs = (index) => {
    setSpecsArr(specsArr.filter((_, subIndex) => index !== subIndex))
  }

  const getPathComponents = (path) => {
    const normalizedPath = path.replace(/\\/g, '/')
    const baseDir = normalizedPath.substring(0, normalizedPath.lastIndexOf('/'))
    const filename = normalizedPath.substring(
      normalizedPath.lastIndexOf('/') + 1
    )
    return { baseDir, filename }
  }

  const handleModelSize = (index, value, id) => {
    setModelSizeAlertId(modelSizeAlertId.filter((item) => item !== id))
    handleUpdateSpecsArr(index, 'model_size_in_billions', value)
    if (value !== '' && (!Number(value) || Number(value) <= 0)) {
      const modelSizeAlertIdArr = Array.from(new Set([...modelSizeAlertId, id]))
      setModelSizeAlertId(modelSizeAlertIdArr)
    }
  }

  const handleQuantization = (model_format, index, value, id) => {
    setQuantizationAlertId(quantizationAlertId.filter((item) => item !== id))
    handleUpdateSpecsArr(index, 'quantization', value)
    if (
      (model_format === 'gptq' ||
        model_format === 'awq' ||
        model_format === 'fp8' ||
        model_format === 'mlx') &&
      value === ''
    ) {
      const quantizationAlertIdArr = Array.from(
        new Set([...quantizationAlertId, id])
      )
      setQuantizationAlertId(quantizationAlertIdArr)
    }
  }

  const handleScrollBottom = () => {
    scrollRef.current.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
        <label style={{ width: '100px' }}>
          {t('registerModel.modelSpecs')}
        </label>
        <Button
          variant="contained"
          size="small"
          endIcon={<AddIcon />}
          className="addBtn"
          onClick={handleAddSpecs}
        >
          {t('registerModel.more')}
        </Button>
      </div>
      <div className="specs_container">
        {specsArr.map((item, index) => (
          <div className="item" key={item.id}>
            <label
              style={{
                paddingLeft: 5,
              }}
            >
              {t('registerModel.modelFormat')}
            </label>
            <RadioGroup
              value={item.model_format}
              onChange={(e) => {
                handleUpdateSpecsArr(index, 'model_format', e.target.value)
                if (
                  e.target.value === 'gptq' ||
                  e.target.value === 'awq' ||
                  e.target.value === 'fp8' ||
                  e.target.value === 'mlx'
                ) {
                  const quantizationAlertIdArr = Array.from(
                    new Set([...quantizationAlertId, item.id])
                  )
                  setQuantizationAlertId(quantizationAlertIdArr)
                } else {
                  setQuantizationAlertId([])
                }
              }}
            >
              <Box sx={styles.checkboxWrapper}>
                {modelFormatData
                  .find((item) => item.type === modelType)
                  .options.map((item) => (
                    <Box key={item.value} sx={{ marginLeft: '10px' }}>
                      <FormControlLabel
                        value={item.value}
                        control={<Radio />}
                        label={item.label}
                      />
                    </Box>
                  ))}
              </Box>
            </RadioGroup>
            <Box padding="15px"></Box>

            <TextField
              error={item.model_uri !== '' ? false : true}
              style={{ minWidth: '60%' }}
              label={t('registerModel.modelPath')}
              size="small"
              value={
                item.model_format !== 'ggufv2'
                  ? item.model_uri
                  : item.model_uri + '/' + item.model_file_name_template
              }
              onChange={(e) => {
                handleUpdateSpecsArr(index, 'model_uri', e.target.value)
              }}
              helperText={t('registerModel.provideModelDirectoryOrFilePath')}
            />
            <Box padding="15px"></Box>

            {modelType === 'LLM' && (
              <>
                <TextField
                  error={Number(item.model_size_in_billions) > 0 ? false : true}
                  label={t('registerModel.modelSizeBillions')}
                  size="small"
                  value={item.model_size_in_billions}
                  onChange={(e) => {
                    handleModelSize(index, e.target.value, item.id)
                  }}
                />
                {modelSizeAlertId.includes(item.id) && (
                  <Alert severity="error">
                    {t('registerModel.enterNumberGreaterThanZero')}
                  </Alert>
                )}
                <Box padding="15px"></Box>
              </>
            )}

            {item.model_format !== 'pytorch' && (
              <>
                <TextField
                  style={{ minWidth: '60%' }}
                  label={
                    item.model_format === 'gptq' ||
                    item.model_format === 'awq' ||
                    item.model_format === 'fp8' ||
                    item.model_format === 'mlx'
                      ? t('registerModel.quantization')
                      : t('registerModel.quantizationOptional')
                  }
                  size="small"
                  value={item.quantization}
                  onChange={(e) => {
                    handleQuantization(
                      item.model_format,
                      index,
                      e.target.value,
                      item.id
                    )
                  }}
                  helperText={
                    item.model_format === 'gptq' ||
                    item.model_format === 'awq' ||
                    item.model_format === 'fp8' ||
                    item.model_format === 'mlx'
                      ? t(
                          'registerModel.carefulQuantizationForModelRegistration'
                        )
                      : ''
                  }
                />
                {item.model_format !== 'ggufv2' &&
                  quantizationAlertId.includes(item.id) &&
                  item.quantization == '' && (
                    <Alert severity="error">
                      {t('registerModel.quantizationCannotBeEmpty')}
                    </Alert>
                  )}
              </>
            )}

            {specsArr.length > 1 && (
              <Tooltip title={t('registerModel.delete')} placement="top">
                <div
                  className="deleteBtn"
                  onClick={() => handleDeleteSpecs(index)}
                >
                  <DeleteIcon className="deleteIcon" />
                </div>
              </Tooltip>
            )}
          </div>
        ))}
      </div>
    </>
  )
}

export default AddModelSpecs

const styles = {
  baseFormControl: {
    width: '100%',
    margin: 'normal',
    size: 'small',
  },
  checkboxWrapper: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    width: '100%',
  },
}
