import FilterNoneIcon from '@mui/icons-material/FilterNone'
import { Alert, Snackbar, Tooltip } from '@mui/material'
import ClipboardJS from 'clipboard'
import React, { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

function CopyComponent({ modelData, predefinedKeys }) {
  const { t } = useTranslation()
  const [isCopySuccess, setIsCopySuccess] = useState(false)
  const copyRef = useRef(null)

  const generateCommandLineStatement = (params) => {
    const args = Object.entries(params)
      .filter(
        ([key, value]) =>
          (predefinedKeys.includes(key) &&
            value !== null &&
            value !== undefined) ||
          !predefinedKeys.includes(key)
      )
      .map(([key, value]) => {
        if (key === 'gpu_idx' && Array.isArray(value)) {
          return `--gpu-idx ${value.join(',')}`
        } else if (key === 'peft_model_config' && typeof value === 'object') {
          let peftArgs = []
          if (value.lora_list) {
            peftArgs.push(
              ...value.lora_list.map(
                (lora) => `--lora-modules ${lora.lora_name} ${lora.local_path}`
              )
            )
          }
          if (value.image_lora_load_kwargs) {
            peftArgs.push(
              ...Object.entries(value.image_lora_load_kwargs).map(
                ([k, v]) => `--image-lora-load-kwargs ${k} ${v}`
              )
            )
          }
          if (value.image_lora_fuse_kwargs) {
            peftArgs.push(
              ...Object.entries(value.image_lora_fuse_kwargs).map(
                ([k, v]) => `--image-lora-fuse-kwargs ${k} ${v}`
              )
            )
          }
          return peftArgs.join(' ')
        } else if (key === 'quantization_config' && typeof value === 'object') {
          let peftArgs = []
          peftArgs.push(
            ...Object.entries(value).map(
              ([k, v]) =>
                `--quantization-config ${k} ${v === null ? 'none' : v}`
            )
          )
          return peftArgs.join(' ')
        } else if (predefinedKeys.includes(key)) {
          let newKey
          if (key === 'model_size_in_billions') {
            newKey = '--size-in-billions'
          } else {
            newKey = `--${key.replace(/_/g, '-')}`
          }
          return `${newKey} ${
            value === false ? 'false' : value === null ? 'none' : value
          }`
        } else {
          return `--${key} ${
            value === false ? 'false' : value === null ? 'none' : value
          }`
        }
      })
      .join(' ')

    return `xinference launch ${args}`
  }

  useEffect(() => {
    const text = generateCommandLineStatement(modelData)
    const clipboard = new ClipboardJS(copyRef.current, {
      text: () => text,
    })

    clipboard.on('success', (e) => {
      e.clearSelection()
      setIsCopySuccess(true)
    })

    const handleClick = (event) => {
      event.stopPropagation()
      copyRef.current.focus()
    }

    const copy = copyRef.current
    copy.addEventListener('click', handleClick)

    return () => {
      clipboard.destroy()
      copy.removeEventListener('click', handleClick)
    }
  }, [modelData])

  return (
    <div style={{ marginInline: '10px', lineHeight: '14px' }}>
      <Tooltip title={t('launchModel.copyToCommandLine')} placement="top">
        <div ref={copyRef}>
          <FilterNoneIcon className="copyToCommandLine" />
        </div>
      </Tooltip>
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={isCopySuccess}
        autoHideDuration={1500}
        onClose={() => setIsCopySuccess(false)}
      >
        <Alert severity="success" variant="filled" sx={{ width: '100%' }}>
          {t('components.copySuccess')}
        </Alert>
      </Snackbar>
    </div>
  )
}

export default CopyComponent
