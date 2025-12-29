import { Clear, Done, FilterNone } from '@mui/icons-material'
import { Tooltip } from '@mui/material'
import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { copyToClipboard } from '../../../components/utils'

const keyMap = {
  model_size_in_billions: '--size-in-billions',
  download_hub: '--download_hub',
  enable_thinking: '--enable-thinking',
  reasoning_content: '--reasoning_content',
  lightning_version: '--lightning_version',
  lightning_model_path: '--lightning_model_path',
}

const CopyComponent = ({ getData, predefinedKeys }) => {
  const { t } = useTranslation()
  const [copyStatus, setCopyStatus] = useState('pending')

  const generateCommandLineStatement = (params) => {
    const args = Object.entries(params)
      .filter(
        ([key, value]) =>
          (predefinedKeys.includes(key) &&
            value !== null &&
            value !== undefined) ||
          !predefinedKeys.includes(key)
      )
      .flatMap(([key, value]) => {
        if (key === 'gpu_idx' && Array.isArray(value)) {
          return `--gpu-idx ${value.join(',')}`
        } else if (key === 'peft_model_config' && typeof value === 'object') {
          const peftArgs = []
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
          return peftArgs
        } else if (key === 'quantization_config' && typeof value === 'object') {
          return Object.entries(value).map(
            ([k, v]) => `--quantization-config ${k} ${v === null ? 'none' : v}`
          )
        } else if (key === 'envs' && typeof value === 'object') {
          return Object.entries(value).map(([k, v]) => `--env ${k} ${v}`)
        } else if (key === 'virtual_env_packages' && Array.isArray(value)) {
          return value.map((pkg) => `--virtual-env-package ${pkg}`)
        } else if (key === 'enable_virtual_env') {
          if (value === true) return `--enable-virtual-env`
          if (value === false) return `--disable-virtual-env`
          return []
        } else if (key === 'enable_thinking') {
          if (value === true) return `--enable-thinking`
          if (value === false) return `--disable-thinking`
          return []
        } else if (predefinedKeys.includes(key)) {
          const newKey = keyMap[key] ?? `--${key.replace(/_/g, '-')}`
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

  const showTooltipTemporarily = (status) => {
    setCopyStatus(status)
    setTimeout(() => setCopyStatus('pending'), 1500)
  }

  const handleCopy = async (event) => {
    const text = generateCommandLineStatement(getData())
    event.stopPropagation()
    const textToCopy = String(text ?? '')
    const success = await copyToClipboard(textToCopy)
    showTooltipTemporarily(success ? 'success' : 'failed')
  }

  return (
    <>
      {copyStatus === 'pending' ? (
        <Tooltip title={t('launchModel.copyToCommandLine')} placement="top">
          <FilterNone className="copyToCommandLine" onClick={handleCopy} />
        </Tooltip>
      ) : copyStatus === 'success' ? (
        <Done fontSize="small" color="success" />
      ) : (
        <Clear fontSize="small" color="error" />
      )}
    </>
  )
}

export default CopyComponent
