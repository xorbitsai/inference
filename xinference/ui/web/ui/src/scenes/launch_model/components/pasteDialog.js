import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
} from '@mui/material'
import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'

const PasteDialog = ({ open, onHandleClose, onHandleCommandLine }) => {
  const { t } = useTranslation()
  const [command, setCommand] = useState('')

  const handleClose = () => {
    onHandleClose()
    setCommand('')
  }

  const parseXinferenceCommand = () => {
    const params = {}
    const peftModelConfig = {
      lora_list: [],
      image_lora_load_kwargs: {},
      image_lora_fuse_kwargs: {},
    }
    const quantizationConfig = {}
    const virtualEnvPackages = []
    const envs = {}

    let newcommand = command.replace('xinference launch', '').trim()
    const args =
      newcommand.match(
        /--[\w-]+(?:\s+(?:"[^"]*"|'[^']*'|(?:[^\s-][^\s]*\s*)+))?/g
      ) || []

    for (const arg of args) {
      const match = arg
        .trim()
        .match(/^--([\w-]+)(?:\s+(?:"([^"]+)"|'([^']+)'|(.+)))?$/)
      if (!match) continue

      const key = match[1]
      const value = match[2] || match[3] || match[4] || ''
      const normalizedKey = key.replace(/-/g, '_')

      if (normalizedKey === 'gpu_idx') {
        params[normalizedKey] = value.split(',').map(Number)
      } else if (normalizedKey === 'lora_modules') {
        const loraPairs = value.split(/\s+/)
        for (let i = 0; i < loraPairs.length; i += 2) {
          const lora_name = loraPairs[i]
          const local_path = loraPairs[i + 1]
          if (lora_name && local_path) {
            peftModelConfig.lora_list.push({ lora_name, local_path })
          }
        }
      } else if (normalizedKey === 'image_lora_load_kwargs') {
        const [load_param, load_value] = value.split(/\s+/)
        peftModelConfig.image_lora_load_kwargs[load_param] = load_value
      } else if (normalizedKey === 'image_lora_fuse_kwargs') {
        const [fuse_param, fuse_value] = value.split(/\s+/)
        peftModelConfig.image_lora_fuse_kwargs[fuse_param] = fuse_value
      } else if (normalizedKey === 'quantization_config') {
        const [k, v] = value.split(/\s+/)
        quantizationConfig[k] = v
      } else if (key === 'enable-virtual-env') {
        params.enable_virtual_env = true
      } else if (key === 'disable-virtual-env') {
        params.enable_virtual_env = false
      } else if (normalizedKey === 'virtual_env_package') {
        virtualEnvPackages.push(value)
      } else if (normalizedKey === 'env') {
        const [envKey, envVal] = value.split(/\s+/)
        if (envKey && envVal !== undefined) {
          envs[envKey] = envVal
        }
      } else {
        if (['cpu_offload', 'reasoning_content'].includes(normalizedKey)) {
          params[normalizedKey] = value === 'true'
        } else if (normalizedKey === 'size_in_billions') {
          params['model_size_in_billions'] = value
        } else {
          params[normalizedKey] = value
        }
      }
    }

    if (
      peftModelConfig.lora_list.length > 0 ||
      Object.keys(peftModelConfig.image_lora_load_kwargs).length > 0 ||
      Object.keys(peftModelConfig.image_lora_fuse_kwargs).length > 0
    ) {
      params.peft_model_config = peftModelConfig
    }

    if (Object.keys(quantizationConfig).length > 0) {
      params.quantization_config = quantizationConfig
    }

    if (virtualEnvPackages.length > 0) {
      params.virtual_env_packages = virtualEnvPackages
    }

    if (Object.keys(envs).length > 0) {
      params.envs = envs
    }

    onHandleCommandLine(params)
    handleClose()
  }

  return (
    <Dialog open={open}>
      <DialogTitle>{t('launchModel.commandLineParsing')}</DialogTitle>
      <DialogContent>
        <div style={{ width: '500px', height: '120px' }}>
          <TextField
            multiline
            fullWidth
            rows={5}
            placeholder={t('launchModel.placeholderTip')}
            value={command}
            onChange={(e) => {
              setCommand(e.target.value)
            }}
          />
        </div>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>{t('launchModel.cancel')}</Button>
        <Button autoFocus onClick={parseXinferenceCommand}>
          {t('launchModel.confirm')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default PasteDialog
