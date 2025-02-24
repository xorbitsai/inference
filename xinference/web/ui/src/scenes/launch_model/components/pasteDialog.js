import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  useTheme,
} from '@mui/material'
import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'

const PasteDialog = ({ open, onHandleClose, onHandleCommandLine }) => {
  const { t } = useTranslation()
  const theme = useTheme()
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
    let newcommand = command.replace('xinference launch', '').trim()
    const args =
      newcommand.match(
        /--[\w-]+(?:\s+(?:"[^"]*"|'[^']*'|(?:[^\s-][^\s]*\s*)+))?/g
      ) || []

    for (const arg of args) {
      const match = arg
        .trim()
        .match(/^--([\w-]+)\s+(?:"([^"]+)"|'([^']+)'|(.+))?$/)
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
      } else {
        if (normalizedKey === 'cpu_offload') {
          params[normalizedKey] = value === 'true' ? true : false
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

    onHandleCommandLine(params)
    handleClose()
  }

  return (
    <Dialog open={open}>
      <DialogTitle>{t('launchModel.commandLineParsing')}</DialogTitle>
      <DialogContent>
        <div style={{ width: '500px', height: '120px' }}>
          <textarea
            className="textarea"
            style={{
              height: '100%',
              color: theme.palette.mode === 'dark' && '#fff',
            }}
            value={command}
            onInput={(e) => {
              setCommand(e.target.value)
            }}
            placeholder={t('launchModel.placeholderTip')}
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
