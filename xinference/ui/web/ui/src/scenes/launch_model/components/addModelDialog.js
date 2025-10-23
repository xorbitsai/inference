import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
} from '@mui/material'
import React, { useContext, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../../components/apiContext'

const API_BASE_URL = 'https://model.xinference.io'

function AddModelDialog({ open, onClose }) {
  const { t } = useTranslation()
  const [modelName, setModelName] = useState('')
  const [loading, setLoading] = useState(false)
  const { endPoint, setErrorMsg } = useContext(ApiContext)

  const searchModelByName = async (name) => {
    try {
      const url = `${API_BASE_URL}/api/models?order=featured&query=${encodeURIComponent(
        name
      )}&page=1&pageSize=5`
      const res = await fetch(url, { method: 'GET' })
      const rawText = await res.text().catch(() => '')
      if (!res.ok) {
        setErrorMsg(rawText || `HTTP ${res.status}`)
        return null
      }
      try {
        const data = JSON.parse(rawText)
        const items = data?.data || []
        const exact = items.find((it) => it?.model_name === name)
        if (!exact) {
          setErrorMsg(t('launchModel.error.name_not_matched'))
          return null
        }
        const id = exact?.id
        const modelType = exact?.model_type
        if (!id || !modelType) {
          setErrorMsg(t('launchModel.error.downloadFailed'))
          return null
        }
        return { id, modelType }
      } catch {
        setErrorMsg(rawText || t('launchModel.error.json_parse_error'))
        return null
      }
    } catch (err) {
      console.error(err)
      setErrorMsg(err.message || t('launchModel.error.requestFailed'))
      return null
    }
  }

  const fetchModelJson = async (modelId) => {
    try {
      const res = await fetch(
        `${API_BASE_URL}/api/models/download?model_id=${encodeURIComponent(
          modelId
        )}`,
        { method: 'GET' }
      )
      const rawText = await res.text().catch(() => '')
      if (!res.ok) {
        setErrorMsg(rawText || `HTTP ${res.status}`)
        return null
      }
      try {
        const data = JSON.parse(rawText)
        return data
      } catch {
        setErrorMsg(rawText || t('launchModel.error.json_parse_error'))
        return null
      }
    } catch (err) {
      console.error(err)
      setErrorMsg(err.message || t('launchModel.error.requestFailed'))
      return null
    }
  }

  const addToLocal = async (modelType, modelJson) => {
    try {
      const res = await fetch(endPoint + '/v1/models/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: modelType, model_json: modelJson}),
      })
      const rawText = await res.text().catch(() => '')
      if (!res.ok) {
        setErrorMsg(rawText || `HTTP ${res.status}`)
        return
      }
      try {
        const data = JSON.parse(rawText)
        console.log('本地 /v1/models/add 响应:', data)
      } catch {
        console.log('本地 /v1/models/add 原始响应:', rawText)
      }
    } catch (error) {
      console.error('Error:', error)
      if (error?.response?.status !== 403) {
        setErrorMsg(error.message)
      }
    }
  }

  const handleFormSubmit = async (e) => {
    e.preventDefault()
    const name = modelName?.trim()
    if (!name) {
      setErrorMsg(t('launchModel.addModelDialog.modelName.tip'))
      return
    }
    setLoading(true)
    setErrorMsg('')
    try {
      const found = await searchModelByName(name)
      if (!found) return
      const { id, modelType } = found

      const modelJson = await fetchModelJson(id)
      if (!modelJson) return

      await addToLocal(modelType, modelJson)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onClose={onClose} width={500}>
      <DialogTitle>{t('launchModel.addModel')}</DialogTitle>
      <DialogContent>
        <div
          style={{
            width: '500px',
            minHeight: '160px',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}
        >
          <div>
            {t('launchModel.addModelDialog.introPrefix')}{' '}
            <a
              href="https://model.xinference.io/models"
              target="_blank"
              rel="noopener noreferrer"
              style={{ textDecoration: 'none', color: '#1976d2' }}
            >
              {t('launchModel.addModelDialog.platformLinkText')}
            </a>
            {t('launchModel.addModelDialog.introSuffix')}
          </div>
          <form onSubmit={handleFormSubmit} id="subscription-form">
            <TextField
              autoFocus
              required
              margin="dense"
              id="modelName"
              name="modelName"
              label={t('launchModel.addModelDialog.modelName')}
              fullWidth
              placeholder={t('launchModel.addModelDialog.placeholder')}
              value={modelName}
              onChange={(e) => {
                setModelName(e.target.value)
              }}
              disabled={loading}
            />
          </form>
        </div>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          {t('launchModel.cancel')}
        </Button>
        <Button autoFocus type="submit" form="subscription-form" disabled={loading}>
          {t('launchModel.confirm')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default AddModelDialog
