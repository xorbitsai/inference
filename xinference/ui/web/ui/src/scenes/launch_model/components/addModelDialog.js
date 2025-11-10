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
import fetchWrapper from '../../../components/fetchWrapper'

const AddModelDialog = ({ open, onClose, onUpdateList }) => {
  const { t } = useTranslation()
  const [modelName, setModelName] = useState('')
  const [loading, setLoading] = useState(false)
  const { setErrorMsg } = useContext(ApiContext)

  const handleFormSubmit = async (e) => {
    e.preventDefault()
    const name = modelName?.trim()
    if (!name) {
      setErrorMsg(t('launchModel.addModelDialog.modelName.tip'))
      return
    }
    setLoading(true)
    setErrorMsg('')

    fetchWrapper
      .post('/v1/models/add', { model_name: modelName })
      .then((data) => {
        onClose(`/launch_model/${data.data.model_type}`)
        onUpdateList(data.data.model_type)
      })
      .catch((error) => {
        console.error('Error:', error)
        if (error.response.status !== 403 && error.response.status !== 401) {
          setErrorMsg(error.message)
        }
      })
      .finally(() => {
        setLoading(false)
      })
  }

  return (
    <Dialog open={open} onClose={() => onClose()} width={500}>
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
        <Button onClick={() => onClose()} disabled={loading}>
          {t('launchModel.cancel')}
        </Button>
        <Button
          autoFocus
          type="submit"
          form="subscription-form"
          disabled={loading}
        >
          {t('launchModel.confirm')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default AddModelDialog
