import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
} from '@mui/material'
import React from 'react'
import { useTranslation } from 'react-i18next'

import CopyComponent from '../../../components/copyComponent'

const EditCustomModel = ({
  open,
  modelData,
  onClose,
  handleJsonDataPresentation,
}) => {
  const { t } = useTranslation()

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle sx={{ m: 0, p: 2 }} id="customized-dialog-title">
        {modelData.model_name}
      </DialogTitle>
      <Box
        sx={(theme) => ({
          position: 'absolute',
          right: 8,
          top: 8,
          color: theme.palette.grey[500],
        })}
      >
        <CopyComponent
          tip={t('launchModel.copyJson')}
          text={JSON.stringify(modelData, null, 4)}
        />
      </Box>
      <DialogContent>
        <TextField
          sx={{ width: '700px' }}
          multiline
          rows={24}
          disabled
          defaultValue={JSON.stringify(modelData, null, 4)}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t('launchModel.cancel')}</Button>
        <Button onClick={handleJsonDataPresentation} autoFocus>
          {t('launchModel.edit')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default EditCustomModel
