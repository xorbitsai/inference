import { WarningAmber } from '@mui/icons-material'
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from '@mui/material'
import React from 'react'
import { useTranslation } from 'react-i18next'

const DeleteDialog = ({ text, isDelete, onHandleIsDelete, onHandleDelete }) => {
  const { t } = useTranslation()
  return (
    <Dialog
      open={isDelete}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title">
        {t('components.warning')}
      </DialogTitle>
      <DialogContent>
        <DialogContentText
          className="deleteDialog"
          id="alert-dialog-description"
        >
          <WarningAmber className="warningIcon" />
          <p>{text}</p>
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button
          onClick={() => {
            onHandleIsDelete()
          }}
        >
          {t('components.cancel')}
        </Button>
        <Button onClick={onHandleDelete} autoFocus>
          {t('components.ok')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default DeleteDialog
