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

const AddPair = ({ text, isDelete, onHandleIsDelete, onHandleDelete }) => {
  return (
    <Dialog
      open={isDelete}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title">Warning</DialogTitle>
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
          no
        </Button>
        <Button onClick={onHandleDelete} autoFocus>
          yes
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default AddPair
