import Snackbar from '@mui/material/Snackbar'
import React, { useContext } from 'react'

import { Alert } from './alertComponent'
import { ApiContext } from './apiContext'

const SuccessMessageSnackBar = () => {
  const { successMsg, setSuccessMsg } = useContext(ApiContext)

  const handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return
    }
    setSuccessMsg('')
  }

  return (
    <Snackbar
      open={successMsg !== ''}
      autoHideDuration={3000}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      onClose={handleClose}
    >
      <Alert severity="success" onClose={handleClose} sx={{ width: '100%' }}>
        {successMsg}
      </Alert>
    </Snackbar>
  )
}

export default SuccessMessageSnackBar
