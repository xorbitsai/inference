import Snackbar from '@mui/material/Snackbar'
import React, { useContext } from 'react'

import { Alert } from './alertComponent'
import { ApiContext } from './apiContext'

const ErrorMessageSnackBar = () => {
  const { errorMsg, setErrorMsg } = useContext(ApiContext)

  const handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return
    }
    setErrorMsg('')
  }

  return (
    <Snackbar
      open={errorMsg !== ''}
      autoHideDuration={10000}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      onClose={handleClose}
    >
      <Alert severity="error" onClose={handleClose} sx={{ width: '100%' }}>
        {errorMsg}
      </Alert>
    </Snackbar>
  )
}

export default ErrorMessageSnackBar
