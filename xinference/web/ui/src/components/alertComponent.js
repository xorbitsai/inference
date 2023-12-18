import MuiAlert from '@mui/material/Alert'
import React from 'react'

const Alert = React.forwardRef(function Alert(props, ref) {
  return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />
})

export { Alert }
