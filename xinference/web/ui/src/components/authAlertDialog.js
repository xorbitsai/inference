import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogContentText from '@mui/material/DialogContentText'
import DialogTitle from '@mui/material/DialogTitle'
import * as React from 'react'
import { useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

export default function AuthAlertDialog() {
  const navigate = useNavigate()
  const [authStatus, setAuthStatus] = useState('')
  const [, , removeCookie] = useCookies(['token'])

  const handleAuthStatus = () => {
    const status = localStorage.getItem('authStatus')
    if (status) {
      setAuthStatus(status)
    } else {
      setAuthStatus('')
    }
  }

  useEffect(() => {
    localStorage.removeItem('authStatus')
    window.addEventListener('auth-status', handleAuthStatus)

    return () => {
      window.removeEventListener('auth-status', handleAuthStatus)
    }
  }, [])

  const handleClose = () => {
    // trigger first
    const code = localStorage.getItem('authStatus')
    localStorage.removeItem('authStatus')
    setAuthStatus('')
    if (code === '401') {
      removeCookie('token', { path: '/' })
      navigate('/login', { replace: true })
    }
  }

  const handleDialogClose = (event, reason) => {
    if (reason && reason === 'backdropClick') {
      return
    }
    localStorage.removeItem('authStatus')
    setAuthStatus('')
  }

  return (
    <React.Fragment>
      <Dialog
        fullWidth
        maxWidth="md"
        open={authStatus === '401' || authStatus === '403'}
        onClose={handleDialogClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        {authStatus === '403' && (
          <DialogTitle id="alert-dialog-title">
            {'Permission Error'}
          </DialogTitle>
        )}
        {authStatus === '401' && (
          <DialogTitle id="alert-dialog-title">
            {'Authentication Error'}
          </DialogTitle>
        )}
        <DialogContent>
          {authStatus === '403' && (
            <DialogContentText id="alert-dialog-description">
              {'You do not have permissions to do this!'}
            </DialogContentText>
          )}
          {authStatus === '401' && (
            <DialogContentText id="alert-dialog-description">
              {'Invalid credentials! Please login.'}
            </DialogContentText>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>CONFIRMED</Button>
        </DialogActions>
      </Dialog>
    </React.Fragment>
  )
}
