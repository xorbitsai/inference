import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from '@mui/material'
import * as React from 'react'
import { useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

export default function AuthAlertDialog() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [authStatus, setAuthStatus] = useState('')
  const [, , removeCookie] = useCookies(['token'])

  const handleAuthStatus = () => {
    const status = localStorage.getItem('authStatus')
    if (status === '401') {
      removeCookie('token', { path: '/' })
      localStorage.removeItem('authStatus')
      sessionStorage.removeItem('token')
      navigate('/login', { replace: true })
      return
    }
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
      sessionStorage.removeItem('token')
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
            {t('auth.permissionError')}
          </DialogTitle>
        )}
        {authStatus === '401' && (
          <DialogTitle id="alert-dialog-title">
            {t('auth.authenticationError')}
          </DialogTitle>
        )}
        <DialogContent>
          {authStatus === '403' && (
            <DialogContentText id="alert-dialog-description">
              {t('auth.noPermission')}
            </DialogContentText>
          )}
          {authStatus === '401' && (
            <DialogContentText id="alert-dialog-description">
              {t('auth.invalidCredentials')}
            </DialogContentText>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>{t('auth.confirmed')}</Button>
        </DialogActions>
      </Dialog>
    </React.Fragment>
  )
}
