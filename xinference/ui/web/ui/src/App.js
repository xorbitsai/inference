import { CssBaseline } from '@mui/material'
import Snackbar from '@mui/material/Snackbar'
import React, { useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { HashRouter } from 'react-router-dom'

import { Alert } from './components/alertComponent'
import { ApiContextProvider } from './components/apiContext'
import AuthAlertDialog from './components/authAlertDialog'
import { ThemeProvider } from './components/themeContext'
import { getEndpoint } from './components/utils'
import WraperRoutes from './router/index'

function App() {
  const [cookie, setCookie, removeCookie] = useCookies(['token'])
  const [msg, setMsg] = useState('')

  const endPoint = getEndpoint()

  useEffect(() => {
    // token possible value: no_auth / need_auth / <real bearer token>
    fetch(endPoint + '/v1/cluster/auth', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }).then((res) => {
      if (!res.ok) {
        res.json().then((errorData) => {
          setMsg(
            `Server error: ${res.status} - ${
              errorData.detail || 'Unknown error'
            }`
          )
        })
      } else {
        res.json().then((data) => {
          if (!data.auth) {
            setCookie('token', 'no_auth', { path: '/' })
            sessionStorage.setItem('token', 'no_auth')
          } else if (
            data.auth &&
            sessionStorage.getItem('token') === 'no_auth'
          ) {
            removeCookie('token', { path: '/' })
            sessionStorage.removeItem('token')
          }
          sessionStorage.setItem('auth', String(data.auth)) // sessionStorage only can set string value
        })
      }
    })
  }, [cookie])

  const handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return
    }
    setMsg('')
  }

  return (
    <div className="app">
      <ThemeProvider>
        <Snackbar
          open={msg !== ''}
          autoHideDuration={10000}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          onClose={handleClose}
        >
          <Alert severity="error" onClose={handleClose} sx={{ width: '100%' }}>
            {msg}
          </Alert>
        </Snackbar>
        <HashRouter>
          <ApiContextProvider>
            <CssBaseline />
            <AuthAlertDialog />
            <WraperRoutes />
          </ApiContextProvider>
        </HashRouter>
      </ThemeProvider>
    </div>
  )
}

export default App
