import { CssBaseline, ThemeProvider } from '@mui/material'
import Snackbar from '@mui/material/Snackbar'
import React, { useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { HashRouter } from 'react-router-dom'

import { Alert } from './components/alertComponent'
import { ApiContextProvider } from './components/apiContext'
import AuthAlertDialog from './components/authAlertDialog'
import { getEndpoint } from './components/utils'
import WraperRoutes from './router/index'
import { useMode } from './theme'

function App() {
  const [theme] = useMode()
  const [cookie, setCookie, removeCookie] = useCookies(['token'])
  const [msg, setMsg] = useState('')
  const [authority, setAuthority] = useState(true)

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
          setAuthority(data.auth)
          if (!data.auth && cookie.token !== 'no_auth') {
            setCookie('token', 'no_auth', { path: '/' })
          } else if (data.auth && !sessionStorage.getItem('token')) {
            removeCookie('token', { path: '/' })
          }
        })
      }
    })
  }, [])

  const handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return
    }
    setMsg('')
  }

  return (
    <div className="app">
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
        <ThemeProvider theme={theme}>
          <ApiContextProvider authority={authority}>
            <CssBaseline />
            <AuthAlertDialog />
            <WraperRoutes />
          </ApiContextProvider>
        </ThemeProvider>
      </HashRouter>
    </div>
  )
}

export default App
