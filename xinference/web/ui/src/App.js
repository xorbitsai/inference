import { CssBaseline, ThemeProvider } from '@mui/material'
import Snackbar from '@mui/material/Snackbar'
import React, { useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { HashRouter, Route, Routes } from 'react-router-dom'

import { Alert } from './components/alertComponent'
import { ApiContextProvider } from './components/apiContext'
import { getEndpoint } from './components/utils'
import Layout from './scenes/_layout'
import LaunchModel from './scenes/launch_model'
import Login from './scenes/login/login'
import RegisterModel from './scenes/register_model'
import RunningModels from './scenes/running_models'
import { useMode } from './theme'

function App() {
  const [theme] = useMode()
  const [cookie, setCookie, removeCookie] = useCookies(['token'])
  const [msg, setMsg] = useState('')

  const endPoint = getEndpoint()

  const removeToken = () => {
    removeCookie('token', { path: '/' })
  }

  useEffect(() => {
    const handleTabPageClose = (e) => {
      removeToken()
      e.returnValue = ''
    }
    window.onbeforeunload = handleTabPageClose

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
          if (data['auth'] === false) {
            if (cookie.token !== 'no_auth') {
              setCookie('token', 'no_auth', { path: '/' })
            }
          } else {
            // TODO: validate bearer token
            if (cookie.token === undefined || cookie.token.length < 10) {
              // not a bearer token, need a bearer token here
              setCookie('token', 'need_auth', { path: '/' })
            }
          }
        })
      }
    })
    // return a function in useEffect means doing something on component unmount
    return () => {
      removeToken()
      window.removeEventListener('beforeunload', handleTabPageClose)
    }
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
          <ApiContextProvider>
            <CssBaseline />
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route element={<Layout />}>
                <Route path="/" element={<LaunchModel />} />
                <Route path="/running_models" element={<RunningModels />} />
                <Route path="/register_model" element={<RegisterModel />} />
              </Route>
            </Routes>
          </ApiContextProvider>
        </ThemeProvider>
      </HashRouter>
    </div>
  )
}

export default App
