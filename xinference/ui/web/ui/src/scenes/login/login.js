import BoltOutlinedIcon from '@mui/icons-material/BoltOutlined'
import HubOutlinedIcon from '@mui/icons-material/HubOutlined'
import SettingsEthernetOutlinedIcon from '@mui/icons-material/SettingsEthernetOutlined'
import { Box, Divider } from '@mui/material'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import * as React from 'react'
import { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import { getEndpoint } from '../../components/utils'
import AuthPageLayout from './authPageLayout'

// Same real Xinference capabilities shown on the setup page (see
// scenes/setup/setup.js), so returning users see a consistent pitch.
const FEATURES = [
  {
    icon: BoltOutlinedIcon,
    title: 'Model serving made easy',
    description:
      'Launch state-of-the-art LLM, embedding, and multimodal models with a single command.',
  },
  {
    icon: SettingsEthernetOutlinedIcon,
    title: 'OpenAI-compatible API',
    description:
      'Drop-in RESTful API, RPC, CLI, and Web UI access -- works with your existing tooling.',
  },
  {
    icon: HubOutlinedIcon,
    title: 'Distributed by design',
    description:
      'Scale inference across GPUs and CPUs, on a single machine or a cluster.',
  },
]

function Login() {
  const { t } = useTranslation()
  const [, setCookie] = useCookies(['token'])
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [oidcEnabled, setOidcEnabled] = useState(false)
  const { setErrorMsg } = useContext(ApiContext)
  const endpoint = getEndpoint()
  // Setup (scenes/setup/setup.js) reloads the whole app after creating the
  // admin account, so it can't pass navigate() state -- it leaves a flag in
  // sessionStorage instead. Read it once and clear it so it doesn't stick
  // around across future visits to this page.
  const [setupComplete] = useState(() => {
    const flag = sessionStorage.getItem('setupComplete') === '1'
    sessionStorage.removeItem('setupComplete')
    return flag
  })

  useEffect(() => {
    fetch(endpoint + '/v1/cluster/ui_config', {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    }).then((res) => {
      if (res.ok) {
        res.json().then((data) => {
          setOidcEnabled(data.oidc_enabled || false)
        })
      }
    })
  }, [])

  const handleSubmit = () => {
    fetch(endpoint + '/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: username,
        password: password,
      }),
    }).then((res) => {
      if (!res.ok) {
        res.json().then((errorData) => {
          setErrorMsg(
            `Login failed: ${res.status} - ${
              errorData.detail || 'Unknown error'
            }`
          )
        })
      } else {
        res.json().then((data) => {
          setCookie('token', data['access_token'], { path: '/' })
          sessionStorage.setItem('token', data['access_token'])
          if (data['refresh_token']) {
            sessionStorage.setItem('refresh_token', data['refresh_token'])
          }
          if (data['must_change_password']) {
            navigate('/change_password')
          } else {
            navigate('/launch_model/llm')
          }
        })
      }
    })
  }

  return (
    <AuthPageLayout
      title="Welcome back"
      description="Sign in to manage your models, monitor your cluster, and configure access."
      features={FEATURES}
    >
      <ErrorMessageSnackBar />
      <Typography component="h1" variant="h5" sx={{ fontWeight: 600 }}>
        Login
      </Typography>
      {setupComplete && (
        <Alert severity="success" sx={{ mt: 2, width: '100%' }}>
          Admin account created. Sign in with your new credentials.
        </Alert>
      )}
      <Box component="main" noValidate sx={{ mt: 2, width: '100%' }}>
        <TextField
          margin="normal"
          required
          fullWidth
          id="username"
          label="Username"
          name="username"
          value={username}
          onChange={(e) => {
            setUsername(e.target.value)
          }}
          autoFocus
        />
        <TextField
          margin="normal"
          required
          fullWidth
          name="password"
          label="Password"
          type="password"
          id="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => {
            setPassword(e.target.value)
          }}
        />
        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 3, mb: 2 }}
          onClick={handleSubmit}
        >
          Sign In
        </Button>
        {oidcEnabled && (
          <>
            <Divider sx={{ my: 2 }}>{t('login.or')}</Divider>
            <Button
              fullWidth
              variant="outlined"
              sx={{ mb: 2 }}
              onClick={() => {
                window.location.href = endpoint + '/api/oidc/authorize'
              }}
            >
              {t('login.sso')}
            </Button>
          </>
        )}
      </Box>
    </AuthPageLayout>
  )
}

export default Login
