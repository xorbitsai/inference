import BoltOutlinedIcon from '@mui/icons-material/BoltOutlined'
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline'
import HubOutlinedIcon from '@mui/icons-material/HubOutlined'
import SettingsEthernetOutlinedIcon from '@mui/icons-material/SettingsEthernetOutlined'
import { Box } from '@mui/material'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import * as React from 'react'
import { useState } from 'react'

import { getEndpoint } from '../../components/utils'
import AuthPageLayout from '../login/authPageLayout'

const MIN_PASSWORD_LENGTH = 8

// Real Xinference capabilities (see README.md "Key Features"), framed for
// someone who has just deployed the service and is about to start using it.
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

// LoginAuth (router/index.js) decides whether to render <Setup /> or
// <Login /> based on state that's only computed once per mount. A plain
// react-router navigate() swaps the URL but doesn't remount LoginAuth, so
// it would keep showing this form. Reload the whole app instead so every
// startup check (including /v1/admin/setup/status) re-runs from scratch.
function goToLogin(setupComplete) {
  window.location.hash = '#/login'
  if (setupComplete) {
    sessionStorage.setItem('setupComplete', '1')
  }
  window.location.reload()
}

function Setup() {
  const endpoint = getEndpoint()
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async () => {
    if (!username.trim()) {
      setError('Username is required')
      return
    }
    if (password.length < MIN_PASSWORD_LENGTH) {
      setError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters`)
      return
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    setSubmitting(true)
    setError('')
    try {
      const res = await fetch(endpoint + '/v1/admin/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        if (res.status === 403) {
          // Someone else completed setup first; send this browser to the
          // normal login page instead of leaving it stuck here.
          goToLogin(false)
          return
        }
        setError(errorData.detail || `Setup failed: ${res.status}`)
        return
      }
      goToLogin(true)
    } catch (e) {
      setError(e.message || 'Setup failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthPageLayout
      title="Welcome to Xinference"
      description="This instance has no accounts yet. Create the first administrator account to finish setting up your deployment."
      features={FEATURES}
    >
      <Typography component="h1" variant="h5" sx={{ fontWeight: 600 }}>
        Set up admin account
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        Create the first administrator account to continue.
      </Typography>
      {error && (
        <Alert severity="error" sx={{ mt: 2, width: '100%' }}>
          {error}
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
          onChange={(e) => setUsername(e.target.value)}
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
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          helperText={`At least ${MIN_PASSWORD_LENGTH} characters`}
        />
        <TextField
          margin="normal"
          required
          fullWidth
          name="confirmPassword"
          label="Confirm Password"
          type="password"
          id="confirmPassword"
          autoComplete="new-password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 3, mb: 1 }}
          disabled={submitting}
          onClick={handleSubmit}
        >
          {submitting ? 'Creating...' : 'Create Admin Account'}
        </Button>
        {/* Compact restatement of the pitch for small screens, where the
            feature column is hidden. */}
        <Box
          sx={{
            display: { xs: 'flex', md: 'none' },
            alignItems: 'center',
            gap: 1,
            mt: 2,
            color: 'text.secondary',
          }}
        >
          <CheckCircleOutlineIcon fontSize="small" color="success" />
          <Typography variant="caption">
            This account will have full administrator access.
          </Typography>
        </Box>
      </Box>
    </AuthPageLayout>
  )
}

export default Setup
