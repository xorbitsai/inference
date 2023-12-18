import { Box } from '@mui/material'
import Button from '@mui/material/Button'
import Container from '@mui/material/Container'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import * as React from 'react'
import { useContext, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import { getEndpoint } from '../../components/utils'

function Login() {
  const [, setCookie] = useCookies(['token'])
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const { setErrorMsg } = useContext(ApiContext)
  const endpoint = getEndpoint()

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
          setCookie('token', data['access_token'])
          navigate('/')
        })
      }
    })
  }

  return (
    <Container component="main" maxWidth="xl">
      <ErrorMessageSnackBar />
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Typography component="h1" variant="h5">
          LOGIN
        </Typography>
        <Box component="main" noValidate sx={{ mt: 1 }}>
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
        </Box>
      </Box>
    </Container>
  )
}

export default Login
