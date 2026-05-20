import { Box } from '@mui/material'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Container from '@mui/material/Container'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import * as React from 'react'
import { Fragment, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import fetchWrapper from '../../components/fetchWrapper'

function ChangePassword() {
  const navigate = useNavigate()
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const handleSubmit = async () => {
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match')
      return
    }
    if (newPassword.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }
    try {
      // Get user_id from token payload
      const token = sessionStorage.getItem('token')
      const payload = JSON.parse(atob(token.split('.')[1]))
      await fetchWrapper.put(`/v1/admin/users/${payload.user_id}/password`, {
        new_password: newPassword,
      })
      setSuccess(true)
      setTimeout(() => navigate('/launch_model/llm'), 1500)
    } catch (e) {
      setError(e.message || 'Failed to change password')
    }
  }

  return (
    <Fragment>
      <Container component="main" maxWidth="sm" sx={{ marginTop: 10 }}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Typography component="h1" variant="h5">
            Change Password
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            You must change your password before continuing.
          </Typography>
          {error && (
            <Alert severity="error" sx={{ mt: 2, width: '100%' }}>
              {error}
            </Alert>
          )}
          {success && (
            <Alert severity="success" sx={{ mt: 2, width: '100%' }}>
              Password changed successfully. Redirecting...
            </Alert>
          )}
          <Box sx={{ mt: 2, width: '100%' }}>
            <TextField
              margin="normal"
              required
              fullWidth
              label="New Password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              label="Confirm Password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
            <Button
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              onClick={handleSubmit}
            >
              Change Password
            </Button>
          </Box>
        </Box>
      </Container>
    </Fragment>
  )
}

export default ChangePassword
