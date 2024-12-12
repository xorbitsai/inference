import ExitToAppIcon from '@mui/icons-material/ExitToApp'
import { Box, Stack, Typography } from '@mui/material'
import Button from '@mui/material/Button'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import { isValidBearerToken } from './utils'

const Title = ({ title }) => {
  const [cookie, , removeCookie] = useCookies(['token'])
  const navigate = useNavigate()

  const handleLogout = () => {
    removeCookie('token', { path: '/' })
    sessionStorage.removeItem('token')
    sessionStorage.removeItem('auth')
    sessionStorage.removeItem('modelType')
    sessionStorage.removeItem('lastActiveUrl')
    sessionStorage.removeItem('runningModelType')
    sessionStorage.removeItem('registerModelType')
    navigate('/login', { replace: true })
  }

  return (
    <Box mb="30px">
      <Stack direction="row" alignItems="center" justifyContent="space-between">
        <Typography variant="h2" fontWeight="bold" sx={{ m: '0 0 5px 0' }}>
          {title}
        </Typography>
        {(isValidBearerToken(cookie.token) ||
          isValidBearerToken(sessionStorage.getItem('token'))) && (
          <Button
            variant="outlined"
            size="large"
            onClick={handleLogout}
            startIcon={<ExitToAppIcon />}
          >
            LOG OUT
          </Button>
        )}
      </Stack>
    </Box>
  )
}

export default Title
