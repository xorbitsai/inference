import ExitToAppIcon from '@mui/icons-material/ExitToApp'
import { Box, Stack, Typography } from '@mui/material'
import Button from '@mui/material/Button'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

const Title = ({ title }) => {
  const [, , removeCookie] = useCookies(['token'])
  const navigate = useNavigate()

  const handleLogout = () => {
    removeCookie('token', { path: '/' })
    navigate('/login', { replace: true })
  }

  return (
    <Box mb="30px">
      <Stack direction="row" alignItems="center" justifyContent="space-between">
        <Typography
          variant="h2"
          color="#141414"
          fontWeight="bold"
          sx={{ m: '0 0 5px 0' }}
        >
          {title}
        </Typography>
        <Button
          variant="outlined"
          size="large"
          onClick={handleLogout}
          startIcon={<ExitToAppIcon />}
        >
          LOG OUT
        </Button>
      </Stack>
    </Box>
  )
}

export default Title
