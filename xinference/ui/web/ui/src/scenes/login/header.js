import { AppBar, Box, Toolbar } from '@mui/material'
import Typography from '@mui/material/Typography'

import icon from '../../media/icon.png'

export default function Header() {
  return (
    <AppBar
      elevation={0}
      color="transparent"
      sx={{
        backdropFilter: 'blur(20px)',
        borderBottom: 1,
        borderColor: 'grey.300',
        zIndex: (theme) => theme.zIndex.drawer + 1,
      }}
    >
      <Toolbar sx={{ justifyContent: 'start' }}>
        <Box
          component="img"
          alt="profile"
          src={icon}
          height="60px"
          width="60px"
          borderRadius="50%"
          sx={{ objectFit: 'cover', mr: 1.5 }}
        />
        <Box textAlign="left">
          <Typography fontWeight="bold" fontSize="1.7rem">
            {'Xinference'}
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  )
}
