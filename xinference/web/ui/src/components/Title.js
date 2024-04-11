import { Box, Stack, Typography } from '@mui/material'

const Title = ({ title }) => {
  return (
    <Box mb="30px">
      <Stack direction="row" alignItems="center" justifyContent="space-between">
        <Typography
          variant="h2"
          color="text.primary"
          fontWeight="bold"
          sx={{ m: '0 0 5px 0' }}
        >
          {title}
        </Typography>
      </Stack>
    </Box>
  )
}

export default Title
