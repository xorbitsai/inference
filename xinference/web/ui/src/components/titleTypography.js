import Typography from '@mui/material/Typography'
import React from 'react'

const h2Style = {
  margin: '10px 10px',
  fontSize: '20px',
  fontWeight: 'bold',
}

export default function TitleTypography({ value }) {
  return (
    <Typography
      variant="h2"
      gutterBottom
      noWrap
      sx={{ ...h2Style }}
      title={value}
    >
      {value}
    </Typography>
  )
}
