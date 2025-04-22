import LinearProgress from '@mui/material/LinearProgress'
import React from 'react'

const Progress = ({ progress }) => {
  return (
    <div style={{ marginBottom: 10 }}>
      <LinearProgress variant="determinate" value={progress * 100} />
    </div>
  )
}

export default Progress
