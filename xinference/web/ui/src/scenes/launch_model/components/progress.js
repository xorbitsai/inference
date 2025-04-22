import MobileStepper from '@mui/material/MobileStepper'
import React from 'react'

const Progress = ({ progress }) => {
  console.log('progress', progress)
  return (
    <div>
      <MobileStepper
        variant="progress"
        steps={100}
        position="static"
        activeStep={progress * 100}
      />
    </div>
  )
}

export default Progress
