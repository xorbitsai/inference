import './style.css'

import FilterNoneIcon from '@mui/icons-material/FilterNone'
import { Alert, Snackbar, Tooltip } from '@mui/material'
import ClipboardJS from 'clipboard'
import React, { useState } from 'react'

const CopyComponent = ({ tip, text }) => {
  const [isCopySuccess, setIsCopySuccess] = useState(false)

  const handleCopy = () => {
    const clipboard = new ClipboardJS('.copyText', {
      text: () => text,
    })

    clipboard.on('success', (e) => {
      e.clearSelection()
      setIsCopySuccess(true)
    })
  }

  return (
    <>
      <Tooltip title={tip} placement="top">
        <FilterNoneIcon className="copyText" onClick={handleCopy} />
      </Tooltip>
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={isCopySuccess}
        autoHideDuration={1500}
        onClose={() => setIsCopySuccess(false)}
      >
        <Alert severity="success" variant="filled" sx={{ width: '100%' }}>
          Copied to clipboard!
        </Alert>
      </Snackbar>
    </>
  )
}

export default CopyComponent
