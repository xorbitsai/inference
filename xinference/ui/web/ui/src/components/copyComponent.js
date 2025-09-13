import { Clear, ContentCopy, Done } from '@mui/icons-material'
import { IconButton, Tooltip } from '@mui/material'
import React, { useState } from 'react'

import { copyToClipboard } from './utils'

const CopyComponent = ({ tip, text }) => {
  const [copyStatus, setCopyStatus] = useState('pending')

  const showTooltipTemporarily = (status) => {
    setCopyStatus(status)
    setTimeout(() => setCopyStatus('pending'), 1500)
  }

  const handleCopy = async (event) => {
    event.stopPropagation()
    const textToCopy = String(text ?? '')
    const success = await copyToClipboard(textToCopy)
    showTooltipTemporarily(success ? 'success' : 'failed')
  }

  return (
    <>
      {copyStatus === 'pending' ? (
        <Tooltip title={tip} placement="top">
          <IconButton aria-label="copy" onClick={handleCopy}>
            <ContentCopy fontSize="small" />
          </IconButton>
        </Tooltip>
      ) : copyStatus === 'success' ? (
        <IconButton aria-label="copy">
          <Done fontSize="small" color="success" />
        </IconButton>
      ) : (
        <IconButton aria-label="copy">
          <Clear fontSize="small" color="error" />
        </IconButton>
      )}
    </>
  )
}

export default CopyComponent
