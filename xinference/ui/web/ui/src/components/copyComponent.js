import { Clear, ContentCopy, Done } from '@mui/icons-material'
import { IconButton, Tooltip } from '@mui/material'
import React, { useState } from 'react'

const CopyComponent = ({ tip, text }) => {
  const [copyStatus, setCopyStatus] = useState('pending')

  const handleCopy = (event) => {
    event.stopPropagation()
    const textToCopy = String(text ?? '')

    const showTooltipTemporarily = (status) => {
      setCopyStatus(status)
      setTimeout(() => {
        setCopyStatus('pending')
      }, 1500)
    }

    if (navigator.clipboard && window.isSecureContext) {
      // for HTTPS
      navigator.clipboard
        .writeText(textToCopy)
        .then(() => showTooltipTemporarily('success'))
        .catch(() => showTooltipTemporarily('failed'))
    } else {
      // for HTTP
      const textArea = document.createElement('textarea')
      textArea.value = textToCopy
      textArea.style.position = 'absolute'
      textArea.style.left = '-9999px'
      document.body.appendChild(textArea)
      textArea.select()
      textArea.setSelectionRange(0, textArea.value.length)

      try {
        const success = document.execCommand('copy')
        if (success) {
          showTooltipTemporarily('success')
        } else {
          showTooltipTemporarily('failed')
        }
      } catch (err) {
        showTooltipTemporarily('failed')
      }

      document.body.removeChild(textArea)
    }
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
