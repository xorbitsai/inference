import { useTheme } from '@mui/material'
import React, { useEffect } from 'react'

const Drawer = ({ isOpen, onClose, children }) => {
  const theme = useTheme()

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }

    return () => {
      document.body.style.overflow = ''
    }
  }, [isOpen])

  return (
    <div className={`drawer ${isOpen ? 'open' : ''}`}>
      <div className="drawer-overlay" onClick={onClose}></div>
      <div
        className="drawer-content"
        style={
          theme.palette.mode === 'dark' ? { backgroundColor: '#272727' } : {}
        }
      >
        {isOpen && children}
      </div>
    </div>
  )
}

export default Drawer
