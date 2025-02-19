import { useTheme } from '@mui/material'
import React from 'react'

const Drawer = ({ isOpen, onClose, children }) => {
  const theme = useTheme()

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
