import React from 'react'

const Drawer = ({ isOpen, onClose, children }) => {
  return (
    <div className={`drawer ${isOpen ? 'open' : ''}`}>
      <div className="drawer-overlay" onClick={onClose}></div>
      <div className="drawer-content">{isOpen && children}</div>
    </div>
  )
}

export default Drawer
