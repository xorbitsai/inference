import Typography from '@mui/material/Typography'
import PropTypes from 'prop-types'
import React from 'react'

export default function TableTitle(props) {
  return (
    <Typography
      component={props.component === undefined ? 'h2' : props.component}
      variant="h6"
      style={{ fontWeight: 'bolder', color: '#781FF5' }}
      gutterBottom
    >
      {props.children}
    </Typography>
  )
}

TableTitle.propTypes = {
  component: PropTypes.elementType,
  children: PropTypes.node,
}
