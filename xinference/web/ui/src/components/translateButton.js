import Translate from '@mui/icons-material/Translate'
import {
  Box,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Tooltip,
} from '@mui/material'
import React from 'react'
import { useTranslation } from 'react-i18next'

const TranslateButton = ({ sx }) => {
  const { i18n } = useTranslation()
  const languages = [
    {
      language: '中文',
      code: 'zh',
    },
    {
      language: 'English',
      code: 'en',
    },
  ]

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng)
  }

  return (
    <Tooltip
      title={
        <List sx={{ pt: 0 }}>
          {languages.map((item, index) => {
            return (
              <ListItem
                key={index}
                disablePadding
                onClick={() => changeLanguage(item.code)}
              >
                <ListItemButton
                  sx={{
                    '&': {
                      paddingY: 0,
                      marginY: '2px',
                    },
                    '&:hover, &:focus': {
                      bgcolor: '#bbb',
                      color: '#333',
                    },
                  }}
                >
                  <ListItemText primary={item.language} />
                </ListItemButton>
              </ListItem>
            )
          })}
        </List>
      }
      placement="top"
      disableFocusListener
      disableTouchListener
    >
      <Box sx={sx}>
        <IconButton size="large">
          <Translate />
        </IconButton>
      </Box>
    </Tooltip>
  )
}

export default TranslateButton
