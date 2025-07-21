import Translate from '@mui/icons-material/Translate'
import {
  Box,
  ClickAwayListener,
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
  const [open, setOpen] = React.useState(false)
  const { i18n } = useTranslation()
  const languages = [
    {
      language: 'English',
      code: 'en',
    },
    {
      language: '中文',
      code: 'zh',
    },
    {
      language: '日本語',
      code: 'ja',
    },
    {
      language: '한국어',
      code: 'ko',
    },
  ]

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng)
    handleTooltipClose()
  }

  const handleTooltipClose = () => {
    setOpen(false)
  }

  return (
    <ClickAwayListener onClickAway={handleTooltipClose}>
      <Tooltip
        onClose={handleTooltipClose}
        open={open}
        disableFocusListener
        disableHoverListener
        disableTouchListener
        placement="top"
        slotProps={{
          popper: {
            disablePortal: true,
          },
        }}
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
      >
        <Box sx={sx}>
          <IconButton size="large" onClick={() => setOpen((prev) => !prev)}>
            <Translate />
          </IconButton>
        </Box>
      </Tooltip>
    </ClickAwayListener>
  )
}

export default TranslateButton
