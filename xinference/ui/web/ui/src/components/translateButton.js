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
  const languages = React.useMemo(
    () => [
      { language: 'English', code: 'en', flag: '🇺🇸' },
      { language: '中文', code: 'zh', flag: '🇨🇳' },
      { language: '日本語', code: 'ja', flag: '🇯🇵' },
      { language: '한국어', code: 'ko', flag: '🇰🇷' },
    ],
    []
  )

  const normalizeLang = (lng) =>
    String(lng || '')
      .toLowerCase()
      .split('-')[0]
  const currentLang = normalizeLang(i18n.language)

  const changeLanguage = (lng) => {
    if (normalizeLang(i18n.language) === lng) {
      handleTooltipClose()
      return
    }
    i18n.changeLanguage(lng)
    handleTooltipClose()
  }

  const handleTooltipClose = () => {
    setOpen(false)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      handleTooltipClose()
    }
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
                    selected={currentLang === item.code}
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
                    <ListItemText primary={item.flag + ' ' + item.language} />
                  </ListItemButton>
                </ListItem>
              )
            })}
          </List>
        }
      >
        <Box sx={sx} onKeyDown={handleKeyDown}>
          <IconButton
            size="large"
            aria-label="Change language"
            onClick={() => setOpen((prev) => !prev)}
          >
            <Translate />
          </IconButton>
        </Box>
      </Tooltip>
    </ClickAwayListener>
  )
}

export default TranslateButton
