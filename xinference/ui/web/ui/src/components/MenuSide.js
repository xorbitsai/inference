import {
  AddBoxOutlined,
  ChevronRightOutlined,
  DescriptionOutlined,
  DnsOutlined,
  GitHub,
  Language,
  OpenInNew,
  RocketLaunchOutlined,
  SmartToyOutlined,
} from '@mui/icons-material'
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  useTheme,
} from '@mui/material'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import icon from '../media/icon.webp'
import ThemeButton from './themeButton'
import TranslateButton from './translateButton'
import VersionLabel from './versionLabel'

const MenuSide = () => {
  const theme = useTheme()
  const navigate = useNavigate()
  const [drawerWidth, setDrawerWidth] = useState(
    `${Math.min(Math.max(window.innerWidth * 0.2, 287), 320)}px`
  )
  const { i18n, t } = useTranslation()

  const navItems = [
    {
      text: 'launch_model',
      label: t('menu.launchModel'),
      icon: <RocketLaunchOutlined />,
      action: 'navigate',
      path: '/launch_model/llm',
      session: {
        modelType: '/launch_model/llm',
        lastActiveUrl: 'launch_model',
      },
    },
    {
      text: 'running_models',
      label: t('menu.runningModels'),
      icon: <SmartToyOutlined />,
      action: 'navigate',
      path: '/running_models/LLM',
      session: {
        runningModelType: '/running_models/LLM',
        lastActiveUrl: 'running_models',
      },
    },
    {
      text: 'register_model',
      label: t('menu.registerModel'),
      icon: <AddBoxOutlined />,
      action: 'navigate',
      path: '/register_model/llm',
      session: {
        registerModelType: '/register_model/llm',
        lastActiveUrl: 'register_model',
      },
    },
    {
      text: 'cluster_information',
      label: t('menu.clusterInfo'),
      icon: <DnsOutlined />,
      action: 'navigate',
      path: '/cluster_info',
    },
    {
      text: 'documentation',
      label: t('menu.documentation'),
      icon: <DescriptionOutlined />,
      action: 'external',
      url:
        'https://inference.readthedocs.io/' +
        (i18n.language === 'zh' ? 'zh-cn' : ''),
    },
    {
      text: 'contact_us',
      label: t('menu.contactUs'),
      icon: <GitHub />,
      action: 'external',
      url: 'https://github.com/xorbitsai/inference',
    },
    {
      text: 'website',
      label: t('menu.website'),
      icon: <Language />,
      action: 'external',
      url:
        i18n.language === 'zh'
          ? 'https://xinference.cn'
          : 'https://xinference.io',
    },
  ]

  useEffect(() => {
    const screenWidth = window.innerWidth
    const maxDrawerWidth = Math.min(Math.max(screenWidth * 0.2, 287), 320)
    setDrawerWidth(`${maxDrawerWidth}px`)

    // Update the drawer width on window resize
    const handleResize = () => {
      const newScreenWidth = window.innerWidth
      const newMaxDrawerWidth = Math.min(
        Math.max(newScreenWidth * 0.2, 287),
        320
      )
      setDrawerWidth(`${newMaxDrawerWidth}px`)
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  const handleNavClick = (item) => {
    const { action, url, path, text, session } = item
    if (action === 'external' && url) {
      window.open(url, '_blank', 'noreferrer')
      return
    }

    if (action === 'navigate') {
      if (session) {
        Object.entries(session).forEach(([key, value]) => {
          sessionStorage.setItem(key, value)
        })
      }
      navigate(path ?? `/${text}`)
      return
    }

    // default behavior
    navigate(`/${text}`)
  }

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        ...theme.mixins.toolbar,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
      style={{ zIndex: 1 }}
    >
      {/* Title */}
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        width="100%"
      >
        <Box display="flex" m="2rem 1rem 0rem 1rem" width="217px">
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            textTransform="none"
          >
            <Box
              component="img"
              alt="profile"
              src={icon}
              height="60px"
              width="60px"
              borderRadius="50%"
              sx={{ objectFit: 'cover', mr: 1.5 }}
            />
            <Box textAlign="left">
              <Typography fontWeight="bold" fontSize="1.7rem">
                {'Xinference'}
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>

      <Box sx={{ flexGrow: 1 }}>
        <Box width="100%">
          <Box m="1.5rem 2rem 2rem 3rem"></Box>
          <List>
            {navItems.map((item) => {
              const { text, label, icon, action } = item
              if (!icon) {
                return (
                  <Typography key={text} sx={{ m: '2.25rem 0 1rem 3rem' }}>
                    {label}
                  </Typography>
                )
              }
              return (
                <ListItem key={text}>
                  <ListItemButton onClick={() => handleNavClick(item)}>
                    <ListItemIcon
                      sx={{
                        ml: '2rem',
                      }}
                    >
                      {icon}
                    </ListItemIcon>
                    <ListItemText primary={label} />
                    {action === 'external' ? (
                      <OpenInNew
                        sx={{ ml: 'auto', mr: '0.5rem', fontSize: 'small' }}
                      />
                    ) : (
                      <ChevronRightOutlined sx={{ ml: 'auto' }} />
                    )}
                  </ListItemButton>
                </ListItem>
              )
            })}
          </List>
        </Box>
      </Box>

      <Box display="flex" alignItems="center" marginX={'3rem'}>
        <ThemeButton sx={{ m: '1rem' }} />
        <TranslateButton />
        <VersionLabel sx={{ ml: 'auto' }} />
      </Box>
    </Drawer>
  )
}

export default MenuSide
