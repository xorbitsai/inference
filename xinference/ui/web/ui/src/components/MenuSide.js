import {
  AddBoxOutlined,
  ArticleOutlined,
  ChevronLeftOutlined,
  ChevronRightOutlined,
  DescriptionOutlined,
  DnsOutlined,
  GitHub,
  Language,
  MonitorHeartOutlined,
  OpenInNew,
  Psychology,
  RocketLaunchOutlined,
  SmartToyOutlined,
} from '@mui/icons-material'
import {
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Typography,
} from '@mui/material'
import { useContext, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import icon from '../media/icon.webp'
import { ApiContext } from './apiContext'
import ThemeButton from './themeButton'
import TranslateButton from './translateButton'
import VersionLabel from './versionLabel'

const COLLAPSED_KEY = 'xinference_sidebar_collapsed'
const COLLAPSED_WIDTH = 64

function readCollapsed() {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === 'true'
  } catch {
    return false
  }
}

function writeCollapsed(val) {
  try {
    localStorage.setItem(COLLAPSED_KEY, String(val))
  } catch {
    // ignore
  }
}

const MenuSide = () => {
  const navigate = useNavigate()
  const [collapsed, setCollapsed] = useState(readCollapsed)
  const [drawerWidth, setDrawerWidth] = useState(
    `${Math.min(Math.max(window.innerWidth * 0.2, 287), 320)}px`
  )
  const { i18n, t } = useTranslation()
  const { endPoint } = useContext(ApiContext)
  const [esEnabled, setEsEnabled] = useState(false)

  const currentWidth = collapsed ? `${COLLAPSED_WIDTH}px` : drawerWidth

  useEffect(() => {
    fetch(endPoint + '/v1/cluster/ui_config')
      .then((res) => {
        if (!res.ok) throw new Error('Network response was not ok')
        return res.json()
      })
      .then((data) => setEsEnabled(data.es_enabled || false))
      .catch(() => setEsEnabled(false))
  }, [endPoint])

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
      text: 'monitoring',
      label: t('menu.monitoring'),
      icon: <MonitorHeartOutlined />,
      action: 'navigate',
      path: '/monitoring',
    },
    ...(esEnabled
      ? [
          {
            text: 'logs',
            label: t('menu.logs'),
            icon: <ArticleOutlined />,
            action: 'navigate',
            path: '/logs',
          },
        ]
      : []),
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
    {
      text: 'xagent',
      label: t('menu.xagent'),
      icon: (
        <Psychology sx={{ fontSize: 26, marginLeft: -0.5, color: '#1e88e5' }} />
      ),
      action: 'external',
      url: 'https://github.com/xorbitsai/xagent',
    },
  ]

  useEffect(() => {
    const handleResize = () => {
      const screenWidth = window.innerWidth
      const maxDrawerWidth = Math.min(Math.max(screenWidth * 0.2, 287), 320)
      setDrawerWidth(`${maxDrawerWidth}px`)
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  const toggleCollapsed = () => {
    const next = !collapsed
    setCollapsed(next)
    writeCollapsed(next)
  }

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

    navigate(`/${text}`)
  }

  const transition = 'width 0.2s ease'

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: currentWidth,
        transition,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: currentWidth,
          transition,
          boxSizing: 'border-box',
          overflowX: 'hidden',
          overflowY: 'auto',
        },
      }}
      style={{ zIndex: 1 }}
    >
      {/* Logo */}
      <Box
        display="flex"
        alignItems="center"
        justifyContent={collapsed ? 'center' : 'flex-start'}
        gap={collapsed ? 0 : '1rem'}
        margin={collapsed ? '2rem 0' : '2rem 0rem 2rem 2rem'}
        sx={{ transition: 'all 0.2s ease' }}
      >
        <Box
          component="img"
          alt="profile"
          src={icon}
          width={collapsed ? '36px' : '60px'}
          borderRadius="50%"
          sx={{ transition: 'width 0.2s ease' }}
        />
        {!collapsed && (
          <Box textAlign="left">
            <Typography fontWeight="bold" fontSize="1.7rem">
              {'Xinference'}
            </Typography>
          </Box>
        )}
      </Box>

      {/* Nav items */}
      <Box sx={{ flexGrow: 1 }}>
        <List>
          {navItems.map((item) => {
            const { text, label, icon: navIcon, action } = item
            return (
              <ListItem key={text} sx={{ px: collapsed ? 0.5 : 1 }}>
                {collapsed ? (
                  <Tooltip title={label} placement="right" arrow>
                    <ListItemButton
                      sx={{ justifyContent: 'center', px: 1, minHeight: 48 }}
                      onClick={() => handleNavClick(item)}
                    >
                      <ListItemIcon
                        sx={{ minWidth: 0, justifyContent: 'center' }}
                      >
                        {navIcon}
                      </ListItemIcon>
                    </ListItemButton>
                  </Tooltip>
                ) : (
                  <ListItemButton
                    sx={{ pl: '1.5rem' }}
                    onClick={() => handleNavClick(item)}
                  >
                    <ListItemIcon sx={{ minWidth: '2rem' }}>
                      {navIcon}
                    </ListItemIcon>
                    <ListItemText primary={label} />
                    {action === 'external' ? (
                      <OpenInNew sx={{ fontSize: 'small' }} />
                    ) : (
                      <ChevronRightOutlined />
                    )}
                  </ListItemButton>
                )}
              </ListItem>
            )
          })}
        </List>
      </Box>

      {/* Bottom toolbar — collapses to zero height when sidebar collapsed */}
      <Box
        display="flex"
        alignItems="center"
        sx={{
          mx: collapsed ? 0 : '3rem',
          maxHeight: collapsed ? 0 : '100px',
          overflow: 'hidden',
          transition: 'max-height 0.2s ease, margin 0.2s ease',
        }}
      >
        <ThemeButton sx={{ m: '1rem' }} />
        <TranslateButton />
        <VersionLabel sx={{ ml: 'auto' }} />
      </Box>

      {/* Collapse toggle button */}
      <Box
        display="flex"
        justifyContent={collapsed ? 'center' : 'flex-end'}
        px={1}
        pb={1}
      >
        <Tooltip
          title={collapsed ? t('menu.expand') : t('menu.collapse')}
          placement="right"
        >
          <IconButton size="small" onClick={toggleCollapsed}>
            {collapsed ? <ChevronRightOutlined /> : <ChevronLeftOutlined />}
          </IconButton>
        </Tooltip>
      </Box>
    </Drawer>
  )
}

export default MenuSide
