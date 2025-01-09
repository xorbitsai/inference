import {
  AddBoxOutlined,
  ChevronRightOutlined,
  DnsOutlined,
  GitHub,
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
import { useLocation, useNavigate } from 'react-router-dom'

import icon from '../media/icon.webp'
import ThemeButton from './themeButton'
import TranslateButton from './translateButton'

const MenuSide = () => {
  const theme = useTheme()
  const { pathname } = useLocation()
  const [active, setActive] = useState('')
  const navigate = useNavigate()
  const [drawerWidth, setDrawerWidth] = useState(
    `${Math.min(Math.max(window.innerWidth * 0.2, 287), 320)}px`
  )
  const { t } = useTranslation()

  const navItems = [
    {
      text: 'launch_model',
      label: t('menu.launchModel'),
      icon: <RocketLaunchOutlined />,
    },
    {
      text: 'running_models',
      label: t('menu.runningModels'),
      icon: <SmartToyOutlined />,
    },
    {
      text: 'register_model',
      label: t('menu.registerModel'),
      icon: <AddBoxOutlined />,
    },
    {
      text: 'cluster_information',
      label: t('menu.clusterInfo'),
      icon: <DnsOutlined />,
    },
    {
      text: 'contact_us',
      label: t('menu.contactUs'),
      icon: <GitHub />,
    },
  ]

  useEffect(() => {
    setActive(pathname.substring(1))
  }, [pathname])

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
            {navItems.map(({ text, label, icon }) => {
              if (!icon) {
                return (
                  <Typography key={text} sx={{ m: '2.25rem 0 1rem 3rem' }}>
                    {label}
                  </Typography>
                )
              }
              return (
                <ListItem key={text}>
                  <ListItemButton
                    onClick={() => {
                      if (text === 'contact_us') {
                        window.open(
                          'https://github.com/xorbitsai/inference',
                          '_blank',
                          'noreferrer'
                        )
                      } else if (text === 'launch_model') {
                        sessionStorage.setItem('modelType', '/launch_model/llm')
                        navigate('/launch_model/llm')
                        setActive(text)
                        sessionStorage.setItem('lastActiveUrl', text)
                        console.log(active)
                      } else if (text === 'cluster_information') {
                        navigate('/cluster_info')
                        setActive(text)
                      } else if (text === 'running_models') {
                        navigate('/running_models/LLM')
                        sessionStorage.setItem(
                          'runningModelType',
                          '/running_models/LLM'
                        )
                        setActive(text)
                        sessionStorage.setItem('lastActiveUrl', text)
                        console.log(active)
                      } else if (text === 'register_model') {
                        sessionStorage.setItem(
                          'registerModelType',
                          '/register_model/llm'
                        )
                        navigate('/register_model/llm')
                        setActive(text)
                        sessionStorage.setItem('lastActiveUrl', text)
                        console.log(active)
                      } else {
                        navigate(`/${text}`)
                        setActive(text)
                        console.log(active)
                      }
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        ml: '2rem',
                      }}
                    >
                      {icon}
                    </ListItemIcon>
                    <ListItemText primary={label} />
                    <ChevronRightOutlined sx={{ ml: 'auto' }} />
                  </ListItemButton>
                </ListItem>
              )
            })}
          </List>
        </Box>
      </Box>

      <Box display="flex" alignItems="center" marginLeft={'3rem'}>
        <ThemeButton sx={{ m: '1rem' }} />
        <TranslateButton />
      </Box>
    </Drawer>
  )
}

export default MenuSide
