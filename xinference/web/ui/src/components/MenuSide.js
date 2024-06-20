import {
  AddBoxOutlined,
  ChevronRightOutlined,
  DnsOutlined,
  GitHub,
  RocketLaunchOutlined,
  SmartToyOutlined,
} from '@mui/icons-material'
import ExitToAppIcon from '@mui/icons-material/ExitToApp'
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
import Button from '@mui/material/Button'
import { useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useLocation, useNavigate } from 'react-router-dom'

import icon from '../media/icon.webp'
import { isValidBearerToken } from './utils'

const navItems = [
  {
    text: 'Launch Model',
    icon: <RocketLaunchOutlined />,
    link: '/launch_model/llm',
    sessionKey: 'modelType',
  },
  {
    text: 'Running Models',
    icon: <SmartToyOutlined />,
    link: '/running_models/LLM',
    sessionKey: 'runningModelType',
  },
  {
    text: 'Register Model',
    icon: <AddBoxOutlined />,
    link: '/register_model',
  },
  {
    text: 'Cluster Information',
    icon: <DnsOutlined />,
    link: '/cluster_info',
  },
  {
    text: 'Contact Us',
    icon: <GitHub />,
    link: 'https://github.com/xorbitsai/inference',
  },
]

const MenuSide = () => {
  const theme = useTheme()
  const { pathname } = useLocation()
  const [active, setActive] = useState('')
  const navigate = useNavigate()
  const [drawerWidth, setDrawerWidth] = useState(
    `${Math.min(Math.max(window.innerWidth * 0.2, 287), 320)}px`
  )

  const [cookie, removeCookie] = useCookies(['token'])
  const handleLogout = () => {
    removeCookie('token', { path: '/' })
    navigate('/login', { replace: true })
  }

  const [selectedIndex, setSelectedIndex] = useState(-1)

  useEffect(() => {
    setActive(pathname.substring(1))
    const pathFirst = pathname.split('/').filter((p) => p)[0]
    const idx = navItems.findIndex(
      (item) => item.link.split('/').filter((p) => p)[0] === pathFirst
    )
    if (idx > -1) {
      setSelectedIndex(idx)
    }
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
      <Box flex="1" display="flex" flexDirection="column">
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

        <Box>
          <Box width="100%">
            <Box m="1.5rem 2rem 2rem 3rem"></Box>
            <List>
              {navItems.map(({ text, icon, link, sessionKey }, index) => {
                if (!icon) {
                  return (
                    <Typography key={text} sx={{ m: '2.25rem 0 1rem 3rem' }}>
                      {text}
                    </Typography>
                  )
                }

                return (
                  <ListItem key={text}>
                    <ListItemButton
                      onClick={() => {
                        if (link.toLowerCase().startsWith('http')) {
                          window.open(link, '_blank', 'noreferrer')
                        } else {
                          navigate(link)
                          setActive(link.substring(1))
                          console.log(active)
                          if (sessionKey) {
                            sessionStorage.setItem(sessionKey, link)
                          }
                        }
                      }}
                      selected={selectedIndex === index}
                    >
                      <ListItemIcon
                        sx={{
                          ml: '2rem',
                        }}
                      >
                        {icon}
                      </ListItemIcon>
                      <ListItemText primary={text} />
                      <ChevronRightOutlined sx={{ ml: 'auto' }} />
                    </ListItemButton>
                  </ListItem>
                )
              })}
            </List>
          </Box>
        </Box>

        <Box flexGrow={1}></Box>
        {isValidBearerToken(cookie.token) && (
          <Button
            variant="outlined"
            size="large"
            onClick={handleLogout}
            startIcon={<ExitToAppIcon />}
            sx={{ m: '1rem', mt: 'auto' }} // 使用marginTop: 'auto'来推动按钮到底部，并添加一些外边距
          >
            LOG OUT
          </Button>
        )}
      </Box>
    </Drawer>
  )
}

export default MenuSide
