import { useEffect, useState } from "react";
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
} from "@mui/material";
import {
  ChevronRightOutlined,
  HomeOutlined,
  StorageOutlined,
  DnsOutlined,
  KeyboardVoiceOutlined,
  SmartToyOutlined,
  ChatOutlined,
} from "@mui/icons-material";
import icon from "../media/icon.png";

const navItems = [
  {
    text: "Dashboard",
    icon: <HomeOutlined />,
  },
  {
    text: "My Models",
    icon: null,
  },
  {
    text: "Vicuna-v1.3",
    icon: <ChatOutlined />,
  },
  {
    text: "Vicuna-v1.3 (1)",
    icon: <ChatOutlined />,
  },
  {
    text: "ChatGLM",
    icon: <ChatOutlined />,
  },
  {
    text: "Whisper",
    icon: <KeyboardVoiceOutlined />,
  },
  {
    text: "My Workers",
    icon: null,
  },
  {
    text: "Worker1",
    icon: <SmartToyOutlined />,
  },
  {
    text: "Worker2",
    icon: <SmartToyOutlined />,
  },
  {
    text: "Hardware",
    icon: null,
  },
  {
    text: "CPU1",
    icon: <DnsOutlined />,
  },
  {
    text: "CPU2",
    icon: <DnsOutlined />,
  },
  {
    text: "GPU1",
    icon: <StorageOutlined />,
  },
];

const MenuSide = () => {
  const theme = useTheme();
  const [drawerWidth, setDrawerWidth] = useState("0px");

  useEffect(() => {
    const screenWidth = window.innerWidth;
    const maxDrawerWidth = Math.min(Math.max(screenWidth * 0.25, 250), 320);
    setDrawerWidth(`${maxDrawerWidth}px`);

    // Update the drawer width on window resize
    const handleResize = () => {
      const newScreenWidth = window.innerWidth;
      const newMaxDrawerWidth = Math.min(
        Math.max(newScreenWidth * 0.25, 250),
        320
      );
      setDrawerWidth(`${newMaxDrawerWidth}px`);
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <Drawer
      variant="permanent"
      sx={{
        zIndex: -1,
        width: drawerWidth,
        ...theme.mixins.toolbar,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: drawerWidth,
          boxSizing: "border-box",
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
              sx={{ objectFit: "cover", mr: 1.5 }}
            />
            <Box textAlign="left">
              <Typography
                fontWeight="bold"
                fontSize="1.7rem"
                sx={{ color: theme.palette.secondary[100] }}
              >
                {"Xinference"}
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>

      <Box>
        <Box width="100%">
          <Box m="1.5rem 2rem 2rem 3rem"></Box>
          <List>
            {navItems.map(({ text, icon }) => {
              if (!icon) {
                return (
                  <Typography key={text} sx={{ m: "2.25rem 0 1rem 3rem" }}>
                    {text}
                  </Typography>
                );
              }

              return (
                <ListItem key={text} disablePadding>
                  <ListItemButton
                    onClick={() => {}}
                    sx={{
                      backgroundColor: theme.palette.secondary[300],
                      color: theme.palette.primary[600],
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        ml: "2rem",
                        color: theme.palette.primary[600],
                      }}
                    >
                      {icon}
                    </ListItemIcon>
                    <ListItemText primary={text} />
                    <ChevronRightOutlined sx={{ ml: "auto" }} />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </Box>
      </Box>
    </Drawer>
  );
};

export default MenuSide;
