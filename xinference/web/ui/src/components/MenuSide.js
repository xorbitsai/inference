import { useEffect, useState, useContext } from "react";
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Select,
  MenuItem,
  Typography,
  useTheme,
} from "@mui/material";
import {
  ChevronRightOutlined,
  RocketLaunchOutlined,
  SmartToyOutlined,
  TranslateOutlined,
  AddBoxOutlined,
  GitHub,
} from "@mui/icons-material";
import icon from "../media/icon.webp";
import { useLocation, useNavigate } from "react-router-dom";
import { LanguageContext } from "../theme";

const MenuSide = () => {
  const theme = useTheme();
  const { translation, translation_dict, language, setLanguage } =
    useContext(LanguageContext);

  const { pathname } = useLocation();
  const [active, setActive] = useState("");
  const navigate = useNavigate();
  const [drawerWidth, setDrawerWidth] = useState(
    `${Math.min(Math.max(window.innerWidth * 0.2, 287), 320)}px`
  );

  const navItems = [
    {
      link: "launch_model",
      icon: <RocketLaunchOutlined />,
    },
    {
      link: "running_models",
      icon: <SmartToyOutlined />,
    },
    {
      link: "register_model",
      icon: <AddBoxOutlined />,
    },
    {
      link: "contact_us",
      icon: <GitHub />,
    },
  ];

  console.log(translation);

  useEffect(() => {
    setActive(pathname.substring(1));
  }, [pathname]);

  useEffect(() => {
    const screenWidth = window.innerWidth;
    const maxDrawerWidth = Math.min(Math.max(screenWidth * 0.2, 287), 320);
    setDrawerWidth(`${maxDrawerWidth}px`);

    // Update the drawer width on window resize
    const handleResize = () => {
      const newScreenWidth = window.innerWidth;
      const newMaxDrawerWidth = Math.min(
        Math.max(newScreenWidth * 0.2, 287),
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
              <Typography fontWeight="bold" fontSize="1.7rem">
                {"Xinference"}
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>

      {/* Pages or Screens */}
      <Box>
        <Box width="100%">
          <Box m="1.5rem 2rem 2rem 3rem"></Box>
          <List>
            {navItems.map(({ link, icon }) => {
              if (!icon) {
                return (
                  <Typography
                    key={translation.page_title[link]}
                    sx={{ m: "2.25rem 0 1rem 3rem" }}
                  >
                    {translation.page_title[link]}
                  </Typography>
                );
              }

              return (
                <ListItem key={translation.page_title[link]}>
                  <ListItemButton
                    onClick={() => {
                      if (link === "contact_us") {
                        window.open(
                          "https://github.com/xorbitsai/inference",
                          "_blank",
                          "noreferrer"
                        );
                      } else if (link === "launch_model") {
                        navigate(`/`);
                        setActive(link);
                        console.log(active);
                      } else {
                        navigate(`/${link}`);
                        setActive(link);
                        console.log(active);
                      }
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        ml: "2rem",
                      }}
                    >
                      {icon}
                    </ListItemIcon>
                    <ListItemText primary={translation.page_title[link]} />
                    <ChevronRightOutlined sx={{ ml: "auto" }} />
                  </ListItemButton>
                </ListItem>
              );
            })}

            {/* Translation */}
            <ListItem key="translation">
              <ListItemButton
                onClick={() => {}}
                disableRipple={true}
                sx={{
                  "&:hover": {
                    backgroundColor: "transparent",
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    ml: "2rem",
                    mr: "-0.4rem",
                  }}
                >
                  <TranslateOutlined />
                </ListItemIcon>
                <Select
                  value={language}
                  onChange={(event) => setLanguage(event.target.value)}
                  variant="outlined"
                  displayEmpty
                  sx={{
                    "& .MuiOutlinedInput-input": {
                      padding: "10px 0px 10px 14px",
                      height: "fit-content",
                    },
                    "& .MuiOutlinedInput-root": {
                      marginLeft: "-14px",
                    },
                  }}
                >
                  {Object.keys(translation_dict).map((lang) => (
                    <MenuItem key={lang} value={lang}>
                      {lang}
                    </MenuItem>
                  ))}
                </Select>
              </ListItemButton>
            </ListItem>
          </List>
        </Box>
      </Box>
    </Drawer>
  );
};

export default MenuSide;
