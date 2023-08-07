import { Box, IconButton, useTheme } from "@mui/material";
import { useContext } from "react";
import { ColorModeContext } from "../theme";
import NotificationsOutlinedIcon from "@mui/icons-material/NotificationsOutlined";

const MenuTop = () => {
  return (
    <Box display="flex" justifyContent="right" p={2}>
      {/* Settings */}
      <Box display="flex">
        <IconButton>
          <NotificationsOutlinedIcon />
        </IconButton>
      </Box>
    </Box>
  );
};

export default MenuTop;
