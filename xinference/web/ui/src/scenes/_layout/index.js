import React from "react";
import { Box } from "@mui/material";
import { Outlet } from "react-router-dom";
import MenuTop from "../../components/MenuTop";
import MenuSide from "../../components/MenuSide";

const Layout = () => {
  return (
    <Box display="flex" width="100%" height="100%">
      <MenuSide />
      <Box flexGrow={1}>
        <MenuTop />
        <Outlet />
      </Box>
    </Box>
  );
};

export default Layout;
