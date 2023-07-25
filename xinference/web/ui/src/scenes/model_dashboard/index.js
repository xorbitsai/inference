import React, { useEffect, useState } from "react";
import { Box, Typography, useTheme } from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import { tokens } from "../../theme";
import AdminPanelSettingsOutlinedIcon from "@mui/icons-material/AdminPanelSettingsOutlined";
import LockOpenOutlinedIcon from "@mui/icons-material/LockOpenOutlined";
import SecurityOutlinedIcon from "@mui/icons-material/SecurityOutlined";

const ModelDashboard = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  const [modelData, setModelData] = useState([]);

  const columns = [
    { field: "id", headerName: "ID" },
    {
      field: "model_name",
      headerName: "Name",
      cellClassName: "name-column--cell",
    },
    {
      field: "model_size_in_billions",
      headerName: "Age",
    },
    {
      field: "quantization",
      headerName: "Phone Number",
    },
  ];

  useEffect(() => {
    fetch("http://localhost:9997/v1/models", {
      method: "GET",
    })
      .then((response) => response.json())
      .then((data) => {
        const newModelData = [];
        Object.entries(data).forEach(([key, value]) => {
          let newValue = { ...value, id: key };
          newModelData.push(newValue);
        });
        setModelData(newModelData); // Update state
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }, []);

  return (
    <Box m="20px">
      <Box
        m="40px 0 0 0"
        height="75vh"
        sx={{
          "& .MuiDataGrid-root": {
            border: "none",
          },
          "& .MuiDataGrid-cell": {
            borderBottom: "none",
          },
          "& .name-column--cell": {
            color: colors.greenAccent[300],
          },
          "& .MuiDataGrid-columnHeaders": {
            backgroundColor: colors.blueAccent[700],
            borderBottom: "none",
          },
          "& .MuiDataGrid-virtualScroller": {
            backgroundColor: colors.primary[400],
          },
          "& .MuiDataGrid-footerContainer": {
            borderTop: "none",
            backgroundColor: colors.blueAccent[700],
          },
          "& .MuiCheckbox-root": {
            color: `${colors.greenAccent[200]} !important`,
          },
        }}
      >
        <DataGrid checkboxSelection rows={modelData} columns={columns} />
      </Box>
    </Box>
  );
};

export default ModelDashboard;
