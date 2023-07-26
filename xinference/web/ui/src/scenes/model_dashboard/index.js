import React, { useEffect, useState } from "react";
import { Box, Typography, useTheme } from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import { tokens } from "../../theme";
import OpenInBrowserOutlinedIcon from "@mui/icons-material/OpenInBrowserOutlined";
import DeleteOutlineOutlinedIcon from "@mui/icons-material/DeleteOutlineOutlined";

const ModelDashboard = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [modelData, setModelData] = useState([]);
  const [isCalling, setIsCalling] = useState(false);

  const update = () => {
    fetch("http://localhost:9997/v1/models", {
      method: "GET",
    })
      .then((response) => response.json())
      .then((data) => {
        const newModelData = [];
        Object.entries(data).forEach(([key, value]) => {
          let newValue = {
            ...value,
            id: key,
            url: key,
          };
          newModelData.push(newValue);
        });
        setModelData(newModelData);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  useEffect(() => {
    update();
  }, []);

  const columns = [
    {
      field: "id",
      headerName: "ID",
      flex: 1,
      minWidth: 250,
    },
    {
      field: "model_name",
      headerName: "Name",
      flex: 1,
    },
    {
      field: "model_size_in_billions",
      headerName: "Size",
      flex: 1,
    },
    {
      field: "quantization",
      headerName: "Quantization",
      flex: 1,
    },
    {
      field: "url",
      headerName: "",
      flex: 1,
      minWidth: 200,
      renderCell: ({ row: { url } }) => {
        const openUrl = "http://localhost:9997/" + url;
        const closeUrl = "http://localhost:9997/v1/models/" + url;
        return (
          <Box
            style={{
              width: "100%",
              display: "flex",
              justifyContent: "space-around",
              alignItems: "center",
            }}
          >
            <button
              style={{ borderWidth: "0px", backgroundColor: "transparent" }}
              onClick={() => window.open(openUrl, "_blank", "noreferrer")}
            >
              <Box
                width="70px"
                m="0 auto"
                p="5px"
                display="flex"
                justifyContent="center"
                backgroundColor={colors.greenAccent[600]}
                borderRadius="4px"
              >
                <OpenInBrowserOutlinedIcon />
                <Typography sx={{ ml: "5px" }}>Open</Typography>
              </Box>
            </button>
            <button
              style={{ borderWidth: "0px", backgroundColor: "transparent" }}
              onClick={() => {
                if (isCalling) {
                  return;
                }
                setIsCalling(true);
                fetch(closeUrl, {
                  method: "DELETE",
                })
                  .then((response) => {
                    response.json();
                  })
                  .then((data) => {
                    console.log(data);
                    setIsCalling(false);
                    update();
                  })
                  .catch((error) => {
                    console.error("Error:", error);
                    setIsCalling(false);
                    update();
                  });
              }}
            >
              <Box
                width="75px"
                m="0 auto"
                p="5px"
                display="flex"
                justifyContent="center"
                backgroundColor="red"
                borderRadius="4px"
              >
                <DeleteOutlineOutlinedIcon />
                <Typography sx={{ ml: "5px" }}>Delete</Typography>
              </Box>
            </button>
          </Box>
        );
      },
    },
  ];

  return (
    <Box m="20px">
      <Box
        m="40px 0 0 0"
        height="75vh"
        sx={{
          "& .MuiDataGrid-root": {
            border: "none",
            width: "95% !important",
            maxWidth: "1200px !important",
            minWidth: "600px !important",
          },
          "& .MuiDataGrid-cell": {
            borderBottom: "none",
          },
          "& .CustomWide-cell": {
            minWidth: "250px !important",
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
        }}
      >
        <DataGrid rows={modelData} columns={columns} />
      </Box>
    </Box>
  );
};

export default ModelDashboard;
