import React, { useEffect, useState, useContext } from "react";
import { Box, Stack } from "@mui/material";
import { ApiContext } from "../../components/apiContext";
import { DataGrid } from "@mui/x-data-grid";
import Title from "../../components/Title";
import OpenInBrowserOutlinedIcon from "@mui/icons-material/OpenInBrowserOutlined";
import DeleteOutlineOutlinedIcon from "@mui/icons-material/DeleteOutlineOutlined";

const RunningModels = () => {
  const [modelData, setModelData] = useState([]);
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext);
  const { isUpdatingModel, setIsUpdatingModel } = useContext(ApiContext);
  const endPoint = useContext(ApiContext).endPoint;

  const update = (isCallingApi) => {
    if (isCallingApi) {
      setModelData([
        { id: "Loading, do not refresh page...", url: "IS_LOADING" },
      ]);
    } else {
      setIsUpdatingModel(true);
      fetch(`${endPoint}/v1/models`, {
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
          setIsUpdatingModel(false);
        })
        .catch((error) => {
          console.error("Error:", error);
          setIsUpdatingModel(false);
        });
    }
  };

  useEffect(() => {
    update(isCallingApi);
    // eslint-disable-next-line
  }, [isCallingApi]);

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
      headerName: "Actions",
      flex: 1,
      minWidth: 200,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
      renderCell: ({ row }) => {
        const url = row.url;
        const openUrl = `${endPoint}/` + url;
        const closeUrl = `${endPoint}/v1/models/` + url;
        const gradioUrl = `${endPoint}/v1/gradio/` + url;

        if (url === "IS_LOADING") {
          return <div></div>;
        }

        return (
          <Box
            style={{
              width: "100%",
              display: "flex",
              justifyContent: "left",
              alignItems: "left",
            }}
          >
            <button
              style={{
                borderWidth: "0px",
                backgroundColor: "transparent",
                paddingLeft: "0px",
                paddingRight: "10px",
              }}
              onClick={() => {
                if (isCallingApi || isUpdatingModel) {
                  // Make sure no ongoing call
                  return;
                }

                setIsCallingApi(true);

                fetch(openUrl, {
                  method: "HEAD",
                })
                  .then((response) => {
                    if (response.status === 404) {
                      // If web UI doesn't exist (404 Not Found)
                      console.log("UI does not exist, creating new...");
                      return fetch(gradioUrl, {
                        method: "POST",
                        headers: {
                          "Content-Type": "application/json",
                        },
                      })
                        .then((response) => response.json())
                        .then(() =>
                          window.open(openUrl, "_blank", "noopener noreferrer")
                        )
                        .finally(() => setIsCallingApi(false));
                    } else if (response.ok) {
                      // If web UI does exist
                      console.log("UI exists, opening...");
                      window.open(openUrl, "_blank", "noopener noreferrer");
                      setIsCallingApi(false);
                    } else {
                      // Other HTTP errors
                      console.error(
                        `Unexpected response status: ${response.status}`
                      );
                      setIsCallingApi(false);
                    }
                  })
                  .catch((error) => {
                    console.error("Error:", error);
                    setIsCallingApi(false);
                  });
              }}
            >
              <Box
                width="70px"
                m="0 auto"
                p="5px"
                display="flex"
                justifyContent="center"
                borderRadius="4px"
                style={{
                  border: "1px solid #e5e7eb",
                  borderWidth: "1px",
                  borderColor: "#e5e7eb",
                  background:
                    "linear-gradient(to bottom right, #f3f4f6, #e5e7eb)",
                }}
              >
                <OpenInBrowserOutlinedIcon />
              </Box>
            </button>
            <button
              style={{
                borderWidth: "0px",
                backgroundColor: "transparent",
                paddingLeft: "0px",
                paddingRight: "10px",
              }}
              onClick={() => {
                if (isCallingApi | isUpdatingModel) {
                  return;
                }
                setIsCallingApi(true);
                fetch(closeUrl, {
                  method: "DELETE",
                })
                  .then((response) => {
                    response.json();
                  })
                  .then((data) => {
                    console.log(data);
                    setIsCallingApi(false);
                  })
                  .catch((error) => {
                    console.error("Error:", error);
                    setIsCallingApi(false);
                  });
              }}
            >
              <Box
                width="70px"
                m="0 auto"
                p="5px"
                display="flex"
                justifyContent="center"
                borderRadius="4px"
                style={{
                  border: "1px solid #e5e7eb",
                  borderWidth: "1px",
                  borderColor: "#e5e7eb",
                  background:
                    "linear-gradient(to bottom right, #f3f4f6, #e5e7eb)",
                }}
              >
                <DeleteOutlineOutlinedIcon />
              </Box>
            </button>
          </Box>
        );
      },
    },
  ];

  return (
    <Box m="20px">
      <Title title="Running Models" />
      <Box m="40px 0 0 0" height="75vh">
        <DataGrid
          rows={modelData}
          columns={columns}
          sx={{
            "& .MuiDataGrid-main": {
              width: "95% !important",
              overflow: "visible",
            },
            "& .MuiDataGrid-row": {
              background: "white",
              margin: "10px 0px",
            },
            "& .MuiDataGrid-cell": {
              borderBottom: "none",
            },
            "& .CustomWide-cell": {
              minWidth: "250px !important",
            },
            "& .MuiDataGrid-columnHeaders": {
              borderBottom: "none",
            },
            "& .MuiDataGrid-columnHeaderTitle": {
              fontWeight: "bold",
            },
            "& .MuiDataGrid-virtualScroller": {
              overflowX: "visible !important",
              overflow: "visible",
            },
            "& .MuiDataGrid-footerContainer": {
              borderTop: "none",
            },
            "border-width": "0px",
          }}
          slots={{
            noRowsOverlay: () => (
              <Stack height="100%" alignItems="center" justifyContent="center">
                No Running Models
              </Stack>
            ),
            noResultsOverlay: () => (
              <Stack height="100%" alignItems="center" justifyContent="center">
                No Running Models Matches
              </Stack>
            ),
          }}
        />
      </Box>
    </Box>
  );
};

export default RunningModels;
