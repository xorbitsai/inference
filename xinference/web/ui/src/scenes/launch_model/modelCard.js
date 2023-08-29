import React, { useState, useContext, useEffect } from "react";
import { v1 as uuidv1 } from "uuid";
import { ApiContext } from "../../components/apiContext";
import { FormControl, InputLabel, Select, MenuItem, Box } from "@mui/material";

const ModelCard = ({ imgURL, url, jsonData }) => {
  const modelData = jsonData;
  const [selected, setSelected] = useState(false);
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext);
  const { isUpdatingModel } = useContext(ApiContext);

  // Model parameter selections
  const [modelFormat, setModelFormat] = useState("");
  const [modelSize, setModelSize] = useState("");
  const [quantization, setQuantization] = useState("");

  const [formatOptions, setFormatOptions] = useState([]);
  const [sizeOptions, setSizeOptions] = useState([]);
  const [quantizationOptions, setQuantizationOptions] = useState([]);

  useEffect(() => {
    if (modelData) {
      const modelFamily = modelData.model_specs;
      const formats = [
        ...new Set(modelFamily.map((spec) => spec.model_format)),
      ];
      setFormatOptions(formats);
    }
  }, [modelData]);

  useEffect(() => {
    if (modelFormat && modelData) {
      const modelFamily = modelData.model_specs;
      const sizes = [
        ...new Set(
          modelFamily
            .filter((spec) => spec.model_format === modelFormat)
            .map((spec) => spec.model_size_in_billions)
        ),
      ];
      setSizeOptions(sizes);
    }
  }, [modelFormat, modelData]);

  useEffect(() => {
    if (modelFormat && modelSize && modelData) {
      const modelFamily = modelData.model_specs;
      const quants = [
        ...new Set(
          modelFamily
            .filter(
              (spec) =>
                spec.model_format === modelFormat &&
                spec.model_size_in_billions === parseFloat(modelSize)
            )
            .flatMap((spec) => spec.quantizations)
        ),
      ];
      setQuantizationOptions(quants);
    }
  }, [modelFormat, modelSize, modelData]);

  const launchModel = (url) => {
    setIsCallingApi(true);

    const uuid = uuidv1();
    const modelDataWithID = {
      model_uid: uuid,
      model_name: modelData.model_name,
      model_format: modelFormat,
      model_size_in_billions: modelSize,
      quantization: quantization,
    };

    // First fetch request to initiate the model
    fetch(url + "/v1/models", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(modelDataWithID),
    })
      .then((response) => {
        response.json();
      })
      .then(() => {
        // Second fetch request to build the gradio page
        return fetch(url + "/v1/ui/" + uuid, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(modelDataWithID),
        });
      })
      .then((response) => {
        response.json();
      })
      .then(() => {
        window.open(url + "/" + uuid, "_blank", "noreferrer");
        setIsCallingApi(false);
      })
      .catch((error) => {
        console.error("Error:", error);
        setIsCallingApi(false);
      });
  };

  const styles = {
    card: {
      width: "280px",
      border: "1px solid #ddd",
      borderRadius: "20px",
      padding: "15px",
      background: "white",
    },
    img: {
      display: "block",
      margin: "0 auto",
      width: "180px",
      height: "180px",
      objectFit: "cover",
      borderRadius: "10px",
    },
    h2: {
      margin: "10px 10px",
      fontSize: "20px",
    },
    p: {
      fontSize: "14px",
      padding: "0px 0px 15px 0px",
    },
    button: {
      display: "block",
      padding: "10px 24px",
      margin: "0 auto",
      marginTop: "30px",
      border: "none",
      borderRadius: "5px",
      cursor: "pointer",
      fontWeight: "bold",
    },
    instructionText: {
      fontSize: "12px",
      color: "#666666",
      fontStyle: "italic",
      margin: "10px 0",
      textAlign: "center",
    },
  };

  if (selected) {
    return (
      <Box style={styles.card} onMouseLeave={() => setSelected(false)}>
        <Box display="flex" flexDirection="column" width="80%" mx="auto">
          <FormControl variant="outlined" margin="dense">
            <InputLabel id="modelFormat-label">Model Format</InputLabel>
            <Select
              labelId="modelFormat-label"
              value={modelFormat}
              onChange={(e) => setModelFormat(e.target.value)}
              label="Model Format"
            >
              {formatOptions.map((format) => (
                <MenuItem key={format} value={format}>
                  {format}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl
            variant="outlined"
            margin="dense"
            disabled={!modelFormat}
          >
            <InputLabel id="modelSize-label">Model Size</InputLabel>
            <Select
              labelId="modelSize-label"
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value)}
              label="Model Size"
            >
              {sizeOptions.map((size) => (
                <MenuItem key={size} value={size}>
                  {size}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl
            variant="outlined"
            margin="dense"
            disabled={!modelFormat || !modelSize}
          >
            <InputLabel id="quantization-label">Quantization</InputLabel>
            <Select
              labelId="quantization-label"
              value={quantization}
              onChange={(e) => setQuantization(e.target.value)}
              label="Quantization"
            >
              {quantizationOptions.map((quant) => (
                <MenuItem key={quant} value={quant}>
                  {quant}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        <button
          style={{
            ...styles.button,
            color: isCallingApi || isUpdatingModel ? "white" : "#ea580c",
            background:
              isCallingApi || isUpdatingModel
                ? "gray"
                : "linear-gradient(to bottom right, #ffedd5, #fdba74)",
          }}
          onClick={() => {
            launchModel(url, modelData);
          }}
          disabled={isCallingApi || isUpdatingModel}
        >
          {isCallingApi || isUpdatingModel ? "Loading..." : "Launch"}
        </button>
      </Box>
    );
  }

  return (
    <Box style={styles.card} onMouseEnter={() => setSelected(true)}>
      <img style={styles.img} src={imgURL} alt={modelData.model_name} />
      <h2 style={styles.h2}>{modelData.model_name}</h2>
      <p style={styles.p}>{modelData.model_description}</p>
      <p style={styles.instructionText}>Hover with mouse to launch the model</p>
    </Box>
  );
};

export default ModelCard;
