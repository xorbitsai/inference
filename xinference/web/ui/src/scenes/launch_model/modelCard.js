import React, { useState, useContext, useEffect } from "react";
import { v1 as uuidv1 } from "uuid";
import { ApiContext } from "../../components/apiContext";
import { FormControl, InputLabel, Select, MenuItem, Box } from "@mui/material";

const CARD_HEIGHT = 350;
const CARD_WIDTH = 270;

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

  // UseEffects for parameter selection, change options based on previous selections
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
    container: {
      display: "block",
      position: "relative",
      width: `${CARD_WIDTH}px`,
      height: `${CARD_HEIGHT}px`,
      border: "1px solid #ddd",
      borderRadius: "20px",
      background: "white",
      overflow: "hidden",
    },
    descriptionCard: {
      position: "relative",
      top: "-1px",
      left: "-1px",
      width: `${CARD_WIDTH}px`,
      height: `${CARD_HEIGHT}px`,
      border: "1px solid #ddd",
      padding: "20px",
      borderRadius: "20px",
      background: "white",
    },
    parameterCard: {
      position: "relative",
      top: `-${CARD_HEIGHT + 1}px`,
      left: "-1px",
      width: `${CARD_WIDTH}px`,
      height: `${CARD_HEIGHT}px`,
      border: "1px solid #ddd",
      padding: "20px",
      borderRadius: "20px",
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
    slideIn: {
      transform: "translateX(0%)",
      transition: "transform 0.2s ease-in-out",
    },
    slideOut: {
      transform: "translateX(100%)",
      transition: "transform 0.2s ease-in-out",
    },
  };

  // Set two different states based on mouse hover
  return (
    <Box
      style={styles.container}
      onMouseEnter={() => setSelected(true)}
      onMouseLeave={() => setSelected(false)}
    >
      {/* First state: show description page */}
      <Box style={styles.descriptionCard}>
        <img style={styles.img} src={imgURL} alt={modelData.model_name} />
        <h2 style={styles.h2}>{modelData.model_name}</h2>
        <p style={styles.p}>{modelData.model_description}</p>
        <p style={styles.instructionText}>
          Hover with mouse to launch the model
        </p>
      </Box>
      {/* Second state: show parameter selection page */}
      <Box
        style={
          selected
            ? { ...styles.parameterCard, ...styles.slideIn }
            : { ...styles.parameterCard, ...styles.slideOut }
        }
      >
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
    </Box>
  );
};

export default ModelCard;
