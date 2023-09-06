import React, { useState, useContext } from "react";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import { ApiContext } from "../../components/apiContext";
import {
  Box,
  FormControl,
  FormGroup,
  Checkbox,
  FormControlLabel,
} from "@mui/material";

const SUPPORTED_LANGUAGES_DICT = { en: "english", zh: "mandarin" };
const SUPPORTED_FEATURES = ["embed", "generate", "chat"];
const PYTORCH_QUANTIZATIONS = ["4-bit", "8-bit", "none"];
const GGMLV3_QUANTIZATIONS = [
  "q2_K",
  "q3_K_L",
  "q3_K_M",
  "q3_K_S",
  "q4_0",
  "q4_1",
  "q4_K_M",
  "q4_K_S",
  "q5_0",
  "q5_1",
  "q5_K_M",
  "q5_K_S",
  "q6_K",
  "q8_0",
];

// Convert dictionary of supported languages into list
const SUPPORTED_LANGUAGES = Object.keys(SUPPORTED_LANGUAGES_DICT);

const RegisterModel = () => {
  const endPoint = useContext(ApiContext).endPoint;
  const [formData, setFormData] = useState({
    version: 1,
    context_length: 2048,
    model_name: "custom-llama-2",
    model_lang: ["en"],
    model_ability: ["generate"],
    model_specs: [
      {
        model_format: "pytorch",
        model_size_in_billions: 7,
        quantizations: ["4-bit", "8-bit", "none"],
        model_id: "meta-llama/Llama-2-7b",
        model_uri: "file:///path/to/llama-2-7b",
      },
      {
        model_format: "ggmlv3",
        model_size_in_billions: 7,
        quantizations: ["q4_0", "q8_0"],
        model_id: "TheBloke/Llama-2-7B-GGML",
        model_file_name_template: "llama-2-7b.ggmlv3.{quantization}.bin",
      },
    ],
  });
  const pytorchSelected = formData.model_specs.some(
    (spec) => spec.model_format === "pytorch"
  );
  const ggmlSelected = formData.model_specs.some(
    (spec) => spec.model_format === "ggmlv3"
  );

  // context length must be
  // 1. all numeric characters (0-9)
  // 2. does not begin with 0
  // 3. between 1 and 10 digits long
  const errorContextLength = !/^(?!0)\d{1,10}$/.test(formData.context_length);
  // model name must be
  // 1. all alphanueric characters, dashes, or periods
  // 2. does not contain two consecutive dashes or periods
  // 3. does not begin nor end with dashes or periods
  // 4. between 1 and 255 characters long
  const errorModelName =
    !/^(?!-|\.)(?!.*--)(?!.*\.\.)(?!.*\.-)(?!.*-\.)[A-Za-z0-9-.]{1,255}(?<!-|\.)$/.test(
      formData.model_name
    );
  const errorLanguage =
    formData.model_lang === undefined || formData.model_lang.length === 0;
  const errorAbility =
    formData.model_ability === undefined || formData.model_ability.length === 0;
  const errorModelFormat =
    formData.model_specs === undefined || formData.model_specs.length === 0;
  const errorPytorchQuantization = false;
  const errorGgmlv3Quantization = false;

  const toggleLanguage = (lang) => {
    if (formData.model_lang.includes(lang)) {
      setFormData({
        ...formData,
        model_lang: formData.model_lang.filter((l) => l !== lang),
      });
    } else {
      setFormData({
        ...formData,
        model_lang: [...formData.model_lang, lang],
      });
    }
  };

  const toggleAbility = (ability) => {
    if (formData.model_ability.includes(ability)) {
      setFormData({
        ...formData,
        model_ability: formData.model_ability.filter((a) => a !== ability),
      });
    } else {
      setFormData({
        ...formData,
        model_ability: [...formData.model_ability, ability],
      });
    }
  };

  const togglePytorchFormat = () => {
    const updatedSpecs = [...formData.model_specs];
    const specIndex = updatedSpecs.findIndex(
      (spec) => spec.model_format === "pytorch"
    );
    if (specIndex > -1) {
      updatedSpecs.splice(specIndex, 1);
    } else {
      updatedSpecs.push({
        model_format: "pytorch",
        model_size_in_billions: 7,
        quantizations: ["4-bit", "8-bit", "none"],
        model_id: "meta-llama/Llama-2-7b",
        model_uri: "file:///path/to/llama-2-7b",
      });
    }
    setFormData({ ...formData, model_specs: updatedSpecs });
  };

  const toggleGgmlv3Format = () => {
    const updatedSpecs = [...formData.model_specs];
    const specIndex = updatedSpecs.findIndex(
      (spec) => spec.model_format === "ggmlv3"
    );
    if (specIndex > -1) {
      updatedSpecs.splice(specIndex, 1);
    } else {
      updatedSpecs.push({
        model_format: "ggmlv3",
        model_size_in_billions: 7,
        quantizations: ["q4_0", "q8_0"],
        model_id: "TheBloke/Llama-2-7B-GGML",
        model_file_name_template: "llama-2-7b.ggmlv3.{quantization}.bin",
      });
    }
    setFormData({ ...formData, model_specs: updatedSpecs });
  };

  // Update the PyTorch spec in model_specs
  function updatePyTorchSpec(updatedField) {
    setFormData((prevState) => {
      const updatedSpecs = prevState.model_specs.map((spec) => {
        if (spec.model_format === "pytorch") {
          return {
            ...spec,
            ...updatedField,
          };
        }
        return spec;
      });
      return { ...prevState, model_specs: updatedSpecs };
    });
  }

  // Update the GGMLv3 spec in model_specs
  function updateGGMLSpec(updatedField) {
    setFormData((prevState) => {
      const updatedSpecs = prevState.model_specs.map((spec) => {
        if (spec.model_format === "ggmlv3") {
          return {
            ...spec,
            ...updatedField,
          };
        }
        return spec;
      });
      return { ...prevState, model_specs: updatedSpecs };
    });
  }

  function updateQuantizations(model_format, quantization) {
    setFormData((prevState) => {
      const updatedSpecs = prevState.model_specs.map((spec) => {
        if (spec.model_format === model_format) {
          const isQuantizationSelected =
            spec.quantizations.includes(quantization);

          if (isQuantizationSelected) {
            return {
              ...spec,
              quantizations: spec.quantizations.filter(
                (q) => q !== quantization
              ),
            };
          } else {
            return {
              ...spec,
              quantizations: [...spec.quantizations, quantization],
            };
          }
        }
        return spec;
      });
      return { ...prevState, model_specs: updatedSpecs };
    });
  }

  return (
    <Box m="20px">
      <h1>Model Registration</h1>

      {/* Base Information */}
      <FormControl sx={styles.baseFormControl}>
        <TextField
          error={errorContextLength}
          label="Context Length"
          value={formData.context_length}
          size="small"
          helperText="Positive integer between 1 and 10 digits"
          onChange={(event) => {
            let value = event.target.value;
            // Remove leading zeros
            if (/^0+/.test(value)) {
              value = value.replace(/^0+/, "") || "0";
            }
            // Ensure it's a positive integer, if not set it to the minimum
            if (!/^\d+$/.test(value) || parseInt(value) < 1) {
              value = "1";
            }
            // Update with the processed value
            setFormData({
              ...formData,
              context_length: Number(value),
            });
          }}
        />
        <Box padding="10px"></Box>
        <TextField
          label="Model Name"
          error={errorModelName}
          defaultValue={formData.model_name}
          size="small"
          helperText="Alphanumeric characters with properly placed hyphens and periods"
          onChange={(event) =>
            setFormData({ ...formData, model_name: event.target.value })
          }
        />
        <Box padding="5px"></Box>

        {/* Model Languages Checkboxes */}
        <label
          style={{
            paddingLeft: 5,
            color: errorLanguage ? "#d8342c" : "inherit",
          }}
        >
          Model Languages
        </label>
        <Box sx={styles.checkboxWrapper}>
          {SUPPORTED_LANGUAGES.map((lang) => (
            <Box key={lang} sx={{ marginRight: "10px" }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={formData.model_lang.includes(lang)}
                    onChange={() => toggleLanguage(lang)}
                    name={lang}
                    sx={errorLanguage ? styles.checkboxError : {}}
                  />
                }
                label={SUPPORTED_LANGUAGES_DICT[lang]}
                style={{
                  paddingLeft: 10,
                  color: errorLanguage ? "#d8342c" : "inherit",
                }}
              />
            </Box>
          ))}
        </Box>
        <Box padding="5px"></Box>

        {/* Model Abilities Checkboxes */}
        <label
          style={{
            paddingLeft: 5,
            color: errorAbility ? "#d8342c" : "inherit",
          }}
        >
          Model Abilities
        </label>
        <Box sx={styles.checkboxWrapper}>
          {SUPPORTED_FEATURES.map((ability) => (
            <Box key={ability} sx={{ marginRight: "10px" }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={formData.model_ability.includes(ability)}
                    onChange={() => toggleAbility(ability)}
                    name={ability}
                    sx={errorAbility ? styles.checkboxError : {}}
                  />
                }
                label={ability}
                style={{
                  paddingLeft: 10,
                  color: errorAbility ? "#d8342c" : "inherit",
                }}
              />
            </Box>
          ))}
        </Box>
        <Box padding="5px"></Box>

        {/* Model Spec Checkboxes */}
        <label
          style={{
            paddingLeft: 5,
            color: errorModelFormat ? "#d8342c" : "inherit",
          }}
        >
          Model Formats
        </label>
        <Box sx={styles.checkboxWrapper}>
          {/* PyTorch */}
          <Box sx={{ marginRight: "10px" }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={formData.model_specs.some(
                    (spec) => spec.model_format === "pytorch"
                  )}
                  onChange={togglePytorchFormat}
                  name="pytorch"
                  sx={
                    errorModelFormat
                      ? {
                          color: "#d8342c",
                          "&.Mui-checked": {
                            color: "#d8342c",
                          },
                        }
                      : {}
                  }
                />
              }
              label="pytorch"
              style={{
                paddingLeft: 10,
                color: errorModelFormat ? "#d8342c" : "inherit",
              }}
            />
          </Box>

          {/* GGMLv3 */}
          <Box sx={{ marginRight: "10px" }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={formData.model_specs.some(
                    (spec) => spec.model_format === "ggmlv3"
                  )}
                  onChange={toggleGgmlv3Format}
                  name="ggmlv3"
                  sx={errorModelFormat ? styles.checkboxError : {}}
                />
              }
              label="ggmlv3"
              style={{
                paddingLeft: 10,
                color: errorModelFormat ? "#d8342c" : "inherit",
              }}
            />
          </Box>
        </Box>
      </FormControl>

      {/* PyTorch Model Spec Form */}
      {pytorchSelected && (
        <FormGroup>
          <h2>PyTorch Model Spec</h2>
          <TextField
            label="Model Size in Billions"
            type="number"
            size="small"
            value={
              formData.model_specs.find(
                (spec) => spec.model_format === "pytorch"
              ).model_size_in_billions
            }
            onChange={(e) => {
              let value = e.target.value;
              // Remove leading zeros
              if (/^0+/.test(value)) {
                value = value.replace(/^0+/, "") || "0";
              }
              // Ensure it's a positive integer, if not set it to the minimum
              if (!/^\d+$/.test(value) || parseInt(value) < 1) {
                value = "1";
              }
              // Update the spec with the processed value
              updatePyTorchSpec({
                model_size_in_billions: Number(value),
              });
            }}
          />
          <Box padding="10px"></Box>
          <TextField
            label="Model ID"
            size="small"
            value={
              formData.model_specs.find(
                (spec) => spec.model_format === "pytorch"
              ).model_id
            }
            onChange={(e) => updatePyTorchSpec({ model_id: e.target.value })}
          />
          <Box padding="10px"></Box>
          <TextField
            label="Model URI"
            size="small"
            value={
              formData.model_specs.find(
                (spec) => spec.model_format === "pytorch"
              ).model_uri
            }
            onChange={(e) => updatePyTorchSpec({ model_uri: e.target.value })}
          />
          <Box padding="10px"></Box>

          {/* PyTorch Quantizations */}
          <label
            style={{
              paddingLeft: 10,
              color: errorPytorchQuantization ? "#d8342c" : "inherit",
            }}
          >
            PyTorch Quantizations
          </label>
          <Box sx={styles.checkboxWrapper}>
            {PYTORCH_QUANTIZATIONS.map((quantization) => (
              <Box key={quantization} sx={{ marginRight: "10px" }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.model_specs
                        .find((spec) => spec.model_format === "pytorch")
                        .quantizations.includes(quantization)}
                      onChange={() =>
                        updateQuantizations("pytorch", quantization)
                      }
                      name={quantization}
                      sx={
                        errorPytorchQuantization
                          ? {
                              color: "#d8342c",
                              "&.Mui-checked": {
                                color: "#d8342c",
                              },
                            }
                          : {}
                      }
                    />
                  }
                  label={quantization}
                  style={{
                    color: errorPytorchQuantization ? "#d8342c" : "inherit",
                  }}
                />
              </Box>
            ))}
          </Box>
        </FormGroup>
      )}

      {/* GGMLv3 Model Spec Form */}
      {ggmlSelected && (
        <FormGroup>
          <h2>GGMLv3 Model Spec</h2>
          <TextField
            label="Model Size in Billions"
            type="number"
            size="small"
            value={
              formData.model_specs.find(
                (spec) => spec.model_format === "ggmlv3"
              ).model_size_in_billions
            }
            onChange={(e) => {
              let value = e.target.value;
              // Remove leading zeros
              if (/^0+/.test(value)) {
                value = value.replace(/^0+/, "") || "0";
              }
              // Ensure it's a positive integer, if not set it to the minimum
              if (!/^\d+$/.test(value) || parseInt(value) < 1) {
                value = "1";
              }
              // Update the spec with the processed value
              updateGGMLSpec({
                model_size_in_billions: Number(value),
              });
            }}
          />
          <Box padding="10px"></Box>
          <TextField
            label="Model ID"
            size="small"
            value={
              formData.model_specs.find(
                (spec) => spec.model_format === "ggmlv3"
              ).model_id
            }
            onChange={(e) => updateGGMLSpec({ model_id: e.target.value })}
          />
          <Box padding="10px"></Box>
          <TextField
            label="Model File Name Template"
            size="small"
            value={
              formData.model_specs.find(
                (spec) => spec.model_format === "ggmlv3"
              ).model_file_name_template
            }
            onChange={(e) =>
              updateGGMLSpec({ model_file_name_template: e.target.value })
            }
          />
          <Box padding="10px"></Box>

          {/* GGMLv3 Quantizations */}
          <label
            style={{
              paddingLeft: 10,
              color: errorGgmlv3Quantization ? "#d8342c" : "inherit",
            }}
          >
            GGMLv3 Quantizations
          </label>
          <Box sx={styles.checkboxWrapper}>
            {GGMLV3_QUANTIZATIONS.map((quantization) => (
              <Box key={quantization} sx={{ marginRight: "10px" }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.model_specs
                        .find((spec) => spec.model_format === "ggmlv3")
                        .quantizations.includes(quantization)}
                      onChange={() =>
                        updateQuantizations("ggmlv3", quantization)
                      }
                      name={quantization}
                      sx={
                        errorGgmlv3Quantization
                          ? {
                              color: "#d8342c",
                              "&.Mui-checked": {
                                color: "#d8342c",
                              },
                            }
                          : {}
                      }
                    />
                  }
                  label={quantization}
                  style={{
                    color: errorGgmlv3Quantization ? "#d8342c" : "inherit",
                  }}
                />
              </Box>
            ))}
          </Box>
        </FormGroup>
      )}

      <Box width={"100%"} m="20px">
        <Button
          variant="contained"
          color="primary"
          type="submit"
          onClick={() => {
            fetch(endPoint + "/v1/model_registrations/LLM", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                model: JSON.stringify(formData),
                persist: false,
              }),
            })
              .then((response) => {
                return response.json();
              })
              .then((data) => console.log(data))
              .catch((error) =>
                console.error(
                  "There was a problem with the fetch operation:",
                  error
                )
              );
          }}
        >
          Register Model
        </Button>
      </Box>

      {/* Debug: Display form data */}
      <pre>{JSON.stringify(formData, null, 2)}</pre>
    </Box>
  );
};

export default RegisterModel;

const styles = {
  baseFormControl: {
    width: "100%",
    margin: "normal",
    size: "small",
  },
  checkboxWrapper: {
    display: "flex",
    flexWrap: "wrap",
    maxWidth: "80%",
  },
  checkboxError: {
    color: "#d8342c",
    "&.Mui-checked": {
      color: "#d8342c",
    },
  },
  labelPaddingLeft: {
    paddingLeft: 5,
  },
  formControlLabelPaddingLeft: {
    paddingLeft: 10,
  },
  buttonBox: {
    width: "100%",
    margin: "20px",
  },
};
