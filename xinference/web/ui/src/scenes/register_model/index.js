import React, { useState, useContext } from "react";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import { ApiContext } from "../../components/apiContext";
import {
  Box,
  Checkbox,
  FormGroup,
  FormControlLabel,
  Radio,
  RadioGroup,
  FormControl,
} from "@mui/material";
import { useMode } from "../../theme";

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
  const ERROR_COLOR = useMode();
  const endPoint = useContext(ApiContext).endPoint;
  const [persist, setPersist] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [formData, setFormData] = useState({
    version: 1,
    context_length: 2048,
    model_name: "custom-llama-2",
    model_lang: ["en"],
    model_ability: ["generate"],
    model_description: "This is a custom model description.",
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

  // model name must be
  // 1. all alphanueric characters, dashes, or periods
  // 2. does not contain two consecutive dashes or periods
  // 3. does not begin nor end with dashes or periods
  // 4. between 1 and 255 characters long
  const errorModelName =
    !/^(?!-|\.)(?!.*--)(?!.*\.\.)(?!.*\.-)(?!.*-\.)[A-Za-z0-9-.]{1,255}(?<!-|\.)$/.test(
      formData.model_name
    );
  const errorModelDescription =
    !/^[A-Za-z0-9\s!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]{0,500}$/.test(
      formData.model_description
    );
  const errorContextLength = formData.context_length === 0;
  const errorLanguage =
    formData.model_lang === undefined || formData.model_lang.length === 0;
  const errorAbility =
    formData.model_ability === undefined || formData.model_ability.length === 0;
  const errorModelFormat =
    formData.model_specs === undefined || formData.model_specs.length === 0;
  const errorPytorchQuantization =
    formData.model_specs &&
    formData.model_specs.some((spec) => {
      return (
        spec.model_format === "pytorch" &&
        (spec.quantizations === undefined || spec.quantizations.length === 0)
      );
    });
  const errorPytorchModelSize =
    formData.model_specs &&
    formData.model_specs.some((spec) => {
      return (
        spec.model_format === "pytorch" &&
        (spec.model_size_in_billions === undefined ||
          spec.model_size_in_billions === 0)
      );
    });
  const errorGgmlv3Quantization =
    formData.model_specs &&
    formData.model_specs.some((spec) => {
      return (
        spec.model_format === "ggmlv3" &&
        (spec.quantizations === undefined || spec.quantizations.length === 0)
      );
    });
  const errorGgmlv3ModelSize =
    formData.model_specs &&
    formData.model_specs.some((spec) => {
      return (
        spec.model_format === "ggmlv3" &&
        (spec.model_size_in_billions === undefined ||
          spec.model_size_in_billions === 0)
      );
    });
  const errorAny =
    errorModelName ||
    errorModelDescription ||
    errorContextLength ||
    errorLanguage ||
    errorAbility ||
    errorModelFormat ||
    errorPytorchQuantization ||
    errorPytorchModelSize ||
    errorGgmlv3Quantization ||
    errorGgmlv3ModelSize;

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
            if (!/^\d+$/.test(value) || parseInt(value) < 0) {
              value = "0";
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
        <Box padding="10px"></Box>
        <TextField
          label="Model Description"
          error={errorModelDescription}
          defaultValue={formData.model_description}
          size="small"
          helperText="A short, 1-2 sentence description of your custom model"
          onChange={(event) =>
            setFormData({ ...formData, model_description: event.target.value })
          }
        />
        <Box padding="5px"></Box>

        {/* Model Languages Checkboxes */}
        <label
          style={{
            paddingLeft: 5,
            color: errorLanguage ? ERROR_COLOR : "inherit",
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
                    sx={
                      errorLanguage
                        ? {
                            color: ERROR_COLOR,
                            "&.Mui-checked": {
                              color: ERROR_COLOR,
                            },
                          }
                        : {}
                    }
                  />
                }
                label={SUPPORTED_LANGUAGES_DICT[lang]}
                style={{
                  paddingLeft: 10,
                  color: errorLanguage ? ERROR_COLOR : "inherit",
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
            color: errorAbility ? ERROR_COLOR : "inherit",
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
                    sx={
                      errorAbility
                        ? {
                            color: ERROR_COLOR,
                            "&.Mui-checked": {
                              color: ERROR_COLOR,
                            },
                          }
                        : {}
                    }
                  />
                }
                label={ability}
                style={{
                  paddingLeft: 10,
                  color: errorAbility ? ERROR_COLOR : "inherit",
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
            color: errorModelFormat ? ERROR_COLOR : "inherit",
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
                          color: ERROR_COLOR,
                          "&.Mui-checked": {
                            color: ERROR_COLOR,
                          },
                        }
                      : {}
                  }
                />
              }
              label="pytorch"
              style={{
                paddingLeft: 10,
                color: errorModelFormat ? ERROR_COLOR : "inherit",
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
                  sx={
                    errorModelFormat
                      ? {
                          color: ERROR_COLOR,
                          "&.Mui-checked": {
                            color: ERROR_COLOR,
                          },
                        }
                      : {}
                  }
                />
              }
              label="ggmlv3"
              style={{
                paddingLeft: 10,
                color: errorModelFormat ? ERROR_COLOR : "inherit",
              }}
            />
          </Box>
        </Box>
      </FormControl>
      <Box padding="5px"></Box>

      <label
        style={{
          paddingLeft: 5,
        }}
      >
        Persist Model
      </label>

      <RadioGroup
        value={persist ? "acrossSessions" : "thisSessionOnly"}
        onChange={() => setPersist(!persist)}
      >
        <Box sx={styles.checkboxWrapper}>
          <Box sx={{ marginLeft: "10px" }}>
            <FormControlLabel
              value="thisSessionOnly"
              control={<Radio />}
              label="Register this model in this session only"
            />
          </Box>
          <Box sx={{ marginLeft: "10px" }}>
            <FormControlLabel
              value="acrossSessions"
              control={<Radio />}
              label="Register this model across sessions"
            />
          </Box>
        </Box>
      </RadioGroup>
      <Box padding="5px"></Box>

      <label
        style={{
          paddingLeft: 5,
        }}
      >
        Show Raw
      </label>

      <RadioGroup
        value={showRaw ? "show" : "notShow"}
        onChange={() => setShowRaw(!showRaw)}
      >
        <Box sx={styles.checkboxWrapper}>
          <Box sx={{ marginLeft: "10px" }}>
            <FormControlLabel
              value="notShow"
              control={<Radio />}
              label="Only display interactible selections"
            />
          </Box>
          <Box sx={{ marginLeft: "10px" }}>
            <FormControlLabel
              value="show"
              control={<Radio />}
              label="Show raw json for API call"
            />
          </Box>
        </Box>
      </RadioGroup>

      {/* PyTorch Model Spec Form */}
      {pytorchSelected && (
        <FormGroup>
          <h2>PyTorch Model Spec</h2>
          <TextField
            label="Model Size in Billions"
            size="small"
            error={errorPytorchModelSize}
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
              if (!/^\d+$/.test(value) || parseInt(value) < 0) {
                value = "0";
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
              color: errorPytorchQuantization ? ERROR_COLOR : "inherit",
            }}
          >
            PyTorch Quantizations
          </label>
          <Box sx={styles.checkboxWrapper}>
            {PYTORCH_QUANTIZATIONS.map((quantization) => (
              <Box key={quantization} sx={{ marginLeft: "10px" }}>
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
                              color: ERROR_COLOR,
                              "&.Mui-checked": {
                                color: ERROR_COLOR,
                              },
                            }
                          : {}
                      }
                    />
                  }
                  label={quantization}
                  style={{
                    color: errorPytorchQuantization ? ERROR_COLOR : "inherit",
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
            size="small"
            error={errorGgmlv3ModelSize}
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
              if (!/^\d+$/.test(value) || parseInt(value) < 0) {
                value = "0";
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
              color: errorGgmlv3Quantization ? ERROR_COLOR : "inherit",
            }}
          >
            GGMLv3 Quantizations
          </label>
          <Box sx={styles.checkboxWrapper}>
            {GGMLV3_QUANTIZATIONS.map((quantization) => (
              <Box key={quantization} sx={{ marginLeft: "10px" }}>
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
                              color: ERROR_COLOR,
                              "&.Mui-checked": {
                                color: ERROR_COLOR,
                              },
                            }
                          : {}
                      }
                    />
                  }
                  label={quantization}
                  style={{
                    color: errorGgmlv3Quantization ? ERROR_COLOR : "inherit",
                  }}
                />
              </Box>
            ))}
          </Box>
        </FormGroup>
      )}

      <Box width={"100%"} m="20px">
        <div style={{ ...styles.error, color: ERROR_COLOR }}>
          {errorMessage}
        </div>
        <Button
          variant="contained"
          color="primary"
          type="submit"
          onClick={() => {
            if (errorAny) {
              setErrorMessage("Please fill in valid value for all fields");
              return;
            }
            fetch(endPoint + "/v1/model_registrations/LLM", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                model: JSON.stringify(formData),
                persist: persist,
              }),
            })
              .then((response) => {
                return response.json();
              })
              .then((data) => console.log(data))
              .catch((error) => {
                console.error(
                  "There was a problem with the fetch operation:",
                  error
                );
                setErrorMessage(
                  error.message || "An unexpected error occurred."
                );
              });
          }}
        >
          Register Model
        </Button>
      </Box>

      {/* Debug: Display form data */}
      {showRaw && (
        <pre style={{ padding: "30px" }}>
          {JSON.stringify(
            {
              model: formData,
              persist: persist,
            },
            null,
            2
          )}
        </pre>
      )}
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
  error: {
    fontWeight: "bold",
    margin: "5px 0",
    padding: "1px",
    borderRadius: "5px",
  },
};
