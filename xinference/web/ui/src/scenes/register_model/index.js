import React, { useState, useContext } from "react";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import { useMode, ApiContext } from "../../theme";
import {
  Box,
  Checkbox,
  FormHelperText,
  FormControlLabel,
  Radio,
  RadioGroup,
  FormControl,
} from "@mui/material";
import Title from "../../components/Title";

const SUPPORTED_LANGUAGES_DICT = { en: "English", zh: "Chinese" };
const SUPPORTED_FEATURES = ["Generate", "Chat"];

// Convert dictionary of supported languages into list
const SUPPORTED_LANGUAGES = Object.keys(SUPPORTED_LANGUAGES_DICT);

const RegisterModel = () => {
  const ERROR_COLOR = useMode();
  const { endPoint } = useContext(ApiContext);
  
  const [errorMessage, setErrorMessage] = useState("");
  const [modelFormat, setModelFormat] = useState("pytorch");
  const [modelSize, setModelSize] = useState(7);
  const [modelUri, setModelUri] = useState("/path/to/llama-2");
  const [formData, setFormData] = useState({
    version: 1,
    context_length: 2048,
    model_name: "custom-llama-2",
    model_lang: ["en"],
    model_ability: ["generate"],
    model_description: "This is a custom model description.",
    model_specs: [],
    prompt_style: undefined,
  });
  const [promptStyleLabel, setPromptStyleLabel] = useState("vicuna");
  const promptStyles = [
    {
      name: "vicuna",
      style_name: "ADD_COLON_TWO",
      system_prompt:
        "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
      roles: ["USER", "ASSISTANT"],
      intra_message_sep: " ",
      inter_message_sep: "</s>",
    },
    {
      name: "llama-2-chat",
      style_name: "LLAMA2",
      system_prompt:
        "<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n",
      roles: ["[INST]", "[/INST]"],
      intra_message_sep: " ",
      inter_message_sep: " </s><s>",
      stop_token_ids: [2],
    },
    {
      name: "falcon-instruct",
      style_name: "FALCON",
      system_prompt: "",
      roles: ["User", "Assistant"],
      intra_message_sep: "\n",
      inter_message_sep: "<|endoftext|>",
      stop: ["\nUser"],
      stop_token_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    {
      name: "baichuan-chat",
      style_name: "NO_COLON_TWO",
      system_prompt: "",
      roles: [" <reserved_102> ", " <reserved_103> "],
      intra_message_sep: "",
      inter_message_sep: "</s>",
      stop_token_ids: [2, 195],
    },
    {
      name: "baichuan-2-chat",
      style_name: "NO_COLON_TWO",
      system_prompt: "",
      roles: ["<reserved_106>", "<reserved_107>"],
      intra_message_sep: "",
      inter_message_sep: "</s>",
      stop_token_ids: [2, 195],
    },
    {
      name: "internlm-chat",
      style_name: "INTERNLM",
      system_prompt: "",
      roles: ["<|User|>", "<|Bot|>"],
      intra_message_sep: "<eoh>\n",
      inter_message_sep: "<eoa>\n",
      stop_token_ids: [1, 103028],
      stop: ["<eoa>"],
    },
  ];

  // model name must be
  // 1. Starts with an alphanumeric character (a letter or a digit).
  // 2. Followed by any number of alphanumeric characters, underscores (_), or hyphens (-).
  const errorModelName = !/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(
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
  const errorModelSize =
    formData.model_specs &&
    formData.model_specs.some((spec) => {
      return (
        spec.model_size_in_billions === undefined ||
        spec.model_size_in_billions === 0
      );
    });
  const errorAny =
    errorModelName ||
    errorModelDescription ||
    errorContextLength ||
    errorLanguage ||
    errorAbility ||
    errorModelSize;

  const isModelFormatPytorch = () => {
    return modelFormat === "pytorch";
  };

  const getPathComponents = (path) => {
    const normalizedPath = path.replace(/\\/g, "/");
    const baseDir = normalizedPath.substring(
      0,
      normalizedPath.lastIndexOf("/")
    );
    const filename = normalizedPath.substring(
      normalizedPath.lastIndexOf("/") + 1
    );
    return { baseDir, filename };
  };

  const handleClick = async () => {
    if (!isModelFormatPytorch()) {
      const { baseDir, filename } = getPathComponents(modelUri);
      formData.model_specs = [
        {
          model_format: modelFormat,
          model_size_in_billions: modelSize,
          quantizations: [""],
          model_id: "",
          model_file_name_template: filename,
          model_uri: baseDir,
        },
      ];
    } else {
      formData.model_specs = [
        {
          model_format: modelFormat,
          model_size_in_billions: modelSize,
          quantizations: ["4-bit", "8-bit", "none"],
          model_id: "",
          model_uri: modelUri,
        },
      ];
    }

    const ps = promptStyles.find((item) => item.name === promptStyleLabel);
    formData.prompt_style = {
      style_name: ps.style_name,
      system_prompt: ps.system_prompt,
      roles: ps.roles,
      intra_message_sep: ps.intra_message_sep,
      inter_message_sep: ps.inter_message_sep,
    };

    if (errorAny) {
      setErrorMessage("Please fill in valid value for all fields");
      return;
    }

    try {
      const response = await fetch(endPoint + "/v1/model_registrations/LLM", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: JSON.stringify(formData),
          persist: true,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json(); // Assuming the server returns error details in JSON format
        throw new Error(
          `Server error: ${response.status} - ${
            errorData.detail || "Unknown error"
          }`
        );
      }

      setErrorMessage(
        "Model has been registered successfully! Navigate to launch model page to proceed."
      );
    } catch (error) {
      console.error("There was a problem with the fetch operation:", error);
      setErrorMessage(error.message || "An unexpected error occurred.");
    }
  };

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

  return (
    <Box m="20px">
      <Title title="Register Model" />
      <Box padding="20px"></Box>

      {/* Base Information */}
      <FormControl sx={styles.baseFormControl}>
        <TextField
          label="Model Name"
          error={errorModelName}
          defaultValue={formData.model_name}
          size="small"
          helperText="Alphanumeric characters with properly placed hyphens and underscores. Must not match any built-in model names."
          onChange={(event) =>
            setFormData({ ...formData, model_name: event.target.value })
          }
        />
        <Box padding="15px"></Box>

        <label
          style={{
            paddingLeft: 5,
          }}
        >
          Model Format
        </label>

        <RadioGroup
          value={modelFormat}
          onChange={(e) => {
            setModelFormat(e.target.value);
          }}
        >
          <Box sx={styles.checkboxWrapper}>
            <Box sx={{ marginLeft: "10px" }}>
              <FormControlLabel
                value="pytorch"
                control={<Radio />}
                label="PyTorch"
              />
            </Box>
            <Box sx={{ marginLeft: "10px" }}>
              <FormControlLabel
                value="ggmlv3"
                control={<Radio />}
                label="GGML"
              />
            </Box>
            <Box sx={{ marginLeft: "10px" }}>
              <FormControlLabel
                value="ggufv2"
                control={<Radio />}
                label="GGUF"
              />
            </Box>
          </Box>
        </RadioGroup>
        <Box padding="15px"></Box>

        <TextField
          error={errorContextLength}
          label="Context Length"
          value={formData.context_length}
          size="small"
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
        <Box padding="15px"></Box>

        <TextField
          label="Model Size in Billions"
          size="small"
          error={errorModelSize}
          value={modelSize}
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
            setModelSize(Number(value));
          }}
        />
        <Box padding="15px"></Box>

        <TextField
          label="Model Path"
          size="small"
          value={modelUri}
          onChange={(e) => {
            setModelUri(e.target.value);
          }}
          helperText="For PyTorch, provide the model directory. For GGML/GGUF, provide the model file path."
        />
        <Box padding="15px"></Box>

        <TextField
          label="Model Description (Optional)"
          error={errorModelDescription}
          defaultValue={formData.model_description}
          size="small"
          onChange={(event) =>
            setFormData({ ...formData, model_description: event.target.value })
          }
        />
        <Box padding="15px"></Box>

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
        <Box padding="15px"></Box>

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
                    checked={formData.model_ability.includes(
                      ability.toLowerCase()
                    )}
                    onChange={() => toggleAbility(ability.toLowerCase())}
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
        <Box padding="15px"></Box>
      </FormControl>

      {formData.model_ability.includes("chat") && (
        <FormControl sx={styles.baseFormControl}>
          <label
            style={{
              paddingLeft: 5,
              color: errorAbility ? ERROR_COLOR : "inherit",
            }}
          >
            Prompt styles
          </label>
          <FormHelperText>
            Select a prompt style that aligns with the training data of your
            model.
          </FormHelperText>
          <RadioGroup
            value={promptStyleLabel}
            onChange={(e) => {
              setPromptStyleLabel(e.target.value);
            }}
          >
            <Box sx={styles.checkboxWrapper}>
              {promptStyles.map((p) => (
                <Box sx={{ marginLeft: "10px" }}>
                  <FormControlLabel
                    value={p.name}
                    control={<Radio />}
                    label={p.name}
                  />
                </Box>
              ))}
            </Box>
          </RadioGroup>
        </FormControl>
      )}

      <Box width={"100%"}>
        <div style={{ ...styles.error, color: ERROR_COLOR }}>
          {errorMessage}
        </div>
        <Button
          variant="contained"
          color="primary"
          type="submit"
          onClick={handleClick}
        >
          Register Model
        </Button>
      </Box>
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
