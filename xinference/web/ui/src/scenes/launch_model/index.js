import React, { useContext, useState, useEffect } from "react";
import ModelCard from "./modelCard";
import Title from "../../components/Title";
import {
  Box,
  TextField,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Tabs, Tab
} from "@mui/material";
import { ApiContext } from "../../components/apiContext";

const LaunchModel = () => {
  let endPoint = useContext(ApiContext).endPoint;
  const [registrationData, setRegistrationData] = useState([]);
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext);
  const { isUpdatingModel } = useContext(ApiContext);

  // States used for filtering
  const [searchTerm, setSearchTerm] = useState("");

  const [modelAbility, setModelAbility] = useState("all");
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleAbilityChange = (event) => {
    setModelAbility(event.target.value);
  };

  const filter = (registration) => {
    if (!registration || typeof searchTerm !== "string") return false;
    const modelName = registration.model_name
      ? registration.model_name.toLowerCase()
      : "";
    const modelDescription = registration.model_description
      ? registration.model_description.toLowerCase()
      : "";

    if (
      !modelName.includes(searchTerm.toLowerCase()) &&
      !modelDescription.includes(searchTerm.toLowerCase())
    ) {
      return false;
    }
    if (modelAbility && modelAbility !== "all") {
      if (registration.model_ability.indexOf(modelAbility) < 0) {
        return false;
      }
    }
    return true;
  };

  const update = async () => {
    if (isCallingApi || isUpdatingModel) return;

    try {
      setIsCallingApi(true);

      const response = await fetch(`${endPoint}/v1/model_registrations/LLM`, {
        method: "GET",
      });

      const registrations = await response.json();

      const newRegistrationData = await Promise.all(
        registrations.map(async (registration) => {
          const desc = await fetch(
            `${endPoint}/v1/model_registrations/LLM/${registration.model_name}`,
            {
              method: "GET",
            },
          );

          return {
            ...(await desc.json()),
            is_builtin: registration.is_builtin,
          };
        }),
      );

      setRegistrationData(newRegistrationData);
      console.log(newRegistrationData);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsCallingApi(false);
    }
  };

  useEffect(() => {
    update();
    // eslint-disable-next-line
  }, []);

  const style = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
    paddingLeft: "2rem",
    gridGap: "2rem 0rem",
  };

  function a11yProps(index) {
    return {
      id: `simple-tab-${index}`,
      'aria-controls': `simple-tabpanel-${index}`,
    };
  }

  return (
    <Box m="20px">
      <Title title="Launch Model" />
      <Tabs value={tabValue} onChange={handleTabChange} aria-label="basic tabs example">
        <Tab label="LLMs" {...a11yProps(0)} />
        <Tab label="Embeddings" {...a11yProps(0)} />
      </Tabs>
      <div style={{display: "grid", gridTemplateColumns: "150px 1fr", columnGap: "20px", margin: "30px 2rem"}}>
        <FormControl
          variant="outlined"
          margin="normal"
        >
          <InputLabel id="ability-select-label">Model Ability</InputLabel>
          <Select
            id="ability"
            labelId="ability-select-label"
            label="Model Ability"
            onChange={handleAbilityChange}
            value={modelAbility}
            size="small"
            sx={{ width: "150px" }}
          >
            <MenuItem value="all">all</MenuItem>
            <MenuItem value="generate">generate</MenuItem>
            <MenuItem value="chat">chat</MenuItem>
          </Select>
        </FormControl>
        <FormControl
          variant="outlined"
          margin="normal"
        >
          <TextField
            id="search"
            type="search"
            label="Search for model name and description"
            value={searchTerm}
            onChange={handleChange}
            size="small"
          />
        </FormControl>
      </div>
      <div style={style}>
        {registrationData
          .filter((registration) => filter(registration))
          .map((filteredRegistration) => (
            <ModelCard url={endPoint} modelData={filteredRegistration} />
          ))}
      </div>
    </Box>
  );
};

export default LaunchModel;
