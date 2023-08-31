import React, { useContext, useState, useEffect } from "react";
import ModelCard from "./modelCard";
import Title from "../../components/Title";
import { Box } from "@mui/material";
import { ApiContext } from "../../components/apiContext";

const LaunchModel = () => {
  let endPoint = useContext(ApiContext).endPoint;
  const [registrationData, setRegistrationData] = useState([]);
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext);
  const { isUpdatingModel } = useContext(ApiContext);

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
          const detailResponse = await fetch(
            `${endPoint}/v1/model_registrations/LLM/${registration.model_name}`,
            {
              method: "GET",
            }
          );

          return await detailResponse.json();
        })
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

  return (
    <Box m="20px">
      <Title title="Launch Model" />
      <div style={style}>
        {registrationData.map((registration) => (
          <ModelCard url={endPoint} modelData={registration} />
        ))}
      </div>
    </Box>
  );
};

export default LaunchModel;
