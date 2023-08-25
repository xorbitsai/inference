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
        <ModelCard
          imgURL={require("../../media/logo_wizardlm.webp")}
          serviceName="WizardLM-v1.0"
          description="WizardLM is an open-source chatbot trained by fine-tuning LLaMA using the innovative Evol-Instruct method."
          url={endPoint}
          jsonData={{
            model_name: "wizardlm-v1.0",
            model_size_in_billions: 7,
          }}
        />
        <ModelCard
          imgURL={require("../../media/logo_vicuna.webp")}
          serviceName="Vicuna-v1.3"
          description="Vicuna is an open-source chatbot trained by fine-tuning LLaMA on data collected from ShareGPT. "
          url={endPoint}
          jsonData={{
            model_name: "vicuna-v1.3",
            model_size_in_billions: 7,
          }}
        />
        {registrationData.map((registration, index) => (
          <ModelCard
            imgURL={require("../../media/logo_vicuna.webp")}
            serviceName={registration.model_name}
            description={registration.model_description}
            url={endPoint}
            jsonData={{
              model_name: registration.model_name,
            }} // or any other properties
          />
        ))}
      </div>
    </Box>
  );
};

export default LaunchModel;
