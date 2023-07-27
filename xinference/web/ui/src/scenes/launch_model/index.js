import React from "react";
import ModelCard from "./model_card";

const LaunchModel = () => {
  const fullUrl = window.location.href;
  let endPoint = "";
  if ("XINFERENCE_ENDPOINT" in process.env) {
    endPoint = process.env.XINFERENCE_ENDPOINT;
  } else {
    endPoint = fullUrl.split("/ui")[0];
  }

  const style = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
    paddingLeft: "2rem",
    gridGap: "2rem 0rem",
  };

  return (
    <div style={style}>
      <ModelCard
        imgURL="https://t4.ftcdn.net/jpg/04/24/72/21/360_F_424722112_grNCAEGRwk5bYm2SDYVbBnW6VzU8qmKN.jpg"
        serviceName="WizardLM-v1.0"
        description="WizardLM is an open-source chatbot trained by fine-tuning LLaMA using the innovative Evol-Instruct method."
        url={endPoint}
        jsonData={{
          model_name: "wizardlm-v1.0",
          model_size_in_billions: 7,
        }}
      />
      <ModelCard
        imgURL="https://lmsys.org/images/blog/vicuna/vicuna.jpeg"
        serviceName="Vicuna-v1.3"
        description="Vicuna is an open-source chatbot trained by fine-tuning LLaMA on data collected from ShareGPT. "
        url={endPoint}
        jsonData={{
          model_name: "vicuna-v1.3",
          model_size_in_billions: 7,
        }}
      />
    </div>
  );
};

export default LaunchModel;
