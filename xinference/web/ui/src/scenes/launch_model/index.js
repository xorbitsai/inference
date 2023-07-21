import React from "react";
import ModelCard from "./model_card";

const LaunchModel = () => {
  const jsonData = {
    model_name: "vicuna-v1.3",
    model_size_in_billions: 7,
  };

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
        serviceName="WizardLM-1.3b"
        description="WizardLM is an open-source chatbot trained by fine-tuning LLaMA using the innovative Evol-Instruct method."
        postURL="http://localhost:9997/v1/models"
        jsonData={{
          model_name: "wizardlm-v1.0",
          model_size_in_billions: 7,
        }}
      />
      <ModelCard
        imgURL="https://lmsys.org/images/blog/vicuna/vicuna.jpeg"
        serviceName="Vicuna-7b"
        description="Vicuna is an open-source chatbot trained by fine-tuning LLaMA on data collected from ShareGPT. "
        postURL="http://localhost:9997/v1/models"
        jsonData={{
          model_name: "vicuna-v1.3",
          model_size_in_billions: 7,
        }}
      />
      {[...Array(20)].map((_, index) => (
        <ModelCard
          imgURL="https://api.time.com/wp-content/uploads/2022/11/GettyImages-1358149692.jpg"
          serviceName="Placeholder"
          description="This is a one-sentence description of the model, suitable for models based on text, image, or audio."
          postURL="https://example.com/api/launch"
          jsonData={jsonData}
        />
      ))}
    </div>
  );
};

export default LaunchModel;
