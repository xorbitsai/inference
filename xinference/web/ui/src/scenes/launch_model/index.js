import React from "react";
import ModelCard from "./model_card";
import { LaunchingModelProvider } from "./launching_context";

const LaunchModel = () => {
  const jsonData = {
    endpoint: "http://localhost:9997",
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
    <LaunchingModelProvider>
      <div style={style}>
        <ModelCard
          imgURL="https://t4.ftcdn.net/jpg/04/24/72/21/360_F_424722112_grNCAEGRwk5bYm2SDYVbBnW6VzU8qmKN.jpg"
          serviceName="WizardLM-v1.0"
          description="WizardLM is an open-source chatbot trained by fine-tuning LLaMA using the innovative Evol-Instruct method."
          url="http://localhost:9997"
          jsonData={{
            model_name: "wizardlm-v1.0",
            model_size_in_billions: 7,
          }}
        />
        <ModelCard
          imgURL="https://lmsys.org/images/blog/vicuna/vicuna.jpeg"
          serviceName="Vicuna-v1.3"
          description="Vicuna is an open-source chatbot trained by fine-tuning LLaMA on data collected from ShareGPT. "
          url="http://localhost:9997"
          jsonData={{
            model_name: "vicuna-v1.3",
            model_size_in_billions: 7,
          }}
        />
        {/* <ModelCard
          imgURL="https://www.zhipuai.cn/assets/images/home/chatglm/chatglm_logo1.png"
          serviceName="ChatGLM"
          description="ChatGLM is an open bilingual language model based on General Language Model (GLM) framework."
          url="http://localhost:9997"
          jsonData={{
            model_name: "chatglm",
            model_size_in_billions: 6,
          }}
        />
        <ModelCard
          imgURL="https://www.zhipuai.cn/assets/images/home/chatglm/chatglm_logo1.png"
          serviceName="ChatGLM2"
          description="ChatGLM2 is the second-generation version of the open-source bilingual (Chinese-English) chat model ChatGLM-6B."
          url="http://localhost:9997"
          jsonData={{
            model_name: "chatglm2",
            model_size_in_billions: 6,
          }}
        /> */}
        {[...Array(20)].map((_, index) => (
          <ModelCard
            imgURL="https://api.time.com/wp-content/uploads/2022/11/GettyImages-1358149692.jpg"
            serviceName="Placeholder"
            description="This is a one-sentence description of the model, suitable for models based on text, image, or audio."
            url="http://localhost:9997"
            jsonData={jsonData}
          />
        ))}
      </div>
    </LaunchingModelProvider>
  );
};

export default LaunchModel;
