import React, { createContext, useState } from "react";

export const ApiContext = createContext();

export const ApiContextProvider = ({ children }) => {
  const [isCallingApi, setIsCallingApi] = useState(false);
  const [isUpdatingModel, setIsUpdatingModel] = useState(false);
  let endPoint = "";
  if (!process.env.NODE_ENV || process.env.NODE_ENV === "development") {
    endPoint = "http://127.0.0.1:9997";
  } else {
    const fullUrl = window.location.href;
    endPoint = fullUrl.split("/ui")[0];
  }

  return (
    <ApiContext.Provider
      value={{
        isCallingApi,
        setIsCallingApi,
        isUpdatingModel,
        setIsUpdatingModel,
        endPoint,
      }}
    >
      {children}
    </ApiContext.Provider>
  );
};
