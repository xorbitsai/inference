import React, { createContext, useState } from "react";

export const LaunchingModelContext = createContext();

export const LaunchingModelProvider = ({ children }) => {
  const [isLaunchingModel, setIsLaunchingModel] = useState(false);

  return (
    <LaunchingModelContext.Provider
      value={{ isLaunchingModel, setIsLaunchingModel }}
    >
      {children}
    </LaunchingModelContext.Provider>
  );
};
