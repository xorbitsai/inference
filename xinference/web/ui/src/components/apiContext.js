import React, { createContext, useState } from "react";

export const ApiContext = createContext();

export const ApiContextProvider = ({ children }) => {
  const [isCallingApi, setIsCallingApi] = useState(false);
  const [isUpdatingModel, setIsUpdatingModel] = useState(false);

  return (
    <ApiContext.Provider
      value={{
        isCallingApi,
        setIsCallingApi,
        isUpdatingModel,
        setIsUpdatingModel,
      }}
    >
      {children}
    </ApiContext.Provider>
  );
};
