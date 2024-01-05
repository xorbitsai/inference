import React, { createContext, useState } from 'react'

import { getEndpoint } from './utils'

export const ApiContext = createContext()

export const ApiContextProvider = ({ children }) => {
  const [isCallingApi, setIsCallingApi] = useState(false)
  const [isUpdatingModel, setIsUpdatingModel] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const endPoint = getEndpoint()

  return (
    <ApiContext.Provider
      value={{
        isCallingApi,
        setIsCallingApi,
        isUpdatingModel,
        setIsUpdatingModel,
        endPoint,
        errorMsg,
        setErrorMsg,
      }}
    >
      {children}
    </ApiContext.Provider>
  )
}
