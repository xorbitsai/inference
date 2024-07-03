import React, { createContext, useEffect, useState } from 'react'

import { getEndpoint } from './utils'

export const ApiContext = createContext()

export const ApiContextProvider = ({ authority, children }) => {
  const [isCallingApi, setIsCallingApi] = useState(false)
  const [isUpdatingModel, setIsUpdatingModel] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const [auth, setAuth] = useState(authority)
  const endPoint = getEndpoint()

  useEffect(() => {
    setAuth(authority)
  }, [authority])

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
        auth,
      }}
    >
      {children}
    </ApiContext.Provider>
  )
}
