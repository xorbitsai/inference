import { useContext, useEffect } from 'react'
import { Navigate, useRoutes } from 'react-router-dom'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../components/apiContext'
import Layout from '../scenes/_layout'
import ClusterInfo from '../scenes/cluster_info'
import LaunchModel from '../scenes/launch_model'
import Login from '../scenes/login/login'
import RegisterModel from '../scenes/register_model'
import RunningModels from '../scenes/running_models'

const LoginAuth = () => {
  const navigate = useNavigate()
  const { auth } = useContext(ApiContext)

  useEffect(() => {
    if (!auth) navigate('/launch_model/llm')
  }, [navigate])

  return <Login />
}

const routes = [
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        path: '/',
        element: <Navigate to="launch_model/llm" replace />,
      },
      {
        path: 'launch_model/:Modeltype/:subType?',
        element: <LaunchModel />,
      },
      {
        path: 'running_models/:runningModelType',
        element: <RunningModels />,
      },
      {
        path: 'register_model/:registerModelType/:model_name?',
        element: <RegisterModel />,
      },
      {
        path: 'cluster_info',
        element: <ClusterInfo />,
      },
    ],
  },
  {
    path: '/login',
    element: <LoginAuth />,
  },
  {
    path: '*',
    element: <Navigate to="launch_model/llm" replace />,
  },
]
const WraperRoutes = () => {
  let element = useRoutes(routes)
  return element
}

export default WraperRoutes
