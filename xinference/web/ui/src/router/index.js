import { Navigate, useRoutes } from 'react-router-dom'

import Layout from '../scenes/_layout'
import ClusterInfo from '../scenes/cluster_info'
import LaunchModel from '../scenes/launch_model'
import Login from '../scenes/login/login'
import RegisterModel from '../scenes/register_model'
import RunningModels from '../scenes/running_models'

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
    element: <Login />,
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
