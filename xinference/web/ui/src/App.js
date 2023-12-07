import { CssBaseline, ThemeProvider } from '@mui/material'
import { HashRouter, Route, Routes } from 'react-router-dom'

import { ApiContextProvider } from './components/apiContext'
import Layout from './scenes/_layout'
import LaunchModel from './scenes/launch_model'
import RegisterModel from './scenes/register_model'
import RunningModels from './scenes/running_models'
import { useMode } from './theme'

function App() {
  const [theme] = useMode()
  return (
    <div className="app">
      <HashRouter>
        <ThemeProvider theme={theme}>
          <ApiContextProvider>
            <CssBaseline />
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<LaunchModel />} />
                <Route path="/running_models" element={<RunningModels />} />
                <Route path="/register_model" element={<RegisterModel />} />
              </Route>
            </Routes>
          </ApiContextProvider>
        </ThemeProvider>
      </HashRouter>
    </div>
  )
}

export default App
