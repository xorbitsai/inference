import { CssBaseline, ThemeProvider } from "@mui/material";
import { HashRouter, Route, Routes } from "react-router-dom";
import { useMode, ApiContextProvider, LanguageContextProvider } from "./theme";
import Layout from "./scenes/_layout";

import RunningModels from "./scenes/running_models";
import LaunchModel from "./scenes/launch_model";
import RegisterModel from "./scenes/register_model";

function App() {
  const [theme] = useMode();
  return (
    <div className="app">
      <HashRouter>
        <ThemeProvider theme={theme}>
          <LanguageContextProvider>
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
          </LanguageContextProvider>
        </ThemeProvider>
      </HashRouter>
    </div>
  );
}

export default App;
