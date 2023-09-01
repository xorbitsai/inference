import { CssBaseline, ThemeProvider } from "@mui/material";
import { HashRouter, Route, Routes } from "react-router-dom";
import { ApiContextProvider } from "./components/apiContext";
import { useMode } from "./theme";
import Layout from "./scenes/_layout";

import RunningModels from "./scenes/running_models";
import LaunchModel from "./scenes/launch_model";

function App() {
  const [theme] = useMode();
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
              </Route>
            </Routes>
          </ApiContextProvider>
        </ThemeProvider>
      </HashRouter>
    </div>
  );
}

export default App;
