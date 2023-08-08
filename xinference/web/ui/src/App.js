import { CssBaseline, ThemeProvider } from "@mui/material";
import { ColorModeContext, useMode } from "./theme";
import { HashRouter, Route, Routes } from "react-router-dom";
import { ApiContextProvider } from "./components/apiContext";
import Layout from "./scenes/_layout";

import ContactUs from "./scenes/contact_us";
import LaunchModel from "./scenes/launch_model";
import RunningModels from "./scenes/running_models";

function App() {
  const [theme, colorMode] = useMode();
  return (
    <div className="app">
      <HashRouter>
        <ColorModeContext.Provider value={colorMode}>
          <ThemeProvider theme={theme}>
            <ApiContextProvider>
              <CssBaseline />
              <Routes>
                <Route element={<Layout />}>
                  <Route path="/" element={<LaunchModel />} />
                  <Route path="/contact_us" element={<ContactUs />} />
                  <Route path="/running_models" element={<RunningModels />} />
                </Route>
              </Routes>
            </ApiContextProvider>
          </ThemeProvider>
        </ColorModeContext.Provider>
      </HashRouter>
    </div>
  );
}

export default App;
