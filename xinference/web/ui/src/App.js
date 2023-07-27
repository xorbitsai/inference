import { CssBaseline, ThemeProvider } from "@mui/material";
import { ColorModeContext, useMode } from "./theme";
import { HashRouter, Route, Routes } from "react-router-dom";
import Layout from "./scenes/_layout";

import ContactUs from "./scenes/contact_us";
import LaunchModel from "./scenes/launch_model";

import ModelDashboard from "./scenes/model_dashboard";

function App() {
  const [theme, colorMode] = useMode();
  return (
    <div className="app">
      <HashRouter>
        <ColorModeContext.Provider value={colorMode}>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<LaunchModel />} />
                <Route path="/contact_us" element={<ContactUs />} />
                <Route path="/model_dashboard" element={<ModelDashboard />} />
              </Route>
            </Routes>
          </ThemeProvider>
        </ColorModeContext.Provider>
      </HashRouter>
    </div>
  );
}

export default App;
