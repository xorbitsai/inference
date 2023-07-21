import { CssBaseline, ThemeProvider } from "@mui/material";
import { ColorModeContext, useMode } from "./theme";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./scenes/_layout";
import Home from "./scenes/home";
import ContactUs from "./scenes/contact_us";
import LaunchModel from "./scenes/launch_model";
import MachineSettings from "./scenes/machine_settings";
import ModelDashboard from "./scenes/model_dashboard";
import ResourceDashboard from "./scenes/resource_dashboard";
import WorkerDashboard from "./scenes/worker_dashboard";

function App() {
  const [theme, colorMode] = useMode();
  return (
    <div className="app">
      <BrowserRouter>
        <ColorModeContext.Provider value={colorMode}>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<Navigate to="/home" replace />} />
                <Route path="/home" element={<Home />} />
                <Route path="/contact_us" element={<ContactUs />} />
                <Route path="/launch_model" element={<LaunchModel />} />
                <Route path="/machine_settings" element={<MachineSettings />} />
                <Route path="/model_dashboard" element={<ModelDashboard />} />
                <Route path="/worker_dashboard" element={<WorkerDashboard />} />
                <Route
                  path="/resource_dashboard"
                  element={<ResourceDashboard />}
                />
              </Route>
            </Routes>
          </ThemeProvider>
        </ColorModeContext.Provider>
      </BrowserRouter>
    </div>
  );
}

export default App;
