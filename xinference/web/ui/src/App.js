import { CssBaseline } from "@mui/material";
import { HashRouter, Route, Routes } from "react-router-dom";
import { ApiContextProvider } from "./components/apiContext";
import Layout from "./scenes/_layout";
import RunningModels from "./scenes/running_models";

function App() {
  return (
    <div className="app">
      <HashRouter>
        <ApiContextProvider>
          <CssBaseline />
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<RunningModels />} />
            </Route>
          </Routes>
        </ApiContextProvider>
      </HashRouter>
    </div>
  );
}

export default App;
