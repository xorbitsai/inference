import { createTheme } from "@mui/material/styles";
import typography_setting from "./typography_setting";
export { ApiContext, ApiContextProvider } from "./ContextApi";
export { LanguageContext, LanguageContextProvider } from "./ContextLanguage";

export const themeSettings = () => {
  return {
    ERROR_COLOR: "#d8342c",
    typography: typography_setting,
  };
};

export const useMode = () => {
  const theme = createTheme(themeSettings());
  return [theme];
};
