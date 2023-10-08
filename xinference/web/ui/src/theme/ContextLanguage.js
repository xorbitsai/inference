import english_translation from "./translation_setting_english";
import chinese_translation from "./translation_setting_chinese";

import React, { createContext, useState } from "react";

export const LanguageContext = createContext();

export const LanguageContextProvider = ({ children }) => {
  const [language, setLanguage] = useState("English");

  const translation_dict = {
    "English": english_translation,
    "中文（简体）": chinese_translation,
  };

  let translation = translation_dict[language];

  return (
    <LanguageContext.Provider
      value={{
        language,
        setLanguage,
        translation_dict,
        translation,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
};
