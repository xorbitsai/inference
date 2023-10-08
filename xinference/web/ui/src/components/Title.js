import { useContext } from "react";
import { Typography, Box } from "@mui/material";
import { LanguageContext } from "../theme";

const Title = ({ title, subtitle }) => {
  const { translation } = useContext(LanguageContext);
  return (
    <Box mb="30px">
      <Typography
        variant="h2"
        color="#141414"
        fontWeight="bold"
        sx={{ m: "0 0 5px 0" }}
      >
        {translation.page_title[title.toLowerCase().replace(" ", "_")]}
      </Typography>
      <Typography variant="h5" color="#3d3d3d">
        {subtitle}
      </Typography>
    </Box>
  );
};

export default Title;
