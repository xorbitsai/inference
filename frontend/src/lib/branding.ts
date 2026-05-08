export interface BrandingConfig {
  appName: string
  logoPath: string
  logoAlt: string
  subtitle: string
  description: string
  tagline: string
  gradientFrom: string
  gradientVia: string
  gradientTo: string
}

export const defaultBranding: BrandingConfig = {
  appName: 'Xinference',
  logoPath: '/xagent_logo.svg', // todo
  logoAlt: 'Xinference',
  subtitle: 'Xorbits Inference (Xinference) is an open-source platform to streamline the operation and integration of a wide array of AI models. With Xinference, you’re empowered to run inference using any open-source LLMs, embedding models, and multimodal models either in the cloud or on your own premises, and create robust AI-driven applications',
  description: 'Xorbits Inference(Xinference) is a powerful and versatile library designed to serve language, speech recognition, and multimodal models. With Xorbits Inference, you can effortlessly deploy and serve your or state-of-the-art built-in models using just a single command. Whether you are a researcher, developer, or data scientist, Xorbits Inference empowers you to unleash the full potential of cutting-edge AI models.',
  tagline: 'Model Serving Made Easy',
  gradientFrom: 'blue-400',
  gradientVia: 'blue-500',
  gradientTo: 'indigo-500',
}

export function getBrandingFromEnv(): BrandingConfig {
  return {
    appName: process.env.NEXT_PUBLIC_APP_NAME || defaultBranding.appName,
    logoPath: process.env.NEXT_PUBLIC_LOGO_PATH || defaultBranding.logoPath,
    logoAlt: process.env.NEXT_PUBLIC_LOGO_ALT || defaultBranding.logoAlt,
    subtitle: process.env.NEXT_PUBLIC_APP_SUBTITLE || defaultBranding.subtitle,
    description: process.env.NEXT_PUBLIC_APP_DESCRIPTION || defaultBranding.description,
    tagline: process.env.NEXT_PUBLIC_APP_TAGLINE || defaultBranding.tagline,
    gradientFrom: process.env.NEXT_PUBLIC_GRADIENT_FROM || defaultBranding.gradientFrom,
    gradientVia: process.env.NEXT_PUBLIC_GRADIENT_VIA || defaultBranding.gradientVia,
    gradientTo: process.env.NEXT_PUBLIC_GRADIENT_TO || defaultBranding.gradientTo,
  }
}
