/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for Docker deployment
  output: 'standalone',
  experimental: {
    optimizeCss: false,
  },
  // 确保CSS正确处理
  compiler: {
    removeConsole: false,
  },
  // 解决开发模式错误
  reactStrictMode: true,
  // 开发服务器配置
  devIndicators: {
    position: 'bottom-right',
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  eslint: {
    ignoreDuringBuilds: false,
  },
  async redirects() {
    return [
      {
        source: '/',
        destination: '/launch-model',
        permanent: true, // false = 307 (temporary redirect), true = 301 (permanent redirect)
      },
    ];
  },
  webpack(config: any) {
    const fileLoaderRule = config.module.rules.find(
      (rule: any) =>
        rule.test?.test?.('.svg')
    )

    if (fileLoaderRule) {
      fileLoaderRule.exclude = /\.svg$/i
    }

    config.module.rules.push({
      test: /\.svg$/i,
      issuer: /\.[jt]sx?$/,
      use: ['@svgr/webpack'],
    })

    return config
  },
};

export default nextConfig;
