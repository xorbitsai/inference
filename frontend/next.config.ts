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
        destination: '/workbench',
        permanent: true, // false = 307 (临时重定向), true = 301 (永久重定向)
      },
    ];
  },
};

export default nextConfig;
