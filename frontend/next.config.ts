// Output mode (build only):
//   'export'     -> static assets in out/, bundled into the Python wheel and
//                   served by the Xinference backend (single-process pip path).
//                   This is the default for builds.
//   'standalone' -> self-contained Node server (set NEXT_OUTPUT=standalone) for
//                   deployments that run the frontend as a separate service.
//
// Never set an output mode in `next dev`: 'export' forces dynamicParams=false,
// so navigating to a real dynamic route (/launch-model/llm, ...) whose params
// are not in generateStaticParams throws. In production the backend SPA
// fallback maps those paths to the __shell__ page; the dev server has no such
// layer.
const isDev = process.env.NODE_ENV === 'development';
const outputMode = isDev
  ? undefined
  : process.env.NEXT_OUTPUT === 'standalone'
    ? 'standalone'
    : 'export';

/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(outputMode ? { output: outputMode } : {}),
  images: { unoptimized: true },
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
  // rewrites/redirects are unsupported in static export; in the single-process
  // deployment the backend serves the UI same-origin so no proxy is needed.
  // Keep the API proxy for `next dev` and standalone builds.
  ...(outputMode !== 'export'
    ? {
        async rewrites() {
          const apiUrl = (
            process.env.XINFERENCE_API_URL ||
            process.env.NEXT_PUBLIC_API_URL ||
            'http://127.0.0.1:9997'
          ).replace(/\/+$/, '');
          return [
            {
              source: '/v1/:path*',
              destination: `${apiUrl}/v1/:path*`,
            },
            {
              source: '/token',
              destination: `${apiUrl}/token`,
            },
          ];
        },
      }
    : {}),
  webpack(config: any) {
    const fileLoaderRule = config.module.rules.find((rule: any) => rule.test?.test?.('.svg'));

    if (fileLoaderRule) {
      fileLoaderRule.exclude = /\.svg$/i;
    }

    config.module.rules.push({
      test: /\.svg$/i,
      issuer: /\.[jt]sx?$/,
      use: ['@svgr/webpack'],
    });

    return config;
  },
};

export default nextConfig;
