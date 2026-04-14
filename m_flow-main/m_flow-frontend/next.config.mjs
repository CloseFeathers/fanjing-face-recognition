/**
 * M-flow Frontend — Next.js Configuration
 *
 * Proxies all /api/* requests to the M-flow backend (FastAPI)
 * running on port 8000 during local development.
 */

const BACKEND_URL = process.env.MFLOW_BACKEND_URL ?? 'http://localhost:8000';

/** @type {import('next').NextConfig} */
const mflowNextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  output: 'standalone',

  async rewrites() {
    return {
      fallback: [
        {
          source: '/api/:path*',
          destination: `${BACKEND_URL}/api/:path*`,
        },
      ],
    };
  },
};

export default mflowNextConfig;
