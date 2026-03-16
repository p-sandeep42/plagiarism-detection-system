import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_BACKEND: process.env.NEXT_PUBLIC_BACKEND ?? 'fastapi',
    NEXT_PUBLIC_FASTAPI_URL: process.env.NEXT_PUBLIC_FASTAPI_URL ?? 'http://localhost:8000',
    MAX_BATCH_FILES: process.env.MAX_BATCH_FILES ?? '30',
  },
  async rewrites() {
    return [
      {
        source: '/fastapi/:path*',
        destination: `${process.env.NEXT_PUBLIC_FASTAPI_URL}/:path*`
      },
    ];
  },
  images: {
    domains: ['localhost'],
  },
};

export default nextConfig;
