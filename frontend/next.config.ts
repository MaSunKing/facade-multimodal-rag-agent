import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  basePath: process.env.STATIC_BASE || "",
  assetPrefix: process.env.STATIC_BASE || "",
};

export default nextConfig;
