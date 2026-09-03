import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output only when building custom Docker image, Vercel uses native deployment.
  output: process.env.DOCKER_BUILD ? "standalone" : undefined,
  turbopack: { root: path.resolve(__dirname) },
};

export default nextConfig;
