import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server output for a small production Docker image.
  output: "standalone",
  // Pin the workspace root so Next doesn't pick up an unrelated lockfile higher up.
  turbopack: { root: path.resolve(__dirname) },
};

export default nextConfig;
