import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Brier · The decision layer for robot CI", template: "%s · Brier" },
  description: "Calibrated robot evaluations. Run the scenarios that matter and inspect the evidence behind every decision.",
};
export const runtime = "nodejs";

/** Apply shared typography, metadata, and an accessible content entry point. */
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>
    <a className="sr-only fixed top-3 left-3 z-50 bg-primary px-4 py-3 text-sm text-white focus:not-sr-only" href="#main">Skip to content</a>
    {children}
  </body></html>;
}
