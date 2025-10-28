import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Bharat Foundation Model Framework - India's Sovereign AI",
  description: "India's first comprehensive open-source AI framework designed specifically for India's unique digital transformation needs. Built with sovereignty at its core.",
  keywords: ["BharatFM", "Foundation Model", "AI Framework", "Sovereign AI", "India AI", "Digital Transformation", "Open Source"],
  authors: [{ name: "Bharat AI Team" }],
  icons: {
    icon: [
      { url: "/favicon.png", sizes: "32x32", type: "image/png" },
      { url: "/logo.svg", sizes: "120x120", type: "image/svg+xml" }
    ],
  },
  openGraph: {
    title: "Bharat Foundation Model Framework",
    description: "India's sovereign AI framework powering digital transformation across critical domains",
    url: "https://bharat-ai.github.io/bharat-fm",
    siteName: "BharatFM",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Bharat Foundation Model Framework",
    description: "India's sovereign AI framework powering digital transformation",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
