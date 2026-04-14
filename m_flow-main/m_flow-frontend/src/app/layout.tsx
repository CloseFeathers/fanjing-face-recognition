import type { Metadata } from "next";
import { ClientOnly } from "@/components/providers/ClientOnly";
import "./globals.css";

export const metadata: Metadata = {
  title: "M-Flow | Intelligent Knowledge Management",
  description: "Powerful knowledge management and retrieval system with multiple search modes and knowledge graph visualization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <ClientOnly>{children}</ClientOnly>
      </body>
    </html>
  );
}
