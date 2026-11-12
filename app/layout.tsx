import type { Metadata } from "next";
import { ServerNotice, ServerStatusProvider } from "@/components/server-status";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "RVC Studio",
    template: "%s | RVC Studio",
  },
  description:
    "声のモデルを作って、音声をその声に変換するアプリ。処理はすべてこの PC（Windows・Mac）の中で行います。",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ja" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <ServerStatusProvider>
          <SiteHeader />
          <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
            <ServerNotice />
            {children}
          </main>
        </ServerStatusProvider>
      </body>
    </html>
  );
}
