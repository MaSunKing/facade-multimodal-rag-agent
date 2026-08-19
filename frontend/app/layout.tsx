import type { Metadata } from "next";
import "./globals.css";

const staticBasePath = (process.env.STATIC_BASE ?? "").replace(/\/$/, "");

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "建材知识助手",
  description: "面向建材销售与技术支持的本地多模态 RAG 助手。",
  manifest: `${staticBasePath}/manifest.webmanifest`,
  icons: {
    icon: `${staticBasePath}/favicon.svg`,
    shortcut: `${staticBasePath}/favicon.svg`,
  },
  openGraph: {
    title: "建材知识助手",
    description: "支持企业知识库、客户多格式文件与公开资料的可追溯问答。",
    images: [`${staticBasePath}/og-sales-assistant.png`],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
