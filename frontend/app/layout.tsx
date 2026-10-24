import type { Metadata, Viewport } from 'next'
import { Analytics } from '@vercel/analytics/next'
import { ThemeProvider } from '@/components/theme-provider'
import { SettingsProvider } from '@/lib/settings-context'
import { Toaster } from '@/components/ui/sonner'
import { RunHiveAuthGate } from '@/components/runhive-auth-gate'
import './globals.css'

export const metadata: Metadata = {
  title: 'RunOS',
  applicationName: 'RunOS',
  icons: { icon: '/runos-icon.svg', shortcut: '/runos-icon.svg' },
  description: '专业计算机视觉数据集管理平台 — 标注、筛选、转换、合并与训练',
  keywords: ['computer vision', 'dataset', 'annotation', 'yolo', 'coco', 'machine learning', 'deep learning', '计算机视觉', '数据集', '标注'],
}

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f5f8fc' },
    { media: '(prefers-color-scheme: dark)', color: '#f5f8fc' },
  ],
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="font-sans antialiased">
        {/* ThemeProvider must wrap SettingsProvider so useTheme() is available inside it */}
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          forcedTheme="light"
          enableSystem={false}
          disableTransitionOnChange
        >
          <RunHiveAuthGate>
            <SettingsProvider>
              {children}
              <Toaster richColors position="bottom-right" />
            </SettingsProvider>
          </RunHiveAuthGate>
        </ThemeProvider>
        <Analytics />
      </body>
    </html>
  )
}
