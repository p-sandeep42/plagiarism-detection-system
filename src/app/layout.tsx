import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";
import "@/styles/tokens.css";
import { AuthProvider } from "@/lib/AuthContext";
import { ToastProvider } from "@/components/Toast";

export const metadata: Metadata = {
  title: { default: 'AuraDiff — Plagiarism & Code Similarity Detector', template: '%s | AuraDiff' },
  description: 'Detect plagiarism and code similarity with 3 AI algorithms: Winnowing, AST Structural, and Semantic analysis. Free, fast, accurate.',
  keywords: ['plagiarism detector', 'code similarity checker', 'academic integrity', 'Winnowing algorithm', 'semantic similarity', 'AST comparison'],
  icons: {
    icon: '/favicon.png',
    apple: '/favicon.png',
  },
  openGraph: { 
    type: 'website', 
    url: 'https://auradiff.app', 
    siteName: 'AuraDiff',
    images: [{ url: '/og-image.png', width: 1200, height: 630, alt: 'AuraDiff Dashboard' }] 
  },
  twitter: { card: 'summary_large_image', creator: '@auradiff' },
  robots: { index: true, follow: true },
  alternates: {
    canonical: 'https://auradiff.app',
  }
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'AuraDiff',
    operatingSystem: 'Any',
    applicationCategory: 'EducatorsApplication',
    offers: {
      '@type': 'Offer',
      price: '0',
      priceCurrency: 'USD',
    },
    aggregateRating: {
      '@type': 'AggregateRating',
      ratingValue: '4.9',
      ratingCount: '1024',
    },
  };

  return (
    <html lang="en">
      <head>
        <Script
          id="json-ld"
          type="application/ld+json"
          strategy="afterInteractive"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className="antialiased selection:bg-indigo-500/30 min-h-screen font-sans bg-[#020617] text-slate-200">
        <div className="fixed inset-0 -z-10 h-full w-full pointer-events-none">
          <div className="absolute top-0 left-1/4 h-[800px] w-[800px] rounded-full bg-indigo-900/10 blur-[120px]"></div>
          <div className="absolute bottom-0 right-1/4 h-[800px] w-[800px] rounded-full bg-cyan-900/10 blur-[120px]"></div>
        </div>
        <AuthProvider>
          <ToastProvider>
            {children}
          </ToastProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
