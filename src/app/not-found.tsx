"use client";

import ErrorScreen from "@/components/ErrorScreen";

export default function NotFound() {
  return (
    <main className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
      <ErrorScreen type="404" />
    </main>
  );
}
