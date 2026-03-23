"use client";

import ErrorScreen from "@/components/ErrorScreen";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
      <ErrorScreen type="500" />
    </main>
  );
}
