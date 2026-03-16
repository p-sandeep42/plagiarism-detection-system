import { ComparisonResponse } from '@/types/backend';
import { BatchComparisonResponse } from '@/types/backend';

export interface BackendService {
  compare(fileA: File, fileB: File): Promise<ComparisonResponse>;
  compareBatch(files: File[], sessionId?: string): Promise<BatchComparisonResponse>;
}

export async function getBackend(): Promise<BackendService> {
  if (process.env.NEXT_PUBLIC_BACKEND === 'firebase') {
    const { FirebaseBackend } = await import('./backends/firebase');
    return new FirebaseBackend();
  }
  const { FastAPIBackend } = await import('./backends/fastapi');
  return new FastAPIBackend();
}
