import { BackendService } from '../backend';
import { ComparisonResponse, BatchComparisonResponse } from '@/types/backend';

function getAuthHeaders(): Record<string, string> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('auradiff_token') : null;
  const headers: Record<string, string> = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

function handleAuthError(status: number) {
  if (status === 401 && typeof window !== 'undefined') {
    localStorage.removeItem('auradiff_token');
    window.location.href = '/login';
  }
}

export class FastAPIBackend implements BackendService {
  async compare(fileA: File, fileB: File): Promise<ComparisonResponse> {
    const fd = new FormData();
    fd.append('file1', fileA);
    fd.append('file2', fileB);
    
    const res = await fetch('/api/py/compare', {
      method: 'POST',
      body: fd,
      headers: getAuthHeaders(),
    });
    
    if (!res.ok) {
      handleAuthError(res.status);
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail || body?.error || 'Comparison failed');
    }
    return res.json();
  }

  async compareBatch(files: File[], sessionId?: string): Promise<BatchComparisonResponse> {
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    if (sessionId) fd.append('session_id', sessionId);
    
    const res = await fetch('/api/compare-batch', {
      method: 'POST',
      body: fd,
      headers: getAuthHeaders(),
    });
    
    if (!res.ok) {
      handleAuthError(res.status);
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail || body?.error || 'Batch comparison failed');
    }
    return res.json();
  }
}
