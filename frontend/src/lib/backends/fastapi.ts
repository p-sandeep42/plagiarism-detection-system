import { BackendService } from '../backend';
import { ComparisonResponse, BatchComparisonResponse } from '@/types/backend';

export class FastAPIBackend implements BackendService {
  async compare(fileA: File, fileB: File): Promise<ComparisonResponse> {
    const fd = new FormData();
    fd.append('file1', fileA); // match FastAPI parameter names
    fd.append('file2', fileB);
    
    // Note: This would typically go through /api/compare proxy
    const res = await fetch('/fastapi/compare', {
      method: 'POST',
      body: fd,
    });
    
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async compareBatch(files: File[], sessionId?: string): Promise<BatchComparisonResponse> {
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    if (sessionId) fd.append('session_id', sessionId);
    
    const res = await fetch('/api/compare-batch', {
      method: 'POST',
      body: fd,
    });
    
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
}
