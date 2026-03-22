import { BackendService } from '../backend';
import { ComparisonResponse, BatchComparisonResponse } from '@/types/backend';

export class FirebaseBackend implements BackendService {
  async compare(_fileA: File, _fileB: File): Promise<ComparisonResponse> {
    // To be implemented via Cloud Functions Callable
    throw new Error('FirebaseBackend.compare not yet implemented');
  }

  async compareBatch(_files: File[], _sessionId?: string): Promise<BatchComparisonResponse> {
    throw new Error('FirebaseBackend.compareBatch not yet implemented');
  }
}
