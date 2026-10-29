/** Ingestion and vector-index lifecycle API operations. */
export function createIngestionService(apiRequest) {
  return {
    indexPending(workspaceId) {
      return apiRequest(`/api/v1/ingestion/workspaces/${encodeURIComponent(workspaceId)}/index-pending`, { method: 'POST' });
    },
    status(documentId) {
      return apiRequest(`/api/v1/ingestion/documents/${encodeURIComponent(documentId)}/status`);
    },
    indexDocument(documentId) {
      return apiRequest(`/api/v1/ingestion/documents/${encodeURIComponent(documentId)}/index`, { method: 'POST' });
    },
    processDocument(documentId) {
      return apiRequest(`/api/v1/ingestion/documents/${encodeURIComponent(documentId)}/process?force=true`, { method: 'POST' });
    },
  };
}
