/** Versioned-document API operations. Browser file bytes still go to the presigned storage URL. */
export function createDocumentService(apiRequest) {
  return {
    listByWorkspace(workspaceId) {
      return apiRequest(`/api/v1/documents?workspace_id=${encodeURIComponent(workspaceId)}`);
    },
    createUploadIntent(payload) {
      return apiRequest('/api/v1/documents/upload-intents', { method: 'POST', body: payload });
    },
    completeUpload(intentId) {
      return apiRequest(`/api/v1/documents/upload-intents/${encodeURIComponent(intentId)}/complete`, { method: 'POST' });
    },
    downloadUrl(documentId) {
      return apiRequest(`/api/v1/documents/${encodeURIComponent(documentId)}/download-url`);
    },
    deleteSource(sourceId) {
      return apiRequest(`/api/v1/document-sources/${encodeURIComponent(sourceId)}`, { method: 'DELETE' });
    },
  };
}
