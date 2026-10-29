/** Workspace-scoped grounded RAG query API operation. */
export function createSearchService(apiRequest) {
  return {
    ask(payload) {
      return apiRequest('/api/v1/ask', { method: 'POST', body: payload });
    },
  };
}
