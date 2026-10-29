/** Session/workspace API operations. */
export function createWorkspaceService(apiRequest) {
  return {
    listSessions() {
      return apiRequest('/api/v1/sessions');
    },
    createWorkspace() {
      return apiRequest('/api/v1/workspaces', { method: 'POST', body: {} });
    },
  };
}
