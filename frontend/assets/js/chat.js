/** Durable, workspace-scoped chat history API operations. */
export function createChatService(apiRequest) {
  return {
    list(workspaceId) {
      return apiRequest(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/chat-history`);
    },
    append(workspaceId, message) {
      return apiRequest(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/chat-history`, {
        method: 'POST',
        body: {
          role: message.role,
          content: message.content,
          citations: message.citations || [],
          telemetry: message.telemetry || {},
        },
      });
    },
    clear(workspaceId) {
      return apiRequest(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/chat-history`, {
        method: 'DELETE',
      });
    },
  };
}
