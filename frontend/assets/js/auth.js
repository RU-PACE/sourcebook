/** Authentication API operations; UI state transitions stay in app.js. */
export function createAuthService({ rawRequest, apiRequest }) {
  return {
    authenticate(mode, payload) {
      const action = mode === 'signup' ? 'signup' : 'login';
      return rawRequest(`/api/v1/auth/${action}`, { method: 'POST', body: payload });
    },
    logout() {
      return apiRequest('/api/v1/auth/logout', { method: 'POST' });
    },
  };
}
