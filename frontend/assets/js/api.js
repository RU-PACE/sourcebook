/** Shared fetch client with cookie handling, bearer auth and one refresh retry. */
export function createApiClient({ baseUrl, state, demoApiRouter, onUnauthorized }) {
  function niceError(body, fallback) {
    if (body && typeof body.detail === 'string') return body.detail;
    if (body && typeof body.message === 'string') return body.message;
    if (Array.isArray(body?.detail)) return body.detail.map(x => x.msg || 'Invalid request').join('; ');
    if (body?.detail && typeof body.detail === 'object') {
      const message = body.detail.message || body.detail.error || '';
      const code = body.detail.code || '';
      return message && code ? `${message} (${code})` : (message || code || fallback);
    }
    return fallback;
  }

  async function rawRequest(path, options = {}) {
    const headers = new Headers(options.headers || {});
    let body = options.body;
    if (body && !(body instanceof FormData) && typeof body !== 'string' && !(body instanceof Blob)) {
      headers.set('Content-Type', 'application/json');
      body = JSON.stringify(body);
    }
    const response = await fetch(`${baseUrl}${path}`, { ...options, headers, body, credentials: 'include' });
    const responseText = await response.text();
    let payload = {};
    try { payload = responseText ? JSON.parse(responseText) : {}; }
    catch (_) { payload = { detail: responseText }; }
    if (!response.ok) {
      const error = new Error(niceError(payload, `Request failed (${response.status})`));
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  async function refreshAccessToken() {
    if (state.demoMode) return true;
    if (state.refreshInFlight) return state.refreshInFlight;
    state.refreshInFlight = (async () => {
      try {
        const payload = await rawRequest('/api/v1/auth/refresh', { method: 'POST' });
        state.accessToken = payload.access_token;
        state.tokenExpiresAt = Date.now() + Number(payload.expires_in || 1800) * 1000;
        state.user = payload.user || state.user;
        state.currentSessionId = payload.session?.id || state.currentSessionId;
        return true;
      } catch (_) {
        return false;
      } finally {
        state.refreshInFlight = null;
      }
    })();
    return state.refreshInFlight;
  }

  async function apiRequest(path, options = {}, retry = true) {
    if (state.demoMode) return demoApiRouter(path, options);
    const headers = new Headers(options.headers || {});
    if (state.accessToken) headers.set('Authorization', `Bearer ${state.accessToken}`);
    try {
      return await rawRequest(path, { ...options, headers });
    } catch (error) {
      if (error.status === 401 && retry) {
        const refreshed = await refreshAccessToken();
        if (refreshed) return apiRequest(path, options, false);
        if (typeof onUnauthorized === 'function') onUnauthorized();
      }
      throw error;
    }
  }

  return { rawRequest, apiRequest, refreshAccessToken };
}
