import { createApiClient } from './api.js';
import { createAuthService } from './auth.js';
import { createWorkspaceService } from './workspaces.js';
import { createDocumentService } from './documents.js';
import { createIngestionService } from './ingestion.js';
import { createSearchService } from './search.js';
import { createChatService } from './chat.js';

(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const API_BASE = (new URLSearchParams(location.search).get('api') || 'http://localhost:8000').replace(/\/$/, '');
  $('apiUrlLabel').textContent = API_BASE;

  const MAX_FILE_BYTES = 20 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = new Set(['pdf','docx','doc','txt','md','csv','tsv','xls','xlsx']);

  const state = {
    demoMode: false,
    authMode: 'login',
    accessToken: null,
    tokenExpiresAt: 0,
    user: null,
    sessions: [],
    currentSessionId: null,
    activeWorkspaceId: null,
    documents: [],
    expandedSessionIds: new Set(),
    uploading: false,
    busy: false,
    refreshInFlight: null,
    chatInFlight: false,
    chatHistoryByWorkspace: new Map(),
    demoChatHistoryByWorkspace: new Map(),
    pipelineWorkspaceId: null,
    pollTimer: null,
    confirmResolver: null
  };

  function lastWorkspaceStorageKey(userId = state.user?.id) {
    return userId ? `sourcebook:last-workspace:${userId}` : null;
  }

  function getStoredWorkspaceId(userId = state.user?.id) {
    const key = lastWorkspaceStorageKey(userId);
    if (!key) return null;
    try { return localStorage.getItem(key); } catch (_) { return null; }
  }

  function storeWorkspaceId(workspaceId) {
    const key = lastWorkspaceStorageKey();
    if (!key || !workspaceId) return;
    try { localStorage.setItem(key, workspaceId); } catch (_) { /* Storage can be disabled by the browser. */ }
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }

  function toast(message, kind = '') {
    const node = document.createElement('div');
    node.className = `toast ${kind}`;
    node.textContent = message;
    $('toastRegion').appendChild(node);
    setTimeout(() => node.remove(), 4200);
  }

  function askConfirm(title, message) {
    $('confirmTitle').textContent = title;
    $('confirmMessage').textContent = message;
    $('confirmModal').classList.remove('hidden');
    return new Promise(resolve => { state.confirmResolver = resolve; });
  }
  $('confirmCancelBtn').addEventListener('click', () => {
    $('confirmModal').classList.add('hidden');
    if (state.confirmResolver) state.confirmResolver(false);
    state.confirmResolver = null;
  });
  $('confirmOkBtn').addEventListener('click', () => {
    $('confirmModal').classList.add('hidden');
    if (state.confirmResolver) state.confirmResolver(true);
    state.confirmResolver = null;
  });

  function formatBytes(bytes) {
    const n = Number(bytes || 0);
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  }

  function extOf(filename) {
    const bits = String(filename || '').split('.');
    return bits.length > 1 ? bits.pop().toLowerCase() : '';
  }

  function fileTag(filename) {
    const ext = extOf(filename);
    return ({pdf:'PDF',docx:'DOCX',doc:'DOC',txt:'TXT',md:'MD',csv:'CSV',tsv:'TSV',xls:'XLS',xlsx:'XLSX'})[ext] || 'DOC';
  }

  const apiClient = createApiClient({
    baseUrl: API_BASE,
    state,
    demoApiRouter: (path, options) => demoApiRouter(path, options),
    onUnauthorized: () => showAuth('login'),
  });
  const { rawRequest, apiRequest, refreshAccessToken } = apiClient;
  const authApi = createAuthService(apiClient);
  const workspaceApi = createWorkspaceService(apiRequest);
  const documentApi = createDocumentService(apiRequest);
  const ingestionApi = createIngestionService(apiRequest);
  const searchApi = createSearchService(apiRequest);
  const chatApi = createChatService(apiRequest);

  /* PIPELINE 9-STAGE TRACKER */
  function setRagStep(num, mode, stateText) {
    const row = $(`ragStage${num}`);
    const numEl = $(`ragNum${num}`);
    const stEl = $(`ragState${num}`);
    if (!row) return;
    row.className = `rag-step ${mode || ''}`.trim();
    numEl.textContent = mode === 'done' ? '✓' : (mode === 'failed' || mode === 'quarantined' ? '!' : String(num));
    stEl.textContent = stateText || (mode === 'done' ? 'Done' : (mode === 'active' ? 'Running…' : 'Idle'));
  }

  function resetPipelineTracker(subtitle = 'Automatic Stage 1 → Stage 5 workflow') {
    $('pipelineTitle').textContent = 'End-to-End RAG Pipeline';
    $('pipelineOverall').textContent = 'Ready';
    $('pipelineSub').textContent = subtitle;
    $('pipelineProgressFill').style.width = '0%';
    $('pipelineDot').className = 'dot';
    for (let i = 1; i <= 9; i++) setRagStep(i, '', 'Idle');
  }

  function setPipelineUpTo(doneUpTo, activeStep = null, overallLabel = 'Processing', pct = 0, sub = '') {
    for (let i = 1; i <= 9; i++) {
      if (i <= doneUpTo) setRagStep(i, 'done', 'Done');
      else if (i === activeStep) setRagStep(i, 'active', 'Active…');
      else setRagStep(i, '', 'Waiting');
    }
    $('pipelineOverall').textContent = overallLabel;
    $('pipelineProgressFill').style.width = `${Math.max(0, Math.min(100, pct))}%`;
    if (sub) $('pipelineSub').textContent = sub;
  }

  function setPipelineUpToForWorkspace(workspaceId, ...args) {
    if (state.activeWorkspaceId === workspaceId) setPipelineUpTo(...args);
  }

  /* LEFT PANEL TABS */
  function switchLeftTab(tab) {
    const isSources = tab === 'sources';
    $('tabSourcesBtn').classList.toggle('active', isSources);
    $('tabSessionsBtn').classList.toggle('active', !isSources);
    $('leftSourcesPane').classList.toggle('hidden', !isSources);
    $('leftSessionsPane').classList.toggle('hidden', isSources);
  }
  $('tabSourcesBtn').addEventListener('click', () => switchLeftTab('sources'));
  $('tabSessionsBtn').addEventListener('click', () => switchLeftTab('sessions'));

  function initializePanelResizing() {
    const grid = document.querySelector('.studio-grid');
    const handles = [
      { node: $('leftResizeHandle'), side: 'left', min: 220 },
      { node: $('rightResizeHandle'), side: 'right', min: 230 },
    ];
    if (!grid || handles.some(handle => !handle.node)) return;

    const widthProperty = side => `--${side}-panel-width`;
    const readWidth = side => parseFloat(getComputedStyle(grid).getPropertyValue(widthProperty(side))) || (side === 'left' ? 280 : 300);
    const applyWidth = (side, value) => {
      grid.style.setProperty(widthProperty(side), `${Math.round(value)}px`);
    };
    const persistWidths = () => {
      try {
        localStorage.setItem('sourcebook:panel-widths', JSON.stringify({ left: readWidth('left'), right: readWidth('right') }));
      } catch (_) { /* Resizing still works when browser storage is unavailable. */ }
    };

    try {
      const saved = JSON.parse(localStorage.getItem('sourcebook:panel-widths') || '{}');
      if (Number.isFinite(Number(saved.left)) && Number(saved.left) >= 220) applyWidth('left', Number(saved.left));
      if (Number.isFinite(Number(saved.right)) && Number(saved.right) >= 230) applyWidth('right', Number(saved.right));
    } catch (_) { /* Ignore malformed or unavailable local settings. */ }
    try {
      const collapsed = JSON.parse(localStorage.getItem('sourcebook:panel-collapsed') || '{}');
      if (collapsed.left) grid.classList.add('left-collapsed');
      if (collapsed.right) grid.classList.add('right-collapsed');
    } catch (_) { /* Ignore malformed or unavailable local settings. */ }

    const persistCollapsed = () => {
      try {
        localStorage.setItem('sourcebook:panel-collapsed', JSON.stringify({
          left: grid.classList.contains('left-collapsed'),
          right: grid.classList.contains('right-collapsed'),
        }));
      } catch (_) { /* Panel collapse still works if browser storage is unavailable. */ }
    };

    const limitWidth = (side, value) => {
      const otherSide = side === 'left' ? 'right' : 'left';
      const innerWidth = grid.getBoundingClientRect().width - 24;
      const otherWidth = readWidth(otherSide);
      const handleAndGaps = 8 * 2 + 8 * 4;
      const maxWidth = Math.max(side === 'left' ? 220 : 230,
        Math.min(500, innerWidth - otherWidth - 320 - handleAndGaps));
      return Math.max(side === 'left' ? 220 : 230, Math.min(maxWidth, value));
    };

    function beginResize(side, event) {
      if (event.button !== 0 || window.matchMedia('(max-width: 860px)').matches) return;
      if (grid.classList.contains(`${side}-collapsed`)) return;
      event.preventDefault();
      const handle = handles.find(item => item.side === side)?.node;
      if (handle) handle.classList.add('dragging');
      document.body.classList.add('resizing-panels');
      const startX = event.clientX;
      const startWidth = readWidth(side);

      const move = moveEvent => {
        const delta = moveEvent.clientX - startX;
        const requested = side === 'left' ? startWidth + delta : startWidth - delta;
        applyWidth(side, limitWidth(side, requested));
      };
      const end = () => {
        document.removeEventListener('pointermove', move);
        document.removeEventListener('pointerup', end);
        document.removeEventListener('pointercancel', end);
        if (handle) handle.classList.remove('dragging');
        document.body.classList.remove('resizing-panels');
        persistWidths();
      };
      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', end);
      document.addEventListener('pointercancel', end);
    }

    handles.forEach(({ node, side, min }) => {
      node.addEventListener('pointerdown', event => beginResize(side, event));
      node.addEventListener('click', () => {
        if (!grid.classList.contains(`${side}-collapsed`)) return;
        grid.classList.remove(`${side}-collapsed`);
        persistCollapsed();
      });
      node.addEventListener('dblclick', event => {
        if (window.matchMedia('(max-width: 860px)').matches) return;
        event.preventDefault();
        if (grid.classList.contains(`${side}-collapsed`)) return;
        grid.classList.add(`${side}-collapsed`);
        persistCollapsed();
      });
      node.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        if (grid.classList.contains(`${side}-collapsed`)) grid.classList.remove(`${side}-collapsed`);
        const direction = event.key === 'ArrowRight' ? 1 : -1;
        const change = side === 'left' ? direction : -direction;
        applyWidth(side, limitWidth(side, readWidth(side) + change * 16));
        node.setAttribute('aria-valuemin', String(min));
        node.setAttribute('aria-valuenow', String(Math.round(readWidth(side))));
        persistWidths();
        persistCollapsed();
      });
    });
  }
  initializePanelResizing();

  $('addSourcesBtn').addEventListener('click', () => {
    switchLeftTab('sources');
    if (!$('fileInput').disabled) $('fileInput').click();
  });

  /* AUTH & SESSIONS */
  function showAuth(mode = 'login') {
    state.authMode = mode;
    $('authView').classList.remove('hidden');
    $('appView').classList.add('hidden');
    const signup = mode === 'signup';
    $('loginTab').classList.toggle('active', !signup);
    $('signupTab').classList.toggle('active', signup);
    $('usernameField').classList.toggle('hidden', !signup);
    $('username').required = signup;
    $('authTitle').textContent = signup ? 'Create your Studio account' : 'Welcome back';
    $('authSubmit').textContent = signup ? 'Create account ↗' : 'Log in ↗';
    $('authMessage').classList.add('hidden');
  }

  function showApp() {
    $('authView').classList.add('hidden');
    $('appView').classList.remove('hidden');
    $('userName').textContent = state.user?.username || 'Researcher';
    $('userEmail').textContent = state.user?.email || '';
    $('userAvatar').textContent = (state.user?.username || state.user?.email || 'U').slice(0, 1).toUpperCase();
  }

  $('loginTab').addEventListener('click', () => showAuth('login'));
  $('signupTab').addEventListener('click', () => showAuth('signup'));

  $('authForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (state.busy) return;
    const signup = state.authMode === 'signup';
    const payload = { email: $('email').value.trim(), password: $('password').value };
    if (signup) payload.username = $('username').value.trim();
    state.busy = true;
    $('authSubmit').disabled = true;
    $('authMessage').classList.add('hidden');
    try {
      const result = await authApi.authenticate(signup ? 'signup' : 'login', payload);
      state.demoMode = false;
      state.accessToken = result.access_token;
      state.tokenExpiresAt = Date.now() + Number(payload.expires_in || 1800) * 1000;
      state.user = result.user;
      state.currentSessionId = result.session?.id || null;
      state.expandedSessionIds = new Set(state.currentSessionId ? [state.currentSessionId] : []);
      const lastWorkspaceId = getStoredWorkspaceId(result.user?.id);
      state.activeWorkspaceId = lastWorkspaceId || result.workspace?.id || null;
      showApp();
      resetPipelineTracker();
      await loadSessions(lastWorkspaceId || result.workspace?.id || null);
      await selectWorkspace(state.activeWorkspaceId || firstWorkspaceId());
      checkModelsHealth(true);
      toast(signup ? 'Account ready. Upload a document to start.' : 'Signed in to Sourcebook Studio.');
    } catch (error) {
      $('authMessage').textContent = error.message || 'Authentication failed.';
      $('authMessage').classList.remove('hidden');
    } finally {
      state.busy = false;
      $('authSubmit').disabled = false;
    }
  });

  function allWorkspaces() {
    return state.sessions.flatMap(s => (s.workspaces || []).map(w => ({
      ...w,
      session_id: w.session_id || s.id,
      session_status: s.status,
      session_name: s.name
    })));
  }

  function activeWorkspace() {
    return allWorkspaces().find(w => w.id === state.activeWorkspaceId) || null;
  }

  function firstWorkspaceId() {
    const current = state.sessions.find(s => s.id === state.currentSessionId);
    return current?.workspaces?.[0]?.id || allWorkspaces()[0]?.id || null;
  }

  async function loadSessions(preferredWorkspaceId = null) {
    const payload = await workspaceApi.listSessions();
    state.sessions = payload.sessions || [];
    state.currentSessionId = payload.current_session_id || state.currentSessionId;
    if (state.currentSessionId) state.expandedSessionIds.add(state.currentSessionId);
    const list = allWorkspaces();
    const storedWorkspaceId = getStoredWorkspaceId();
    state.activeWorkspaceId = (preferredWorkspaceId && list.some(w => w.id === preferredWorkspaceId))
      ? preferredWorkspaceId
      : (storedWorkspaceId && list.some(w => w.id === storedWorkspaceId)
        ? storedWorkspaceId
        : (state.activeWorkspaceId && list.some(w => w.id === state.activeWorkspaceId) ? state.activeWorkspaceId : firstWorkspaceId()));
    renderSessionSidebar();
    renderWorkspaceHeader();
  }

  function renderSessionSidebar() {
    $('sessionCount').textContent = String(state.sessions.length);
    const markup = state.sessions.map(session => {
      const isCurrent = session.id === state.currentSessionId;
      const expanded = state.expandedSessionIds.has(session.id);
      const workspaces = session.workspaces || [];
      const wsHtml = workspaces.map(w => {
        const count = Number(w.document_count || 0);
        const active = w.id === state.activeWorkspaceId;
        return `<button type="button" class="workspace-nav-item ${active ? 'active' : ''}" data-ws="${escapeHtml(w.id)}">
          <span class="workspace-nav-icon">▤</span>
          <span style="min-width:0;flex:1">
            <span class="workspace-nav-name">${escapeHtml(w.name)}</span>
            <span class="workspace-nav-count">${count} ${count === 1 ? 'source' : 'sources'}</span>
          </span>
        </button>`;
      }).join('');
      return `<div class="session-group ${expanded ? '' : 'collapsed'}">
        <button type="button" class="session-heading ${isCurrent ? 'current' : ''}" data-sess="${escapeHtml(session.id)}">
          <span class="session-chevron">▾</span>
          <span style="min-width:0;flex:1">
            <span class="session-title">${escapeHtml(session.name)} ${isCurrent ? '<span class="current-pill">Active</span>' : ''}</span>
            <span class="session-meta">${workspaces.length} ${workspaces.length === 1 ? 'workspace' : 'workspaces'}</span>
          </span>
        </button>
        <div class="session-workspaces">${wsHtml}</div>
      </div>`;
    }).join('');
    $('sessionList').innerHTML = markup || '<div style="font-size:11px;color:var(--muted);padding:8px">No sessions yet.</div>';
    $('sessionList').querySelectorAll('[data-ws]').forEach(btn => btn.addEventListener('click', () => {
      selectWorkspace(btn.dataset.ws);
      switchLeftTab('sources');
    }));
    $('sessionList').querySelectorAll('[data-sess]').forEach(btn => btn.addEventListener('click', () => {
      const id = btn.dataset.sess;
      if (state.expandedSessionIds.has(id)) state.expandedSessionIds.delete(id);
      else state.expandedSessionIds.add(id);
      renderSessionSidebar();
    }));
  }

  function renderWorkspaceHeader() {
    const ws = activeWorkspace();
    const readOnly = ws && ws.session_id !== state.currentSessionId;
    $('workspaceTitle').textContent = ws?.name || 'Workspace';
    $('crumbSession').textContent = ws?.session_name || 'Session';
    $('crumbWorkspace').textContent = ws?.name || 'Workspace';
    $('readOnlyBadge').classList.toggle('hidden', !readOnly);
    $('dropZone').classList.toggle('read-only', !!readOnly);
    $('fileInput').disabled = !!readOnly;
    $('addSourcesBtn').disabled = !!readOnly || !ws;
  }

  async function selectWorkspace(workspaceId) {
    if (!workspaceId) return;
    state.activeWorkspaceId = workspaceId;
    storeWorkspaceId(workspaceId);
    state.documents = [];
    renderSessionSidebar();
    renderWorkspaceHeader();
    resetPipelineTracker('No RAG activity in this workspace yet');
    $('pipelineTelemetryBox').innerHTML = '<strong>No indexed sources</strong>';
    renderSourcesList();
    renderChatHistory();
    await Promise.all([loadWorkspaceChatHistory(workspaceId), loadDocuments(true)]);
  }

  function groupedSources() {
    const groups = new Map();
    for (const doc of state.documents) {
      if (!groups.has(doc.source_id)) groups.set(doc.source_id, []);
      groups.get(doc.source_id).push(doc);
    }
    return [...groups.entries()].map(([id, versions]) => {
      versions.sort((a, b) => Number(b.version_number || 0) - Number(a.version_number || 0));
      const current = versions.find(v => v.is_current && v.status === 'UPLOADED') || versions[0];
      return { id, source_name: current?.source_name || 'Document', versions, current };
    });
  }

  async function loadDocuments(autoCatchupIndex = false) {
    const ws = activeWorkspace();
    if (!ws) {
      state.documents = [];
      renderSourcesList();
      resetPipelineTracker('No RAG activity in this workspace yet');
      return;
    }
    const payload = await documentApi.listByWorkspace(ws.id);
    // Ignore a slower request if the user changed workspaces while it was in flight.
    if (state.activeWorkspaceId !== ws.id) return;
    state.documents = payload.documents || [];
    renderSourcesList();
    renderPipelineForWorkspace(ws.id);

    // If any SANITIZED document in this workspace has not been vector-indexed yet, auto-queue indexing silently!
    if (autoCatchupIndex && !state.demoMode) {
      const needsIndex = state.documents.some(d =>
        d.is_current &&
        d.ingestion_status === 'SANITIZED' &&
        ['NOT_INDEXED', 'FAILED'].includes(String(d.embedding_status || 'NOT_INDEXED').toUpperCase())
      );
      if (needsIndex) {
        try {
          await ingestionApi.indexPending(ws.id);
          startBackgroundStatusPoll();
        } catch (_) {}
      }
    }
  }

  function documentPipelineProgress(doc) {
    const ingestion = String(doc.ingestion_status || doc.run_status || doc.status || 'UPLOADED').toUpperCase();
    const embedding = String(doc.embedding_status || 'NOT_INDEXED').toUpperCase();
    if (ingestion === 'QUARANTINED') return { done: 3, failed: 4, label: 'Quarantined' };
    if (ingestion === 'FAILED') return { done: 2, failed: 3, label: 'Extraction failed' };
    if (embedding === 'FAILED') return { done: 6, failed: 7, label: 'Embedding failed' };
    if (embedding === 'INDEXED') return { done: 7, label: 'Indexed' };
    if (['QUEUED', 'EXTRACTING'].includes(ingestion)) return { done: 2, active: 3, label: 'Extracting' };
    if (ingestion === 'PII_SCANNING') return { done: 3, active: 4, label: 'Scanning for PII' };
    if (['NLP_ENRICHMENT', 'CHUNKING'].includes(ingestion)) return { done: 4, active: 5, label: 'Chunking' };
    if (ingestion === 'UPLOADING_ARTIFACT') return { done: 5, active: 6, label: 'Saving extraction' };
    if (ingestion === 'SANITIZED' && ['QUEUED', 'EMBEDDING'].includes(embedding)) return { done: 6, active: 7, label: 'Embedding' };
    if (ingestion === 'SANITIZED') return { done: 6, label: 'Awaiting embedding' };
    return { done: 2, active: 3, label: 'Queued for extraction' };
  }

  function renderPipelineForWorkspace(workspaceId) {
    if (state.activeWorkspaceId !== workspaceId) return;
    const files = groupedSources().map(group => group.current).filter(Boolean);
    if (!files.length) {
      resetPipelineTracker('No RAG activity in this workspace yet');
      $('pipelineTelemetryBox').innerHTML = '<strong>No indexed sources</strong>';
      return;
    }

    const progress = files.map(documentPipelineProgress);
    const indexedCount = progress.filter(item => item.done >= 7 && !item.failed).length;
    const failed = progress.find(item => item.failed);
    const leastProgress = progress.reduce((least, item) => item.done < least.done ? item : least, progress[0]);
    const percent = Math.round((indexedCount / files.length) * 78);

    if (failed) {
      setPipelineUpTo(failed.done, null, failed.label, Math.round((failed.done / 9) * 100), `${indexedCount}/${files.length} sources indexed`);
      setRagStep(failed.failed, 'failed', 'Failed');
    } else if (indexedCount === files.length) {
      setPipelineUpTo(7, null, 'Indexed', 78, `${indexedCount} ${indexedCount === 1 ? 'source' : 'sources'} indexed · ask a question to run Stage 4/5`);
      setRagStep(8, '', 'Awaiting query');
      setRagStep(9, '', 'Awaiting query');
    } else {
      setPipelineUpTo(leastProgress.done, leastProgress.active || null,
        leastProgress.label || 'Processing', percent || Math.round((leastProgress.done / 9) * 100),
        `${indexedCount}/${files.length} sources indexed · ${leastProgress.label}`);
      if (leastProgress.done === 6 && !leastProgress.active) setRagStep(7, '', 'Not started');
    }

    const totalChunks = files.reduce((sum, doc) => sum + Number(doc.embedded_chunk_count || 0), 0);
    $('pipelineTelemetryBox').innerHTML = `<strong>${indexedCount}/${files.length} sources indexed</strong><br>${totalChunks} embedded chunks`;
  }

  function renderSourcesList() {
    const groups = groupedSources();
    $('sourcesTabCount').textContent = String(groups.length);
    const indexedCount = groups.filter(g => String(g.current?.embedding_status || '').toUpperCase() === 'INDEXED').length;
    const totalChunks = groups.reduce((acc, g) => acc + Number(g.current?.embedded_chunk_count || 0), 0);
    $('indexedSourcesBadge').textContent = `${indexedCount}/${groups.length} sources indexed · ${totalChunks} chunks`;

    // Populate uploadMode dropdown
    const select = $('uploadMode');
    const keep = select.value || 'new';
    select.innerHTML = '<option value="new">Upload as: New document source</option>' +
      groups.map(g => `<option value="${escapeHtml(g.id)}">New version of: ${escapeHtml(g.source_name)} (v${Number(g.current?.version_number || 1)})</option>`).join('');
    if ([...select.options].some(o => o.value === keep)) select.value = keep;

    if (!groups.length) {
      $('documentList').innerHTML = `<div style="text-align:center;padding:24px 10px;color:var(--muted);font-size:11.5px;line-height:1.5">
        No sources in this workspace yet.<br>Drop a PDF, Word doc, or Excel sheet above to run the automatic pipeline.
      </div>`;
      return;
    }

    $('documentList').innerHTML = groups.map(group => {
      const cur = group.current;
      const ingStatus = String(cur?.ingestion_status || 'QUEUED').toUpperCase();
      const embStatus = String(cur?.embedding_status || 'NOT_INDEXED').toUpperCase();
      let badgeLabel = 'Processing…';
      let badgeClass = 'working';
      if (ingStatus === 'QUARANTINED') { badgeLabel = 'Quarantined'; badgeClass = 'quarantined'; }
      else if (ingStatus === 'FAILED' || embStatus === 'FAILED') { badgeLabel = 'Failed'; badgeClass = 'failed'; }
      else if (embStatus === 'INDEXED') { badgeLabel = `Ready · ${Number(cur?.embedded_chunk_count || 0)} chunks`; badgeClass = 'indexed'; }
      else if (ingStatus === 'SANITIZED' && ['QUEUED', 'EMBEDDING'].includes(embStatus)) { badgeLabel = 'Indexing vectors…'; badgeClass = 'working'; }

      return `<div class="source-item">
        <div class="source-top">
          <span class="source-badge-ext">${escapeHtml(fileTag(cur?.original_name))}</span>
          <div class="source-name">${escapeHtml(group.source_name)} <span style="color:var(--muted);font-weight:600">v${Number(cur?.version_number || 1)}</span></div>
          <span class="pill-state ${badgeClass}">${escapeHtml(badgeLabel)}</span>
        </div>
        <div class="source-sub">
          <span>${escapeHtml(cur?.original_name || '')}</span>
          <span>·</span>
          <span>${escapeHtml(formatBytes(cur?.file_size_bytes || 0))}</span>
          ${Number(cur?.pii_finding_count || 0) > 0 ? `<span>· ${Number(cur.pii_finding_count)} PII redacted</span>` : ''}
        </div>
        <div class="source-actions">
          <button type="button" class="mini-btn" data-dl="${escapeHtml(cur.id)}">Download</button>
          <button type="button" class="mini-btn" data-ver="${escapeHtml(group.id)}">＋ Version</button>
          ${(ingStatus === 'FAILED' || embStatus === 'FAILED') ? `<button type="button" class="mini-btn" data-retry="${escapeHtml(cur.id)}">↻ Retry</button>` : ''}
          <button type="button" class="mini-btn danger" data-del-src="${escapeHtml(group.id)}">Delete</button>
        </div>
      </div>`;
    }).join('');

    $('documentList').querySelectorAll('[data-dl]').forEach(b => b.addEventListener('click', () => downloadDoc(b.dataset.dl)));
    $('documentList').querySelectorAll('[data-ver]').forEach(b => b.addEventListener('click', () => {
      $('uploadMode').value = b.dataset.ver;
      $('fileInput').click();
    }));
    $('documentList').querySelectorAll('[data-retry]').forEach(b => b.addEventListener('click', () => retryPipeline(b.dataset.retry)));
    $('documentList').querySelectorAll('[data-del-src]').forEach(b => b.addEventListener('click', () => deleteSource(b.dataset.delSrc)));
  }

  /* AUTOMATIC UPLOAD -> STAGE 2 -> STAGE 3 PIPELINE */
  async function sha256Hex(file) {
    const buf = await file.arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-256', buf);
    return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function uploadFormToS3(url, fields, file, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      Object.entries(fields || {}).forEach(([k, v]) => form.append(k, v));
      form.append('file', file);
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);
      xhr.upload.onprogress = e => { if (e.lengthComputable && onProgress) onProgress((e.loaded / e.total) * 100); };
      xhr.onload = () => (xhr.status >= 200 && xhr.status < 300) ? resolve() : reject(new Error(`S3 upload failed (${xhr.status})`));
      xhr.onerror = () => reject(new Error('Network error during S3 upload.'));
      xhr.send(form);
    });
  }

  async function handleFiles(fileList) {
    const ws = activeWorkspace();
    if (!ws) { toast('Select a workspace first.', 'error'); return; }
    const files = [...fileList];
    if (!files.length) return;
    const sourceId = $('uploadMode').value !== 'new' ? $('uploadMode').value : null;

    for (const file of files) {
      const ext = extOf(file.name);
      if (!ALLOWED_EXTENSIONS.has(ext)) {
        toast(`Unsupported format .${ext}. Use PDF, DOCX, DOC, XLSX, XLS, CSV, TSV, MD or TXT.`, 'error');
        continue;
      }
      if (file.size <= 0 || file.size > MAX_FILE_BYTES) {
        toast(`${file.name} exceeds the 20 MiB limit.`, 'error');
        continue;
      }
      await runSingleFileUploadPipeline(ws.id, file, sourceId);
    }
    $('fileInput').value = '';
  }

  async function runSingleFileUploadPipeline(workspaceId, file, sourceId = null) {
    state.uploading = true;
    try {
      state.pipelineWorkspaceId = workspaceId;
      setPipelineUpToForWorkspace(workspaceId, 0, 1, 'Authorizing', 10, `Hashing & authorizing ${file.name}…`);
      const sha256 = await sha256Hex(file);
      const intent = await documentApi.createUploadIntent({ workspace_id: workspaceId, filename: file.name, file_size_bytes: file.size, sha256, source_id: sourceId });

      if (intent.duplicate) {
        // A duplicate response is not evidence that embeddings exist. Re-read this
        // workspace's persisted document/index state before updating the pipeline.
        if (state.activeWorkspaceId === workspaceId) await loadDocuments(false);
        toast(intent.message || 'Identical document already exists.');
        return;
      }

      setPipelineUpToForWorkspace(workspaceId, 1, 2, 'Uploading', 24, `Uploading ${file.name}…`);
      if (!state.demoMode) {
        await uploadFormToS3(intent.s3_upload.url, intent.s3_upload.fields, file, pct => {
          if (state.activeWorkspaceId === workspaceId) $('pipelineProgressFill').style.width = `${12 + Math.round(pct * 0.18)}%`;
        });
      }

      setPipelineUpToForWorkspace(workspaceId, 2, 3, 'Verifying', 35, `Verifying SHA-256 and saving metadata…`);
      const completed = await documentApi.completeUpload(intent.upload_intent_id);
      const docId = completed.document?.id || intent.document?.id;
      if (state.activeWorkspaceId === workspaceId) {
        await loadDocuments();
        await loadSessions(state.activeWorkspaceId);
      }
      if (docId) await pollDocumentEndToEnd(docId, file.name, workspaceId);
    } catch (error) {
      if (state.activeWorkspaceId === workspaceId) {
        $('pipelineOverall').textContent = 'Failed';
        $('pipelineDot').className = 'dot err';
      }
      toast(error.message || 'Upload pipeline failed.', 'error');
    } finally {
      state.uploading = false;
    }
  }

  async function pollDocumentEndToEnd(documentId, filename, workspaceId = state.activeWorkspaceId) {
    for (let attempt = 0; attempt < 180; attempt++) {
      const statusPayload = await ingestionApi.status(documentId);
      const run = statusPayload.run || {};
      const runStatus = String(run.status || statusPayload.status || 'QUEUED').toUpperCase();
      const embStatus = String(statusPayload.embedding_status || 'NOT_INDEXED').toUpperCase();
      const counts = statusPayload.artifact?.element_counts || run.progress_metadata || {};

      if (runStatus === 'QUEUED' || runStatus === 'EXTRACTING') {
        setPipelineUpToForWorkspace(workspaceId, 2, 3, 'Extracting', 48, `Extracting structure and tables from ${filename}…`);
      } else if (runStatus === 'PII_SCANNING') {
        setPipelineUpToForWorkspace(workspaceId, 3, 4, 'PII scan', 60, 'Scanning and redacting sensitive entities…');
      } else if (runStatus === 'NLP_ENRICHMENT' || runStatus === 'CHUNKING') {
        setPipelineUpToForWorkspace(workspaceId, 4, 5, 'Chunking', 72, 'Building concept-focused chunks…');
      } else if (runStatus === 'UPLOADING_ARTIFACT') {
        setPipelineUpToForWorkspace(workspaceId, 5, 6, 'Saving extraction', 82, 'Saving sanitized extraction artifacts…');
      } else if (runStatus === 'QUARANTINED') {
        setPipelineUpToForWorkspace(workspaceId, 3, null, 'Quarantined', 45, `${filename} quarantined for review.`);
        if (state.activeWorkspaceId === workspaceId) setRagStep(4, 'quarantined', 'Quarantined');
        if (state.activeWorkspaceId === workspaceId) await loadDocuments(false);
        return;
      } else if (runStatus === 'FAILED') {
        setPipelineUpToForWorkspace(workspaceId, 2, null, 'Failed', 35, run.error_message || 'Extraction failed.');
        if (state.activeWorkspaceId === workspaceId) {
          setRagStep(3, 'failed', 'Failed');
          $('pipelineDot').className = 'dot err';
          await loadDocuments(false);
        }
        return;
      } else if (runStatus === 'SANITIZED') {
        if (embStatus === 'INDEXED') {
          setPipelineUpToForWorkspace(workspaceId, 7, null, 'Indexed', 78, `${filename} embedded · Stage 4/5 await your question`);
          if (state.activeWorkspaceId === workspaceId) {
            setRagStep(8, '', 'Awaiting query');
            setRagStep(9, '', 'Awaiting query');
            $('pipelineTelemetryBox').innerHTML = `<strong>Embedding complete</strong><br>${Number(statusPayload.embedded_chunk_count ?? counts.chunk_count ?? 0)} chunks · BGE-M3 (${Number(statusPayload.embedding_dimensions || 1024)}D)`;
            await loadDocuments(false);
            toast(`${filename} is embedded and ready.`);
          }
          return;
        }
        if (embStatus === 'FAILED') {
          setPipelineUpToForWorkspace(workspaceId, 6, null, 'Embedding failed', 68, statusPayload.embedding_error_message || 'Ollama vector indexing failed.');
          if (state.activeWorkspaceId === workspaceId) {
            setRagStep(7, 'failed', 'Failed');
            await loadDocuments(false);
          }
          return;
        }
        // Trigger auto-index if backend left it in NOT_INDEXED
        if (embStatus === 'NOT_INDEXED') {
          try { await ingestionApi.indexDocument(documentId); } catch (_) {}
        }
        setPipelineUpToForWorkspace(workspaceId, 6, ['QUEUED', 'EMBEDDING'].includes(embStatus) ? 7 : null,
          ['QUEUED', 'EMBEDDING'].includes(embStatus) ? 'Embedding' : 'Awaiting embedding',
          68, ['QUEUED', 'EMBEDDING'].includes(embStatus) ? 'Generating BGE-M3 vectors in pgvector…' : 'Extraction is done; embeddings have not started.');
      }
      await new Promise(r => setTimeout(r, 1100));
    }
  }

  function startBackgroundStatusPoll() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    let ticks = 0;
    state.pollTimer = setInterval(async () => {
      ticks++;
      await loadDocuments(false);
      const anyBusy = state.documents.some(d =>
        ['QUEUED','EXTRACTING','PII_SCANNING','NLP_ENRICHMENT','CHUNKING','UPLOADING_ARTIFACT'].includes(String(d.ingestion_status || '').toUpperCase()) ||
        ['QUEUED','EMBEDDING'].includes(String(d.embedding_status || '').toUpperCase())
      );
      if (!anyBusy || ticks > 60) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    }, 2000);
  }

  $('dropZone').addEventListener('click', e => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'OPTION') return;
    if (!$('fileInput').disabled) $('fileInput').click();
  });
  $('fileInput').addEventListener('change', e => handleFiles(e.target.files));
  $('dropZone').addEventListener('dragover', e => { e.preventDefault(); $('dropZone').classList.add('drag-over'); });
  $('dropZone').addEventListener('dragleave', () => $('dropZone').classList.remove('drag-over'));
  $('dropZone').addEventListener('drop', e => {
    e.preventDefault();
    $('dropZone').classList.remove('drag-over');
    if (!$('fileInput').disabled) handleFiles(e.dataTransfer.files);
  });

  /* STAGE 4 CONVERSATIONAL CHAT (DIRECT ANSWERS, NO RAW CHUNK DUMPS) */
  function formatAnswerHtml(rawText) {
    let safe = escapeHtml(String(rawText || '').trim());
    // Convert bold **text**
    safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Convert inline code `code`
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Convert bracketed citations [1] or [1, 2] into sleek badges
    safe = safe.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, '<span class="cite-badge">[$1]</span>');

    const blocks = safe.split(/\n{2,}/);
    return blocks.map(block => {
      const lines = block.trim().split('\n');
      if (lines.every(l => /^([-*•]|\d+[.)])\s+/.test(l.trim()))) {
        const items = lines.map(l => `<li>${l.trim().replace(/^([-*•]|\d+[.)])\s+/, '')}</li>`).join('');
        return `<ul>${items}</ul>`;
      }
      return `<p>${lines.join('<br>')}</p>`;
    }).join('');
  }

  async function loadWorkspaceChatHistory(workspaceId) {
    try {
      const payload = await chatApi.list(workspaceId);
      if (state.activeWorkspaceId !== workspaceId) return;
      const messages = (payload.messages || []).map(message => ({
        id: message.id,
        role: message.role,
        content: message.content,
        citations: message.citations || [],
        telemetry: message.telemetry || {},
        created_at: message.created_at,
      }));
      state.chatHistoryByWorkspace.set(workspaceId, messages);
    } catch (error) {
      // Keep any in-memory messages visible if history cannot be fetched temporarily.
      if (!state.chatHistoryByWorkspace.has(workspaceId)) state.chatHistoryByWorkspace.set(workspaceId, []);
      if (state.activeWorkspaceId === workspaceId) toast(error.message || 'Could not load saved chat history.', 'error');
    }
    if (state.activeWorkspaceId === workspaceId) renderChatHistory();
  }

  function getWorkspaceChatHistory() {
    const wsId = state.activeWorkspaceId || 'default';
    if (!state.chatHistoryByWorkspace.has(wsId)) state.chatHistoryByWorkspace.set(wsId, []);
    return state.chatHistoryByWorkspace.get(wsId);
  }

  function renderChatHistory() {
    const history = getWorkspaceChatHistory();
    const stream = $('chatStream');
    if (!history.length) {
      $('chatWelcome').classList.remove('hidden');
      stream.querySelectorAll('.msg-row').forEach(n => n.remove());
      return;
    }
    $('chatWelcome').classList.add('hidden');
    stream.querySelectorAll('.msg-row').forEach(n => n.remove());

    for (const item of history) {
      const row = document.createElement('div');
      row.className = `msg-row ${item.role}`;
      if (item.role === 'user') {
        row.innerHTML = `<div class="msg-user-bubble">${escapeHtml(item.content)}</div>`;
      } else {
        const cites = (item.citations || []).map(c => {
          const label = [c.filename || 'Source', c.version_number ? `v${c.version_number}` : '', c.location || ''].filter(Boolean).join(' · ');
          const scoreBit = c.rerank_score !== undefined ? ` · Rerank ${Number(c.rerank_score).toFixed(2)}` : '';
          return `<span class="citation-pill" title="${escapeHtml(c.excerpt || '')}"><strong>[${Number(c.index)}]</strong> ${escapeHtml(label)}${escapeHtml(scoreBit)}</span>`;
        }).join('');
        const tel = item.telemetry || {};
        const telLine = tel.total_ms !== undefined
          ? `<div class="msg-telemetry">
               <span>Retrieved ${Number(tel.candidates_retrieved ?? 0)} → Reranked Top ${Number(tel.reranked_top_k ?? 0)}</span>
               <span>·</span>
               <span>Embed ${Math.round(Number(tel.query_embedding_ms || 0))}ms · pgvector ${Math.round(Number(tel.pgvector_search_ms || 0))}ms · Rerank ${Math.round(Number(tel.rerank_ms || 0))}ms · Synthesis ${Math.round(Number(tel.generation_ms || 0))}ms</span>
               <span>·</span>
               <span>Model: ${escapeHtml(tel.chat_model || 'BGE-M3')}</span>
             </div>`
          : '';
        row.innerHTML = `<div class="msg-assistant-card">
          <div class="msg-assistant-body">${formatAnswerHtml(item.content)}</div>
          ${cites ? `<div class="citation-strip">${cites}</div>` : ''}
          ${telLine}
        </div>`;
      }
      stream.appendChild(row);
    }
    stream.scrollTop = stream.scrollHeight;
  }

  async function askWorkspaceQuestion(queryText) {
    const ws = activeWorkspace();
    if (!ws) { toast('Select a workspace first.', 'error'); return; }
    const query = String(queryText || '').trim();
    if (!query || state.chatInFlight) return;
    const indexedSourceCount = groupedSources().filter(group => String(group.current?.embedding_status || '').toUpperCase() === 'INDEXED').length;
    if (!indexedSourceCount) {
      toast('Embed at least one source in this workspace before asking a question.', 'error');
      return;
    }

    const topK = Number($('searchTopK').value || 3);
    const selectedModel = $('chatModelSelect') ? $('chatModelSelect').value : 'auto';
    const chatModel = (selectedModel && selectedModel !== 'auto') ? selectedModel : null;
    const includeSuperseded = $('searchIncludeSuperseded').checked;
    const history = getWorkspaceChatHistory();
    const priorTurns = history.slice(-6).map(h => ({ role: h.role, content: h.content }));
    const userMessage = { role: 'user', content: query, citations: [], telemetry: {}, created_at: new Date().toISOString() };
    history.push(userMessage);
    renderChatHistory();
    $('chatQueryInput').value = '';
    state.chatInFlight = true;
    $('chatSubmitBtn').disabled = true;
    $('chatSubmitBtn').textContent = 'Thinking…';
    let userMessageSaved = false;
    let stepTimer = null;

    try {
      await chatApi.append(ws.id, userMessage);
      userMessageSaved = true;
      if (state.activeWorkspaceId === ws.id) {
        renderPipelineForWorkspace(ws.id);
        setRagStep(8, 'active', 'Running…');
        $('pipelineOverall').textContent = 'Reranking';
        $('pipelineProgressFill').style.width = '88%';
        $('pipelineSub').textContent = `Retrieving and reranking the best ${topK} passages…`;
      }
      stepTimer = setTimeout(() => {
        if (state.chatInFlight && state.activeWorkspaceId === ws.id) {
          setRagStep(8, 'done', 'Done');
          setRagStep(9, 'active', 'Running…');
          $('pipelineOverall').textContent = 'Synthesizing';
          $('pipelineProgressFill').style.width = '95%';
          $('pipelineSub').textContent = 'Generating a grounded answer…';
        }
      }, 320);

      const response = await searchApi.ask({
        workspace_id: ws.id,
        query,
        candidate_k: 20,
        top_k: topK,
        include_superseded: includeSuperseded,
        history: priorTurns,
        chat_model: chatModel
      });
      if (stepTimer) clearTimeout(stepTimer);

      const tel = response.pipeline_telemetry || {};
      if (state.activeWorkspaceId === ws.id) {
        setRagStep(8, 'done', 'Done');
        setRagStep(9, 'done', 'Done');
        $('pipelineOverall').textContent = 'Answered';
        $('pipelineProgressFill').style.width = '100%';
        $('pipelineSub').textContent = `Answered in ${Math.round(Number(tel.total_ms || 0))} ms`;
        $('pipelineTelemetryBox').innerHTML = `<strong>Answer complete</strong><br>${Number(tel.candidates_retrieved || 0)} passages · Reranked ${Number(tel.reranked_top_k || topK)} · ${Math.round(Number(tel.total_ms || 0))} ms`;
      }

      const assistantMessage = {
        role: 'assistant',
        content: response.answer || 'No answer returned.',
        citations: response.citations || [],
        telemetry: tel,
        created_at: new Date().toISOString(),
      };
      history.push(assistantMessage);
      try {
        await chatApi.append(ws.id, assistantMessage);
      } catch (saveError) {
        toast(`Answer generated, but it could not be saved: ${saveError.message || 'storage error'}`, 'error');
      }
      renderChatHistory();
      if (state.activeWorkspaceId === ws.id) await loadDocuments(false);
    } catch (error) {
      if (stepTimer) clearTimeout(stepTimer);
      if (state.activeWorkspaceId === ws.id) {
        setRagStep(9, 'failed', 'Error');
        $('pipelineOverall').textContent = 'Error';
      }
      const failureMessage = {
        role: 'assistant',
        content: `**Could not complete query:** ${error.message || 'Check that the local RAG services are running.'}`,
        citations: [],
        telemetry: {},
        created_at: new Date().toISOString(),
      };
      history.push(failureMessage);
      if (userMessageSaved) {
        try { await chatApi.append(ws.id, failureMessage); } catch (_) { /* Keep the failure visible in this page. */ }
      }
      renderChatHistory();
    } finally {
      state.chatInFlight = false;
      $('chatSubmitBtn').disabled = false;
      $('chatSubmitBtn').textContent = 'Ask ↵';
    }
  }

  $('chatForm').addEventListener('submit', e => {
    e.preventDefault();
    askWorkspaceQuestion($('chatQueryInput').value);
  });

  document.querySelectorAll('[data-starter]').forEach(btn => {
    btn.addEventListener('click', () => askWorkspaceQuestion(btn.dataset.starter));
  });

  $('clearChatBtn').addEventListener('click', async () => {
    const wsId = state.activeWorkspaceId;
    if (!wsId) return;
    const ok = await askConfirm('Clear this chat?', 'All messages in this workspace will be deleted. Documents and chats in other workspaces will remain.');
    if (!ok) return;
    try {
      await chatApi.clear(wsId);
      state.chatHistoryByWorkspace.set(wsId, []);
      renderChatHistory();
      toast('This workspace chat was cleared.');
    } catch (error) {
      toast(error.message || 'Could not clear chat history.', 'error');
    }
  });

  /* WORKSPACE & DOCUMENT ACTIONS */
  $('newWorkspaceBtn').addEventListener('click', async () => {
    if (state.busy || state.uploading) return;
    state.busy = true;
    try {
      const payload = await workspaceApi.createWorkspace();
      await loadSessions(payload.workspace?.id);
      await selectWorkspace(payload.workspace?.id);
      toast('New workspace created.');
    } catch (e) {
      toast(e.message || 'Could not create workspace.', 'error');
    } finally {
      state.busy = false;
    }
  });

  $('refreshBtn').addEventListener('click', async () => {
    try {
      await loadSessions(state.activeWorkspaceId);
      await loadDocuments(true);
      await checkModelsHealth(true);
      toast('Workspace synchronized.');
    } catch (e) {
      toast(e.message || 'Could not refresh.', 'error');
    }
  });

  async function checkModelsHealth(silent = false) {
    try {
      const health = await apiRequest('/api/v1/rag/health');
      const chatLabel = health.active_chat_model ? health.active_chat_model : 'extractive';
      $('ollamaStatus').textContent = `${health.embedding_model} · ${chatLabel}`;
      $('ollamaDot').className = 'dot';
      const select = $('chatModelSelect');
      if (select && Array.isArray(health.available_chat_models) && health.available_chat_models.length) {
        const currentVal = select.value;
        const existingValues = new Set([...select.options].map(o => o.value));
        for (const m of health.available_chat_models) {
          if (!existingValues.has(m)) {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = `${m} (installed)`;
            select.appendChild(opt);
          }
        }
        if ([...select.options].some(o => o.value === currentVal)) select.value = currentVal;
      }
      if (!silent) toast(`Ollama ready: Stage 3/4 ${health.embedding_model} (${health.embedding_dimensions}D) + Stage 5 LLM ${chatLabel}`);
    } catch (e) {
      $('ollamaStatus').textContent = 'Ollama check needed';
      $('ollamaDot').className = 'dot warn';
      if (!silent) toast(e.message || 'Could not reach Ollama.', 'error');
    }
  }
  $('checkModelsBtn').addEventListener('click', () => checkModelsHealth(false));

  async function downloadDoc(docId) {
    if (state.demoMode) { toast('Demo document download simulated.'); return; }
    try {
      const payload = await documentApi.downloadUrl(docId);
      if (payload.download_url) window.open(payload.download_url, '_blank', 'noopener,noreferrer');
    } catch (e) {
      toast(e.message || 'Could not generate download link.', 'error');
    }
  }

  async function retryPipeline(docId) {
    try {
      await ingestionApi.processDocument(docId);
      toast('Re-running extraction & vector indexing pipeline…');
      await pollDocumentEndToEnd(docId, 'Document');
    } catch (e) {
      toast(e.message || 'Could not retry pipeline.', 'error');
    }
  }

  async function deleteSource(sourceId) {
    const ok = await askConfirm('Delete document source?', 'This removes all versions and pgvector embeddings for this source from the workspace.');
    if (!ok) return;
    try {
      await documentApi.deleteSource(sourceId);
      await loadDocuments(false);
      await loadSessions(state.activeWorkspaceId);
      toast('Source and its vectors removed.');
    } catch (e) {
      toast(e.message || 'Could not delete source.', 'error');
    }
  }

  $('logoutBtn').addEventListener('click', async () => {
    try { if (!state.demoMode) await authApi.logout(); } catch (_) {}
    state.accessToken = null;
    state.demoMode = false;
    showAuth('login');
  });

  /* INTERACTIVE DEMO MODE FOR CANVAS PREVIEW WHEN LOCALHOST:8000 IS NOT RUNNING */
  function startDemoStudio() {
    state.demoMode = true;
    state.user = { id: 'usr_demo', username: 'demo_user', email: 'demo@example.com' };
    state.currentSessionId = 'sess_demo_1';
    state.activeWorkspaceId = 'wsp_demo_1';
    state.expandedSessionIds = new Set(['sess_demo_1']);
    state.sessions = [{
      id: 'sess_demo_1',
      name: 'Session 1',
      status: 'ACTIVE',
      workspaces: [
        { id: 'wsp_demo_1', session_id: 'sess_demo_1', name: 'NLP & ML Architecture', document_count: 2 },
        { id: 'wsp_demo_2', session_id: 'sess_demo_1', name: 'Q3 Financial Metrics', document_count: 1 }
      ]
    }];
    state.documents = [
      {
        id: 'doc_demo_ner',
        source_id: 'src_demo_1',
        source_name: 'nlp_reference_guide',
        workspace_id: 'wsp_demo_1',
        original_name: 'nlp_reference_guide.pdf',
        file_size_bytes: 418240,
        version_number: 1,
        is_current: true,
        status: 'UPLOADED',
        ingestion_status: 'SANITIZED',
        embedding_status: 'INDEXED',
        embedded_chunk_count: 42,
        embedding_dimensions: 1024,
        pii_finding_count: 0
      },
      {
        id: 'doc_demo_xlsx',
        source_id: 'src_demo_2',
        source_name: 'model_benchmarks_2026',
        workspace_id: 'wsp_demo_1',
        original_name: 'model_benchmarks_2026.xlsx',
        file_size_bytes: 94120,
        version_number: 2,
        is_current: true,
        status: 'UPLOADED',
        ingestion_status: 'SANITIZED',
        embedding_status: 'INDEXED',
        embedded_chunk_count: 18,
        embedding_dimensions: 1024,
        pii_finding_count: 2
      },
      {
        id: 'doc_demo_finance',
        source_id: 'src_demo_finance',
        source_name: 'q3_financial_metrics',
        workspace_id: 'wsp_demo_2',
        original_name: 'q3_financial_metrics.csv',
        file_size_bytes: 16384,
        version_number: 1,
        is_current: true,
        status: 'UPLOADED',
        ingestion_status: 'SANITIZED',
        embedding_status: 'INDEXED',
        embedded_chunk_count: 12,
        embedding_dimensions: 1024,
        pii_finding_count: 0
      }
    ];
    showApp();
    renderSessionSidebar();
    renderWorkspaceHeader();
    renderSourcesList();
    setPipelineUpTo(7, null, 'Indexed', 100, '2 sources automatically extracted & indexed (60 chunks)');
    $('ollamaStatus').textContent = 'bge-m3 · llama3.2';
    toast('Interactive Studio Demo loaded. Ask a question in the center panel or drop a file!');
  }

  async function demoApiRouter(path, options = {}) {
    await new Promise(r => setTimeout(r, 180));
    const chatMatch = path.match(/^\/api\/v1\/workspaces\/([^/]+)\/chat-history$/);
    if (chatMatch) {
      const workspaceId = decodeURIComponent(chatMatch[1]);
      if (!state.demoChatHistoryByWorkspace.has(workspaceId)) state.demoChatHistoryByWorkspace.set(workspaceId, []);
      const messages = state.demoChatHistoryByWorkspace.get(workspaceId);
      if ((options.method || 'GET').toUpperCase() === 'GET') return { workspace_id: workspaceId, messages };
      if ((options.method || '').toUpperCase() === 'POST') {
        const body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
        const message = { ...body, id: `msg_demo_${Date.now()}`, created_at: new Date().toISOString() };
        messages.push(message);
        return { workspace_id: workspaceId, message };
      }
      if ((options.method || '').toUpperCase() === 'DELETE') {
        state.demoChatHistoryByWorkspace.set(workspaceId, []);
        return { workspace_id: workspaceId, deleted: true };
      }
    }
    if (path === '/api/v1/sessions') return { sessions: state.sessions, current_session_id: state.currentSessionId };
    const documentsMatch = path.match(/^\/api\/v1\/documents\?workspace_id=([^&]+)/);
    if (documentsMatch) {
      const workspaceId = decodeURIComponent(documentsMatch[1]);
      return { documents: state.documents.filter(doc => doc.workspace_id === workspaceId) };
    }
    if (path === '/api/v1/rag/health') {
      return { ok: true, embedding_model: 'bge-m3', embedding_dimensions: 1024, active_chat_model: 'llama3.2' };
    }
    if (path === '/api/v1/workspaces' && options.method === 'POST') {
      const nextNum = state.sessions[0].workspaces.length + 1;
      const ws = { id: `wsp_demo_${nextNum}`, session_id: state.currentSessionId, name: `Workspace ${nextNum}`, document_count: 0 };
      state.sessions[0].workspaces.push(ws);
      return { workspace: ws };
    }
    if (path === '/api/v1/documents/upload-intents') {
      const body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
      const id = `doc_demo_${Date.now()}`;
      const srcId = body.source_id || `src_demo_${Date.now()}`;
      state.documents.unshift({
        id,
        source_id: srcId,
        source_name: body.filename.replace(/\.[^.]+$/, ''),
        workspace_id: body.workspace_id,
        original_name: body.filename,
        file_size_bytes: body.file_size_bytes,
        version_number: body.source_id ? 2 : 1,
        is_current: true,
        status: 'UPLOADED',
        ingestion_status: 'SANITIZED',
        embedding_status: 'NOT_INDEXED',
        embedded_chunk_count: 0,
        embedding_dimensions: 1024,
        pii_finding_count: 1
      });
      return { duplicate: false, upload_intent_id: `upl_${id}`, document: { id }, s3_upload: { url: '', fields: {} } };
    }
    if (path.includes('/upload-intents/') && path.endsWith('/complete')) {
      return { duplicate: false, document: state.documents[0] };
    }
    const indexMatch = path.match(/^\/api\/v1\/ingestion\/documents\/([^/]+)\/index$/);
    if (indexMatch && (options.method || 'GET').toUpperCase() === 'POST') {
      const documentId = decodeURIComponent(indexMatch[1]);
      const doc = state.documents.find(item => item.id === documentId);
      if (doc) {
        doc.embedding_status = 'EMBEDDING';
        doc.embedded_chunk_count = 0;
      }
      return { document_id: documentId, status: 'EMBEDDING', queued: true };
    }
    const statusMatch = path.match(/^\/api\/v1\/ingestion\/documents\/([^/]+)\/status$/);
    if (statusMatch) {
      const documentId = decodeURIComponent(statusMatch[1]);
      const doc = state.documents.find(item => item.id === documentId);
      if (doc?.embedding_status === 'EMBEDDING') {
        // The next poll simulates completion; the green indexed state only appears then.
        doc.embedding_status = 'INDEXED';
        doc.embedded_chunk_count = 24;
      }
      return {
        status: doc?.ingestion_status || 'SANITIZED',
        embedding_status: doc?.embedding_status || 'NOT_INDEXED',
        embedded_chunk_count: Number(doc?.embedded_chunk_count || 0),
        embedding_dimensions: Number(doc?.embedding_dimensions || 1024),
        run: { status: doc?.ingestion_status || 'SANITIZED' },
        artifact: { element_counts: { element_count: 31, table_count: 2, chunk_count: 24 } }
      };
    }
    if (path === '/api/v1/ask') {
      const body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
      const q = String(body.query || '').toLowerCase();
      let answer = "Based on the indexed workspace documents, **Sourcebook** extracts layout-aware text and tables with **Docling**, redacts PII with **Presidio**, stores `main_content` in S3, and indexes atomic chunks in `pgvector` using **BGE-M3 (1,024D)** [1]. During Stage 4 chat, it retrieves the top 20 hybrid candidates, reranks them to the top 3, and synthesizes a direct answer [2].";
      if (q.includes('ner')) {
        answer = "**Named Entity Recognition (NER)** is a sequence-labeling subtask of Natural Language Processing that locates and classifies atomic text spans into predefined semantic categories such as **Person (`PERSON`)**, **Organization (`ORG`)**, **Location (`GPE`/`LOC`)**, **Date/Time**, and domain-specific identifiers [1].\n\n- **How it works in Sourcebook**: Stage 2 uses spaCy (`en_core_web_sm`) for entity enrichment alongside Microsoft Presidio recognizers to redact sensitive identifiers before embedding [1, 2].";
      } else if (q.includes('machine learning')) {
        answer = "**Machine Learning (ML)** is a subfield of artificial intelligence where algorithms learn statistical patterns from structured and unstructured data to make predictions or decisions without being explicitly programmed for each rule [1].";
      }
      return {
        workspace_id: body.workspace_id,
        query: body.query,
        answer,
        citations: [
          { index: 1, filename: 'nlp_reference_guide.pdf', version_number: 1, location: 'Page 4', rerank_score: 0.94, excerpt: 'NER (Named Entity Recognition): Identifying and classifying named entities in unstructured text.' },
          { index: 2, filename: 'model_benchmarks_2026.xlsx', version_number: 2, location: 'Sheet Summary · Row 3', rerank_score: 0.88, excerpt: 'Row 3: Stage: Stage 4 Reranker; Input Pool: 20 chunks; Output: Top 3 chunks.' }
        ],
        pipeline_telemetry: {
          candidates_retrieved: 20,
          reranked_top_k: body.top_k || 3,
          embedding_model: 'bge-m3',
          reranker_model: 'bge-m3',
          chat_model: 'llama3.2',
          query_embedding_ms: 18,
          pgvector_search_ms: 9,
          rerank_ms: 24,
          generation_ms: 210,
          total_ms: 261
        }
      };
    }
    return {};
  }

  $('demoModeBtn').addEventListener('click', startDemoStudio);

  /* INITIALIZATION */
  (async () => {
    try {
      await rawRequest('/health');
      $('apiStatus').textContent = 'Backend connected';
      $('apiDot').className = 'dot';
      const ok = await refreshAccessToken();
      if (ok && state.accessToken) {
        showApp();
        await loadSessions();
        await selectWorkspace(firstWorkspaceId());
        checkModelsHealth(true);
      } else {
        showAuth('login');
      }
    } catch (_) {
      $('apiStatus').textContent = 'Local API offline';
      $('apiDot').className = 'dot warn';
      showAuth('login');
    }
  })();
})();
