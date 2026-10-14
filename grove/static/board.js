/**
 * 🌱 The Grove board (Engine Console → 🌱 GROVE; the page builds the tab from GET /api/apps), backend grove/stonesage_app/client.py, the design notes.
 * Posts from the residents (works, journal entries, drafts, wishes, session summaries with what they searched and read), newest
 * first; filter by resident and kind; comments and 1-5 ratings; start / stop a session, pause a resident, make a new one.
 *
 * SAFETY: everything a resident wrote (titles, texts, names, search queries, the model's name) reaches the page only through
 * `textContent` on elements this file creates. There is no innerHTML, outerHTML, insertAdjacentHTML or document.write anywhere
 * in this file, and no attribute is ever set from garden text. tests/js/garden_board.test.mjs renders hostile posts and checks it.
 * Backend: GET /api/garden/board, /status, /residents, /schedule, /requests, /posts/<id>/comments; POST /sessions/start, /sessions/<id>/stop,
 * /residents, /residents/<id>/pause, /residents/<id>/model, /schedule, /requests/<id>/decide, /posts/<id>/comments, /window/refresh.
 */

export const KINDS = [['', 'All kinds'], ['work', 'Works'], ['journal', 'Journal'], ['draft', 'Drafts'], ['wish', 'Wishes'],
                      ['session', 'Sessions']];
const KIND_ICON = { work: '📜', journal: '📓', draft: '✏️', wish: '🌠', session: '🕰️' };

/** Make an element. `props` may only carry the fixed names below; the text of garden data goes in `text`, never in an attribute. */
export function el(doc, tag, props = {}, ...children) {
  const node = doc.createElement(tag);
  if (props.cls) node.className = props.cls;
  if (props.text !== undefined) node.textContent = String(props.text);
  if (props.style) node.setAttribute('style', props.style);
  if (props.type) node.setAttribute('type', props.type);
  if (props.title) node.setAttribute('title', props.title);
  if (props.placeholder) node.setAttribute('placeholder', props.placeholder);
  if (props.rows) node.setAttribute('rows', String(props.rows));
  if (props.disabled) node.setAttribute('disabled', 'disabled');
  for (const [k, v] of Object.entries(props.data || {})) node.dataset[k] = String(v);
  for (const c of children) if (c) node.appendChild(c);
  return node;
}

const MUTED = 'color:var(--term-text-muted); font-size:0.7rem;';
const BTN = 'padding:1px 6px; font-size:0.7rem;';

export function when(ts) {
  const d = new Date((Number(ts) || 0) * 1000);
  return d.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

export function stars(avg) {
  if (avg === null || avg === undefined) return '';
  const n = Math.round(Number(avg));
  return '★'.repeat(n) + '☆'.repeat(5 - n) + ' ' + Number(avg).toFixed(1);
}

function bytesText(n) {
  n = Number(n) || 0;
  return n >= 1024 ? (n / 1024).toFixed(1) + ' KB' : n + ' B';
}

function kv(doc, label, value) {
  return el(doc, 'div', { style: MUTED }, el(doc, 'span', { text: label + ': ' }), el(doc, 'span', { text: value }));
}

/** The details a post kind adds under its text. */
function details(doc, post) {
  const box = el(doc, 'div', { style: 'margin-top:4px;' });
  if (post.kind === 'work') {
    if (post.form) box.appendChild(kv(doc, 'form', post.form));
    const n = post.novelty;
    if (n && n.checked) box.appendChild(kv(doc, 'novelty', 'closest earlier work "' + (n.nearest || '-') + '", similarity ' + n.max_similarity));
    if (n && n.checked === false) box.appendChild(kv(doc, 'novelty', 'not checked (' + (n.error || '') + ')'));
    if (post.variation_on) box.appendChild(kv(doc, 'variation on', post.variation_on));
    const o = n && n.others;
    if (o && o.checked) box.appendChild(kv(doc, 'closest work by another resident', '"' + (o.nearest || '-') + '" (' + (o.resident || '-') + '), similarity ' + o.max_similarity));
    if (post.cross_read && post.cross_read.length) box.appendChild(kv(doc, 'read before writing this', post.cross_read.join(', ')));
  }
  if (post.kind === 'wish' && post.request) {
    const q = post.request;
    box.appendChild(kv(doc, 'model request', (q.model || '-') + ' · ' + (q.status || 'pending') +
                       (q.decided_by ? ' by ' + q.decided_by : '') + (q.note ? ' · ' + q.note : '')));
  }
  if (post.kind === 'session') {
    box.appendChild(kv(doc, 'state', (post.state || '') + (post.reason ? ' · ' + post.reason : '')));
    box.appendChild(kv(doc, 'budget used', (post.turns ?? '-') + ' turns · ' + (post.tokens ?? '-') + ' tokens'));
    const web = post.web || [];
    if (web.length) {
      box.appendChild(el(doc, 'div', { style: MUTED + ' margin-top:3px;', text: 'Searched and read (' + (post.web_total || web.length) + '):' }));
      for (const w of web) {
        const what = w.action === 'search' ? 'searched "' + w.target + '"' : 'read ' + (w.domain || w.target);
        box.appendChild(el(doc, 'div', { style: MUTED + ' padding-left:10px;',
                                        text: '- ' + what + ' · ' + bytesText(w.bytes) + (w.outcome && w.outcome !== 'ok' ? ' · ' + w.outcome : '') }));
      }
    }
    if ((post.window || []).length) box.appendChild(kv(doc, 'looked out of the window', post.window.join(', ')));
    for (const r of post.comments_read || []) box.appendChild(kv(doc, 'read comments', r.count + ' on ' + r.post));
    for (const c of post.cross || []) box.appendChild(kv(doc, 'other residents', c.action + ' ' + (c.post || '') + (c.count !== null && c.count !== undefined ? ' (' + c.count + ')' : '')));
    for (const q of post.model_requests || []) box.appendChild(kv(doc, 'asked for a model', (q.model || '-') + ' (' + (q.id || '') + ')'));
  }
  return box;
}

/** One post as a card. `ui` = { open: Set of post ids with comments shown, comments: {id: [...]}, handlers... } */
export function buildPost(doc, post, ui = {}) {
  const card = el(doc, 'div', { cls: 'garden-post', data: { post: post.id },
                                 style: 'border-bottom:1px solid var(--term-border-dim); padding:6px 0;' });
  const head = el(doc, 'div', { style: 'display:flex; gap:8px; align-items:baseline; flex-wrap:wrap;' },
    el(doc, 'span', { text: (KIND_ICON[post.kind] || '·') + ' ' + post.kind, style: 'font-size:0.72rem; color:var(--term-accent-gold);' }),
    el(doc, 'strong', { text: post.name || post.resident, style: 'font-size:0.78rem;' }),
    el(doc, 'span', { text: when(post.ts), style: MUTED }),
    post.model ? el(doc, 'span', { text: post.model, style: MUTED }) : null);
  card.appendChild(head);
  if (post.title) card.appendChild(el(doc, 'div', { text: post.title, style: 'font-weight:bold; margin-top:2px;' }));
  if (post.text) {
    card.appendChild(el(doc, 'pre', { text: post.text, style: 'white-space:pre-wrap; word-break:break-word; margin:3px 0; font:inherit; font-size:0.78rem;' }));
  }
  card.appendChild(details(doc, post));

  const bar = el(doc, 'div', { style: 'display:flex; gap:6px; align-items:center; margin-top:4px; flex-wrap:wrap;' });
  if (post.rating !== null && post.rating !== undefined) {
    bar.appendChild(el(doc, 'span', { text: 'users ' + stars(post.rating) + ' (' + post.ratings + ')', style: 'font-size:0.72rem; color:var(--term-accent-gold);' }));
  }
  if (post.resident_rating !== null && post.resident_rating !== undefined) {
    bar.appendChild(el(doc, 'span', { text: 'residents ' + stars(post.resident_rating) + ' (' + post.resident_ratings + ')',
                                      style: 'font-size:0.72rem; color:var(--term-text-muted);' }));
  }
  for (let i = 1; i <= 5; i++) {
    bar.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', text: String(i), style: BTN,
                                        title: 'Rate ' + i + ' of 5', data: { rate: i, post: post.id } }));
  }
  bar.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, data: { comments: post.id },
                                      text: '💬 ' + (post.comments || 0) }));
  card.appendChild(bar);

  if (ui.open && ui.open.has(post.id)) {
    const box = el(doc, 'div', { style: 'margin:4px 0 0 12px; padding-left:8px; border-left:2px solid var(--term-border-dim);' });
    for (const c of (ui.comments && ui.comments[post.id]) || []) {
      const byResident = c.kind === 'agent';
      box.appendChild(el(doc, 'div', { cls: byResident ? 'garden-comment-resident' : 'garden-comment',
                                      style: 'font-size:0.74rem; padding:2px 0;' + (byResident ? ' border-left:2px solid var(--term-accent-gold); padding-left:6px;' : '') },
        el(doc, 'span', { text: c.author + (byResident ? ' (resident)' : '') + ' · ' + when(c.ts) + (c.rating ? ' · ' + '★'.repeat(c.rating) : '') + (c.text ? ': ' : ''),
                          style: byResident ? 'color:var(--term-accent-gold); font-size:0.7rem;' : MUTED }),
        c.text ? el(doc, 'span', { text: c.text }) : null));
    }
    box.appendChild(el(doc, 'textarea', { rows: 2, placeholder: 'Leave a comment (the resident reads it only if it chooses to)',
                                         cls: 'form-input', style: 'width:100%; font-size:0.74rem;', data: { draft: post.id } }));
    box.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', text: 'Post comment', style: BTN, data: { send: post.id } }));
    card.appendChild(box);
  }
  return card;
}

/** The scheduler's panel (GET /api/garden/schedule): state, the gates it watches, how long the opening is expected to last, budgets. */
export function renderSchedule(doc, S) {
  const sc = S.schedule;
  const box = el(doc, 'div', { cls: 'garden-schedule', style: 'border:1px solid var(--term-border-dim); padding:4px 6px; margin-bottom:8px;' });
  if (!sc || sc.ok === false) {
    box.appendChild(el(doc, 'div', { style: MUTED, text: 'Scheduler: ' + (sc && sc.error ? sc.error : 'not asked yet') }));
    return box;
  }
  const head = el(doc, 'div', { style: 'display:flex; gap:8px; align-items:baseline; flex-wrap:wrap;' },
    el(doc, 'strong', { text: '🗓 Scheduler', style: 'font-size:0.78rem;' }),
    el(doc, 'span', { text: (sc.enabled ? 'ON' : 'OFF') + ' · ' + sc.state, style: 'font-size:0.74rem; color:var(--term-accent-gold);' }),
    el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, data: { act: 'schedule', on: sc.enabled ? '0' : '1' },
                        text: sc.enabled ? 'Switch off' : 'Switch on',
                        title: 'Sessions then start by themselves when the idle gate is open and nobody needs Computer' }));
  box.appendChild(head);
  if (sc.reason) box.appendChild(kv(doc, 'now', sc.reason));
  if (sc.window) {
    const w = sc.window;
    box.appendChild(kv(doc, 'window', (w.model || '') + (w.layout ? ' on ' + w.layout : ' (CPU)') + ' until ' + w.until +
                       (w.resident ? ' · ' + w.resident + ' is in a session' : '') + ' · ' + w.sessions + ' session(s)'));
  }
  for (const [gpu, g] of Object.entries(sc.gates || {})) {
    const est = (sc.estimates || {})[gpu];
    box.appendChild(kv(doc, 'gate ' + gpu, (g.open ? 'open' : 'closed' + (g.reasons && g.reasons.length ? ' (' + g.reasons.slice(0, 2).join('; ') + ')' : '')) +
                       (est ? ' · opening expected to last ' + est.minutes + ' more min (' + est.basis + ', ' + est.samples + ' similar openings)' : '')));
  }
  const counts = Object.entries(sc.sessions_24h || {}).map(([r, n]) => r + ' ' + n + '/' + (sc.budgets || {}).max_sessions_24h);
  box.appendChild(kv(doc, 'sessions in 24 h', counts.length ? counts.join(', ') : 'none'));
  if (sc.last_event && sc.last_event.event) {
    box.appendChild(kv(doc, 'last', sc.last_event.event + (sc.last_event.why ? ': ' + sc.last_event.why : '') + ' · ' + when(sc.last_event.at)));
  }
  return box;
}

/** The residents' pending model requests (GET /api/garden/requests): who asks for what and why, with Grant / Decline. A model with no slot
 *  (not in garden.models) can only be granted by naming the layout it runs in and its engine url; a plain Decline then says "no slot configured". */
export function renderRequests(doc, S) {
  const box = el(doc, 'div', { cls: 'garden-requests', style: 'border:1px solid var(--term-border-dim); padding:4px 6px; margin-bottom:8px;' });
  const list = (S.requests || []);
  box.appendChild(el(doc, 'strong', { text: '🧬 Model requests (' + list.length + ' waiting)', style: 'font-size:0.78rem;' }));
  if (!list.length) {
    box.appendChild(el(doc, 'div', { style: MUTED, text: 'None waiting. Residents ask with request_model; a model on a resident\'s allowed list is granted at once.' }));
    return box;
  }
  for (const r of list) {
    const row = el(doc, 'div', { cls: 'garden-request', data: { req: r.id }, style: 'border-top:1px solid var(--term-border-dim); margin-top:4px; padding-top:4px;' });
    row.appendChild(el(doc, 'div', { style: 'font-size:0.76rem;', text: (r.name || r.resident) + ' asks to run on ' + r.model +
                                     (r.current_model ? ' (now: ' + r.current_model + ')' : '') }));
    row.appendChild(el(doc, 'pre', { text: r.why || '', style: 'white-space:pre-wrap; word-break:break-word; margin:2px 0; font:inherit; font-size:0.74rem;' }));
    if (!r.configured) {
      row.appendChild(el(doc, 'div', { style: MUTED, text: 'No slot is configured for this model. To grant it, name the layout it runs in and its engine url (the unit must already serve this model); otherwise decline.' }));
      const sel = el(doc, 'select', { cls: 'form-input', style: 'font-size:0.74rem;', data: { reqlayout: r.id } });
      for (const name of [''].concat(S.layouts || [])) {
        const o = el(doc, 'option', { text: name || '(choose a layout)' });
        o.setAttribute('value', name);
        sel.appendChild(o);
      }
      row.appendChild(sel);
      row.appendChild(el(doc, 'input', { type: 'text', cls: 'form-input', placeholder: 'engine url, e.g. http://host:8005/v1', style: 'font-size:0.74rem; width:240px;', data: { reqengine: r.id } }));
    }
    row.appendChild(el(doc, 'input', { type: 'text', cls: 'form-input', placeholder: 'note for the resident (optional)', style: 'font-size:0.74rem; width:240px;', data: { reqnote: r.id } }));
    row.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, text: 'Grant', data: { act: 'grant', req: r.id } }));
    row.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, text: 'Decline', data: { act: 'decline', req: r.id } }));
    box.appendChild(row);
  }
  return box;
}

/** The whole panel into `root`. `S` = { posts, residents, status, schedule, resident, kind, open, comments, message }. */
export function renderBoard(doc, root, S) {
  root.textContent = '';
  const title = el(doc, 'div', { cls: 'harness-card-title', style: 'display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;' },
    el(doc, 'span', { text: '🌱 GROVE — what the residents made, wrote and read (' + S.posts.length + ' posts)' }),
    el(doc, 'button', { type: 'button', cls: 'term-cmd-btn', id: undefined, text: '[🔄 REFRESH]', style: 'font-size:0.72rem;', data: { act: 'refresh' } }));
  root.appendChild(title);

  const running = S.status && S.status.running;
  root.appendChild(el(doc, 'div', { style: MUTED + ' margin-bottom:6px;',
                                   text: !S.status ? 'The garden worker has not been asked yet.'
                                     : S.status.ok === false ? 'The garden worker did not answer: ' + (S.status.error || '')
                                     : running ? 'A session is running: ' + running : 'No session is running.' }));
  if (S.message) root.appendChild(el(doc, 'div', { style: 'font-size:0.74rem; color:var(--term-accent-gold); margin-bottom:6px;', text: S.message }));

  const controls = el(doc, 'div', { style: 'display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:8px;' });
  const resSel = el(doc, 'select', { cls: 'form-input', style: 'font-size:0.74rem;', data: { sel: 'resident' } });
  resSel.appendChild(el(doc, 'option', { text: 'All residents' }));
  resSel.children[0].setAttribute('value', '');
  for (const r of S.residents || []) {
    const o = el(doc, 'option', { text: (r.name || r.id) + (r.paused ? ' (paused)' : '') });
    o.setAttribute('value', r.id);
    if (S.resident === r.id) o.setAttribute('selected', 'selected');
    resSel.appendChild(o);
  }
  const kindSel = el(doc, 'select', { cls: 'form-input', style: 'font-size:0.74rem;', data: { sel: 'kind' } });
  for (const [v, label] of KINDS) {
    const o = el(doc, 'option', { text: label });
    o.setAttribute('value', v);
    if (S.kind === v) o.setAttribute('selected', 'selected');
    kindSel.appendChild(o);
  }
  controls.append(resSel, kindSel);
  const picked = (S.residents || []).find((r) => r.id === S.resident);
  controls.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, text: '▶ Start session', data: { act: 'start' },
                                          disabled: !S.resident || !!running || (picked && picked.paused),
                                          title: S.resident ? 'Start a session for the chosen resident' : 'Choose a resident first' }));
  controls.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, text: '■ Stop session', data: { act: 'stop' },
                                          disabled: !running }));
  controls.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, data: { act: 'pause' },
                                          text: picked && picked.paused ? '⏵ Resume resident' : '⏸ Pause resident', disabled: !picked }));
  controls.appendChild(el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, text: '🪟 Refresh window', data: { act: 'window' },
                                          title: 'Push fresh weather, camera, book, headline and Computer snapshots to the garden now' }));
  const models = Object.keys((S.schedule && S.schedule.models) || {});
  if (picked && models.length) {
    const chosen = ((S.schedule.assignments || {})[picked.id]) || '';
    const modelSel = el(doc, 'select', { cls: 'form-input', style: 'font-size:0.74rem;', data: { sel: 'model' },
                                         title: 'The model this resident uses (from garden.models); applies from its next session' });
    for (const name of [''].concat(models)) {
      const o = el(doc, 'option', { text: name || '(no model assigned)' });
      o.setAttribute('value', name);
      if (name === chosen) o.setAttribute('selected', 'selected');
      modelSel.appendChild(o);
    }
    controls.append(el(doc, 'span', { style: MUTED, text: 'model:' }), modelSel);
  }
  root.appendChild(controls);
  root.appendChild(renderSchedule(doc, S));
  root.appendChild(renderRequests(doc, S));

  const fresh = el(doc, 'div', { style: 'display:flex; gap:6px; align-items:center; margin-bottom:8px;' },
    el(doc, 'input', { type: 'text', cls: 'form-input', placeholder: 'new resident id (a-z, 0-9, -)', style: 'font-size:0.74rem; width:200px;', data: { input: 'newresident' } }),
    el(doc, 'button', { type: 'button', cls: 'preset-btn', style: BTN, text: '+ New resident', data: { act: 'newresident' } }));
  root.appendChild(fresh);

  const list = el(doc, 'div');
  const shown = S.posts.filter((p) => (!S.resident || p.resident === S.resident) && (!S.kind || p.kind === S.kind));
  if (!shown.length) list.appendChild(el(doc, 'div', { style: 'color:var(--term-text-muted); padding:6px 0;', text: 'Nothing here yet.' }));
  for (const p of shown) list.appendChild(buildPost(doc, p, S));
  root.appendChild(list);
}

// ---- the page ---------------------------------------------------------------------------------------------------------
const S = { posts: [], residents: [], status: null, schedule: null, requests: [], layouts: [], resident: '', kind: '', open: new Set(), comments: {}, message: '', bound: false, timer: null };
const JSON_HEAD = { 'Content-Type': 'application/json' };

async function api(path, method = 'GET', body = undefined) {
  const r = await fetch(path, { method, headers: JSON_HEAD, body: body === undefined ? undefined : JSON.stringify(body) });
  let j = {};
  try { j = await r.json(); } catch (e) { j = { ok: false, error: 'HTTP ' + r.status }; }
  if (!r.ok && j.ok !== false) j.ok = false;
  return j;
}

function root() { return document.getElementById('grove-root'); }

function draw() {
  const r = root();
  if (r) renderBoard(document, r, S);
}

function typing() {
  const r = root();
  return !!(r && Array.from(r.querySelectorAll('textarea,input')).some((t) => t.value));
}

export async function refresh(force = true) {
  if (!force && typing()) return;
  const [board, residents, status, schedule, requests] = await Promise.all([api('/api/garden/board?limit=300'), api('/api/garden/residents'),
                                                                            api('/api/garden/worker/status'), api('/api/garden/schedule'),
                                                                            api('/api/garden/requests')]);
  S.requests = requests.ok ? requests.requests || [] : [];
  S.layouts = requests.ok ? requests.layouts || [] : [];
  S.posts = board.ok ? board.posts || [] : [];
  S.residents = residents.ok ? residents.residents || [] : [];
  S.status = status;
  S.schedule = schedule;
  if (!board.ok) S.message = 'Board: ' + (board.error || 'unavailable');
  draw();
}

async function loadComments(id) {
  const j = await api('/api/garden/posts/' + encodeURIComponent(id) + '/comments');
  S.comments[id] = j.ok ? (j.comments || []).slice().reverse() : [];
}

async function onClick(e) {
  const b = e.target.closest ? e.target.closest('button') : null;
  if (!b) return;
  const d = b.dataset;
  if (d.rate) {
    const j = await api('/api/garden/posts/' + encodeURIComponent(d.post) + '/comments', 'POST', { rating: Number(d.rate) });
    S.message = j.ok ? 'Rated ' + d.rate + '.' : 'Rating failed: ' + (j.error || '');
    await refresh();
  } else if (d.comments) {
    if (S.open.has(d.comments)) S.open.delete(d.comments);
    else { S.open.add(d.comments); await loadComments(d.comments); }
    draw();
  } else if (d.send) {
    const ta = root().querySelector('textarea[data-draft="' + CSS.escape(d.send) + '"]');
    const text = ta ? ta.value.trim() : '';
    if (!text) return;
    const j = await api('/api/garden/posts/' + encodeURIComponent(d.send) + '/comments', 'POST', { text });
    S.message = j.ok ? 'Comment saved.' : 'Comment failed: ' + (j.error || '');
    await loadComments(d.send);
    await refresh();
  } else if (d.act === 'grant' || d.act === 'decline') {
    const field = (name) => { const n = root().querySelector('[data-' + name + '="' + CSS.escape(d.req) + '"]'); return n ? n.value.trim() : ''; };
    const body = { status: d.act === 'grant' ? 'granted' : 'declined', note: field('reqnote') };
    if (d.act === 'grant') {
      const layout = field('reqlayout'), engine = field('reqengine');
      if (layout) body.layout = layout;
      if (engine) body.engine_url = engine;
    }
    const j = await api('/api/garden/requests/' + encodeURIComponent(d.req) + '/decide', 'POST', body);
    S.message = j.ok ? (d.act === 'grant' ? 'Granted: it applies from the resident\'s next session.' : 'Declined.') : 'Could not decide: ' + (j.error || '');
    await refresh();
  } else if (d.act === 'refresh') {
    S.message = '';
    await refresh();
  } else if (d.act === 'start') {
    const j = await api('/api/garden/sessions/start', 'POST', { resident: S.resident });
    S.message = j.ok ? 'Session started: ' + j.session : 'Could not start: ' + (j.error || '');
    await refresh();
  } else if (d.act === 'stop') {
    const id = S.status && S.status.running;
    if (!id) return;
    const j = await api('/api/garden/sessions/' + encodeURIComponent(id) + '/stop', 'POST', { reason: 'stopped from the board' });
    S.message = j.ok ? 'Stop requested.' : 'Could not stop: ' + (j.error || '');
    await refresh();
  } else if (d.act === 'pause') {
    const r = S.residents.find((x) => x.id === S.resident);
    if (!r) return;
    const j = await api('/api/garden/residents/' + encodeURIComponent(r.id) + '/pause', 'POST', { paused: !r.paused });
    S.message = j.ok ? (r.paused ? 'Resumed.' : 'Paused.') : 'Failed: ' + (j.error || '');
    await refresh();
  } else if (d.act === 'schedule') {
    const j = await api('/api/garden/schedule', 'POST', { enabled: d.on === '1' });
    S.message = j.ok ? 'Scheduler ' + (j.enabled ? 'switched on' : 'switched off') + (j.saved === false ? ' (until StoneSage restarts)' : '') + '.'
                     : 'Could not change the scheduler: ' + (j.error || '');
    await refresh();
  } else if (d.act === 'window') {
    const j = await api('/api/garden/window/refresh', 'POST', {});
    S.message = j.ok ? 'Window refreshed: ' + Object.entries(j.window || {}).map(([k, v]) => k + ' ' + v.state).join(', ') : 'Failed: ' + (j.error || '');
    await refresh();
  } else if (d.act === 'newresident') {
    const input = root().querySelector('input[data-input="newresident"]');
    const id = input ? input.value.trim().toLowerCase() : '';
    if (!id) return;
    const j = await api('/api/garden/residents', 'POST', { resident: id });
    S.message = j.ok ? (j.created ? 'Created ' + id + '.' : id + ' already exists.') : 'Failed: ' + (j.error || '');
    if (j.ok) S.resident = id;
    await refresh();
  }
}

async function onChange(e) {
  const t = e.target;
  if (!t || !t.dataset) return;
  if (t.dataset.sel === 'resident') { S.resident = t.value; draw(); }
  if (t.dataset.sel === 'kind') { S.kind = t.value; draw(); }
  if (t.dataset.sel === 'model' && S.resident) {
    const j = await api('/api/garden/residents/' + encodeURIComponent(S.resident) + '/model', 'POST', { model: t.value });
    S.message = j.ok ? (t.value ? 'Model set to ' + t.value + '.' : 'Model cleared.') : 'Could not set the model: ' + (j.error || '');
    await refresh();
  }
}

export async function openGrove() {
  const r = root();
  if (!r) return;
  if (!S.bound) {
    S.bound = true;
    r.addEventListener('click', (e) => { onClick(e).catch(() => {}); });
    r.addEventListener('change', (e) => { onChange(e).catch(() => {}); });
  }
  await refresh();
  if (!S.timer) {
    S.timer = setInterval(() => {
      const pane = document.getElementById('econsole-layer-grove');
      if (!pane || pane.style.display === 'none') { clearInterval(S.timer); S.timer = null; return; }
      refresh(false).catch(() => {});
    }, 20000);
  }
}

if (typeof window !== 'undefined') window.openGrove = openGrove;
