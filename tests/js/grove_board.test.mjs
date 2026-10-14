// Headless check that the garden board renders hostile posts inert (StoneSage/backend/grove/static/board.js).
// Run by tests/test_grove_board.py; needs node. Usage: node grove_board.test.mjs <path to board.js>
// The fake DOM below REFUSES innerHTML / outerHTML / insertAdjacentHTML (it throws), records every tag and attribute that is created,
// and serialises text the way a browser would show it. A post is rendered with script tags, onerror images and quotes in every
// field a resident controls; the test then checks that no dangerous element or attribute exists and that the text is shown literally.
import assert from 'assert';
import { pathToFileURL } from 'url';

const mod = await import(pathToFileURL(process.argv[2]).href);

const created = [];            // every tag name createElement was asked for
const attrs = [];              // every attribute name setAttribute was asked for

class Node_ {
  constructor(tag) { this.tagName = tag; this.children = []; this._text = ''; this.dataset = {}; this.className = ''; this.value = ''; }
  setAttribute(name, value) { attrs.push(name); this['@' + name] = String(value); }
  appendChild(c) { this.children.push(c); return c; }
  append(...cs) { cs.forEach((c) => this.children.push(c)); }
  set textContent(v) { this._text = String(v); if (v === '') this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  set innerHTML(v) { throw new Error('innerHTML was used'); }
  set outerHTML(v) { throw new Error('outerHTML was used'); }
  insertAdjacentHTML() { throw new Error('insertAdjacentHTML was used'); }
  all() { return [this, ...this.children.flatMap((c) => c.all())]; }
}
const doc = { createElement(tag) { created.push(tag.toLowerCase()); return new Node_(tag.toLowerCase()); } };

const HOSTILE = [
  '<script>alert(1)</script>',
  '<img src=x onerror=alert(2)>',
  '"><svg onload=alert(3)>',
  "'; document.location='http://evil.example/'; '",
  '<iframe src="javascript:alert(4)"></iframe>',
];
const posts = [];
HOSTILE.forEach((h, i) => {
  posts.push({ id: 'res-' + i + ':work:0001-x', kind: 'work', resident: 'res-' + i, name: h, ts: 1700000000 + i, title: h, form: h, model: h,
               text: h + '\n' + h, comments: 1, rating: 3.5, ratings: 2, session: 'x',
               novelty: { checked: true, nearest: h, max_similarity: 0.5, others: { checked: true, nearest: h, resident: h, max_similarity: 0.4 } },
               variation_on: h, resident_rating: 2.5, resident_ratings: 2, resident_comments: 1, cross_read: [h] });
  posts.push({ id: 'res-' + i + ':wish:1700000005.000', kind: 'wish', resident: 'res-' + i, name: h, ts: 1700000005 + i, text: h,
               request: { id: 'res-' + i + '.r00001', model: h, why: h, status: 'declined', decided_by: h, note: h } });
  posts.push({ id: 'res-' + i + ':session:20260101T000000Z-abcd', kind: 'session', resident: 'res-' + i, name: h, ts: 1700000100 + i, model: h,
               state: 'ended', reason: h, turns: 3, tokens: 99, web_total: 2,
               web: [{ action: 'search', target: h, domain: 'search', bytes: 100, outcome: 'ok' },
                     { action: 'read', target: h, domain: h, bytes: 5000, outcome: h }],
               window: [h], comments_read: [{ post: h, count: 1 }],
               cross: [{ action: h, post: h, count: 2 }], model_requests: [{ id: h, model: h }] });
});
const S = { posts, residents: HOSTILE.map((h, i) => ({ id: 'res-' + i, name: h, paused: i === 0 })), status: { ok: true, running: HOSTILE[0] },
            resident: 'res-0', kind: '', open: new Set([posts[0].id]), message: HOSTILE[1],
            comments: { [posts[0].id]: [{ author: HOSTILE[2], ts: 1700000000, text: HOSTILE[0], rating: 4 },
                                        { author: HOSTILE[3], kind: 'agent', resident: 'res-1', ts: 1700000001, text: HOSTILE[1], rating: 5 }] },
            requests: HOSTILE.map((h, i) => ({ id: 'res-' + i + '.r0000' + (i + 1), resident: 'res-' + i, name: h, model: h, why: h, current_model: h,
                                               configured: i % 2 === 0 })),
            layouts: HOSTILE };

const root = new Node_('div');
mod.renderBoard(doc, root, S);

const all = root.all();
// 1. only harmless element types were ever created
const allowed = new Set(['div', 'span', 'strong', 'pre', 'button', 'select', 'option', 'input', 'textarea']);
for (const t of created) assert.ok(allowed.has(t), 'unexpected element created: ' + t);
for (const bad of ['script', 'img', 'iframe', 'svg', 'a', 'style', 'link', 'object', 'embed']) {
  assert.ok(!created.includes(bad), bad + ' was created');
}
// 2. only fixed attribute names, never an event handler or a url
const okAttrs = new Set(['style', 'type', 'title', 'placeholder', 'rows', 'disabled', 'value', 'selected']);
for (const a of attrs) assert.ok(okAttrs.has(a), 'unexpected attribute: ' + a);
// 3. no attribute value carries garden text (they are all fixed strings)
for (const n of all) {
  for (const k of Object.keys(n)) {
    if (k.startsWith('@') && k !== '@value') {
      for (const h of HOSTILE) assert.ok(!n[k].includes(h), 'garden text in attribute ' + k);
    }
  }
}
// 4. the hostile strings are present as literal text
const text = root.textContent;
for (const h of HOSTILE) assert.ok(text.includes(h), 'the text was not shown literally: ' + h);
// 5. data attributes are only post ids made of safe characters (ids are built by the worker from a-z0-9 and fixed words)
for (const n of all) {
  for (const [k, v] of Object.entries(n.dataset)) {
    if (k === 'post' || k === 'comments' || k === 'send' || k === 'draft') assert.match(v, /^[a-z0-9-]+:(work|session|journal|wish|draft):[A-Za-z0-9._-]+$/);
    if (k === 'req' || k === 'reqnote' || k === 'reqlayout' || k === 'reqengine') assert.match(v, /^[a-z0-9-]+\.r[0-9]{5,}$/);
  }
}
// 6. filters work
root.textContent = '';
mod.renderBoard(doc, root, { ...S, kind: 'work', resident: 'res-1', open: new Set() });
const cards = root.all().filter((n) => n.className === 'garden-post');
assert.strictEqual(cards.length, 1);
assert.strictEqual(cards[0].dataset.post, 'res-1:work:0001-x');

// 6b. a resident's comment is shown distinctly (its own class, tagged "(resident)"), a user's is not; the two ratings are shown apart
root.textContent = '';
mod.renderBoard(doc, root, { ...S, resident: 'res-0', kind: 'work' });
const kinds = root.all().map((n) => n.className);
assert.ok(kinds.includes('garden-comment-resident') && kinds.includes('garden-comment'), 'agent and user comments are told apart');
const t6 = root.textContent;
assert.ok(t6.includes('(resident)') && t6.includes('users ') && t6.includes('residents '), 'separate rating groups');
// 6c. the request panel offers Grant / Decline per request, and layout + engine inputs only for a model with no slot
const reqs = root.all().filter((n) => n.className === 'garden-request');
assert.strictEqual(reqs.length, HOSTILE.length);
const withSlot = reqs.filter((n) => n.all().some((m) => m.dataset && m.dataset.reqlayout));
assert.strictEqual(withSlot.length, 2, 'only the two unconfigured models ask for a layout');
for (const r of reqs) {
  const acts = r.all().filter((m) => m.dataset && m.dataset.act).map((m) => m.dataset.act);
  assert.deepStrictEqual(acts.sort(), ['decline', 'grant']);
}

// 7. helpers
assert.strictEqual(mod.stars(null), '');
assert.strictEqual(mod.stars(4), '★★★★☆ 4.0');
console.log('garden board: all checks passed');
