/* AR-Edit project page: players, the QKV explorer, recipes, method passes, results, charts. */
'use strict';
(function () {
  const D = window.AR;
  if (!D) return;
  const VID = 'static/videos/';
  // Every clip starts at raw frame TRIM (tools/build_media.py cuts frames 0..3): 77 frames at 16 fps.
  const TRIM = 4, NFRAMES = 81 - TRIM, CLIP_DUR = NFRAMES / 16;
  const RM = window.matchMedia('(prefers-reduced-motion: reduce)');
  const $ = (q, r = document) => r.querySelector(q);
  const $$ = (q, r = document) => Array.from(r.querySelectorAll(q));
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}, html) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v; else e.setAttribute(k, v === true ? '' : v);
    }
    if (html != null) e.innerHTML = html;
    return e;
  };
  const sv = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const hex = (c) => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
  const mix = (c1, c2, t) => {
    const a = hex(c1), b = hex(c2);
    return '#' + a.map((v, i) => Math.round(lerp(v, b[i], t)).toString(16).padStart(2, '0')).join('');
  };
  const strip = (s) => s.replace(/<[^>]+>/g, '');

  /* theme: every color lives in a CSS custom property (style.css has a dark and a light set).
     SVG attributes and canvas need resolved values, so PAL caches them and the charts that
     compute colors in script redraw through onTheme() when the visitor flips the theme. */
  const ROOT = document.documentElement;
  const cssv = (n) => getComputedStyle(ROOT).getPropertyValue(n).trim();
  const PAL_KEYS = ['--bg', '--bg-raise', '--ink', '--ink-2', '--ink-3', '--line', '--line-strong', '--line-faint', '--grid',
    '--orange', '--orange-2', '--tan', '--gray-noise', '--c-input', '--c-sdedit', '--c-flowedit', '--c-dynaedit', '--c-invv',
    '--c-streamedit', '--c-purple', '--c-graym', '--c-blue-0', '--c-blue-625', '--c-blue-833', '--c-blue-938', '--c-blue-1000'];
  const PAL = {};
  const readPal = () => PAL_KEYS.forEach((k) => { PAL[k] = cssv(k); });
  readPal();
  const themeHooks = [];
  const onTheme = (fn) => themeHooks.push(fn);
  (function themeToggle() {
    const btn = $('[data-theme-toggle]'), meta = $('meta[name="theme-color"]');
    let anim = 0;
    const set = (t, user) => {
      ROOT.dataset.theme = t;
      if (user) { try { localStorage.setItem('ar-theme', t); } catch (e) { /* private mode: keep it for this visit */ } }
      const label = t === 'light' ? 'Switch to dark theme' : 'Switch to light theme';
      if (btn) { btn.setAttribute('aria-label', label); btn.title = label; }
      readPal();
      if (meta) meta.content = PAL['--bg'];
      themeHooks.forEach((f) => f());
    };
    set(ROOT.dataset.theme === 'light' ? 'light' : 'dark', false);
    if (btn) btn.addEventListener('click', () => {
      if (!RM.matches) { ROOT.classList.add('theme-anim'); clearTimeout(anim); anim = setTimeout(() => ROOT.classList.remove('theme-anim'), 360); }
      set(ROOT.dataset.theme === 'light' ? 'dark' : 'light', true);
    });
  })();

  /* noise ramp: clean (orange) to noisy (gray) */
  const ramp = (u) => (u < 0.5 ? mix(PAL['--orange'], PAL['--tan'], u / 0.5) : mix(PAL['--tan'], PAL['--gray-noise'], (u - 0.5) / 0.5));

  // v: the CSS custom property (var() in HTML), color: its current value (for SVG attributes)
  // name: plain text (tooltips, aria); html: rich label for the DOM; svg: [text, kind] segments for SVG labels
  const meth = (name, v, html, svg) => ({ name, v, html: html || name, svg: svg || [[name, 0]], get color() { return PAL[v]; }, get css() { return 'var(' + v + ')'; } });
  // the paper's recipe names, e.g. <Inv, Inv, {V^t}>
  const Vh = (x) => '<b class="m">V</b><sup>' + x + '</sup>';
  const tupH = (a, b, c) => '&lang;' + a + ', ' + b + ', ' + c + '&rang;';
  const SUP = (x) => String(x).replace(/[0-9t]/g, (d) => (d === 't' ? '\u1d57' : '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079'[+d]));
  const tupP = (a, b, c) => '\u27e8' + a + ', ' + b + ', ' + c + '\u27e9';
  // SVG text with tspans: kinds 0 (plain), 'b' (bold serif V), 'i' (italic), 'sub', 'sup', 'supi'
  function svgSegs(t, segs) {
    segs.forEach(([x, k]) => {
      const a = k === 'sub' ? { 'baseline-shift': 'sub', 'font-size': '72%' } : k === 'sup' ? { 'baseline-shift': 'super', 'font-size': '72%' }
        : k === 'supi' ? { 'baseline-shift': 'super', 'font-size': '72%', 'font-style': 'italic', 'font-family': 'Newsreader, Georgia, serif' }
        : k === 'b' ? { 'font-weight': 800, 'font-family': 'Newsreader, Georgia, serif' } : k === 'i' ? { 'font-style': 'italic', 'font-family': 'Newsreader, Georgia, serif' } : {};
      const sp = document.createElementNS(NS, 'tspan');
      Object.entries(a).forEach(([kk, vv]) => sp.setAttribute(kk, vv));
      sp.textContent = x; t.appendChild(sp);
    });
    return t;
  }
  const METHOD = {
    'Input': meth('Input', '--c-input'),
    'SDEdit': meth('SDEdit', '--c-sdedit'),
    'FlowEdit': meth('FlowEdit', '--c-flowedit'),
    'DynaEdit': meth('DynaEdit', '--c-dynaedit'),
    'Inversion w/ V': meth(tupP('Inv', 'Inv', '{V' + SUP('t') + '}'), '--c-invv', tupH('Inv', 'Inv', '{' + Vh('<i>t</i>') + '}'),
      [['\u27e8Inv, Inv, {', 0], ['V', 'b'], ['t', 'supi'], ['}\u27e9', 0]]),
    'StreamEdit': meth('StreamEdit', '--c-streamedit'),
    'Ours': meth('AR-Edit', '--orange'),
  };
  const CLIP_NAME = {
    'tiger-lion': 'Tiger to lion', 'cow': 'Spotted cow', 'airplane': 'Wooden airplane', 'train': 'Winter train',
    'jellyfish': 'Polka-dot jellyfish', 'pitcher': 'Cobalt pitcher', 'piglet': 'Piglet to cat', 'rabbit': 'Snowy garden',
    'zebra': 'Sprinting zebra', 'coffee': 'Rose-gold machine', 'violin': 'Glass to violin', 'lantern': 'Red lantern',
    'seagull': 'Green seagull', 'goblet': 'Gold goblet', 'tree': 'Swaying tree', 'comic-dog': 'Comic-book dog',
    'cat-dog': 'Cat to dog', 'opera': 'Opera House at sunset', 'butterfly': 'Orange butterfly', 'street': 'Winter street',
  };
  const poster = (f) => VID + f.replace(/\.mp4$/, '.jpg');

  const ICON_PLAY = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3.2 1.6v10.8L12 7z" fill="currentColor"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.6h3v10.8H3zM8 1.6h3v10.8H8z" fill="currentColor"/></svg>';

  /* ------------------------------------------------------------------ */
  /* Synced video group                                                  */
  /* ------------------------------------------------------------------ */
  class SyncGroup {
    constructor(root, playerEl) {
      this.root = root; this.videos = []; this.playing = false; this.userPaused = RM.matches;
      this.visible = false; this.rate = 1; this.listeners = []; this.raf = 0;
      if (playerEl) this.buildPlayer(playerEl);
      new IntersectionObserver((es) => {
        for (const e of es) { this.visible = e.isIntersecting; this.visible ? this.show() : this.pause(); }
      }, { threshold: 0.12 }).observe(root);
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) this.pause(); else if (this.visible) this.show();
      });
    }
    set(list) {
      this.pause();
      this.videos = list.map((x) => x.v);
      for (const { v, src } of list) {
        v.muted = true; v.defaultMuted = true; v.playsInline = true; v.loop = true;
        v.setAttribute('muted', ''); v.setAttribute('playsinline', '');
        v.poster = poster(src);
        v.dataset.src = VID + src;
      }
      if (this.visible) this.show(); else this.emit(0, CLIP_DUR);
    }
    load() {
      for (const v of this.videos) {
        if (v.dataset.src && v.getAttribute('src') !== v.dataset.src) {
          v.setAttribute('src', v.dataset.src); v.preload = 'auto'; v.load(); v.playbackRate = this.rate;
        }
      }
    }
    show() { this.load(); if (!this.userPaused) this.play(); else this.sync(true); }
    play() {
      if (!this.videos.length) return;
      this.load();
      this.playing = true;
      const t = this.videos[0].currentTime || 0;
      for (const v of this.videos) { v.playbackRate = this.rate; if (v.readyState >= 1 && Math.abs(v.currentTime - t) > 0.04) v.currentTime = t; }
      for (const v of this.videos) v.play().catch(() => {});
      this.updateBtn();
      cancelAnimationFrame(this.raf);
      const step = () => { if (!this.playing) return; this.sync(false); this.raf = requestAnimationFrame(step); };
      this.raf = requestAnimationFrame(step);
    }
    pause() {
      this.playing = false; cancelAnimationFrame(this.raf);
      for (const v of this.videos) v.pause();
      this.updateBtn();
    }
    sync(force) {
      const L = this.videos[0]; if (!L) return;
      const t = L.currentTime;
      for (let i = 1; i < this.videos.length; i++) {
        const v = this.videos[i];
        if (v.readyState >= 1 && !v.seeking && Math.abs(v.currentTime - t) > (force ? 0.02 : 0.12)) v.currentTime = t;
        if (this.playing && v.paused && v.readyState >= 2) v.play().catch(() => {});
      }
      this.emit(t, L.duration || CLIP_DUR);
    }
    seek(t) {
      this.load();
      for (const v of this.videos) if (v.readyState >= 1) v.currentTime = t;
      this.emit(t, (this.videos[0] && this.videos[0].duration) || CLIP_DUR);
    }
    buildPlayer(p) {
      p.innerHTML = '<button class="player__btn" type="button" aria-label="Play">' + ICON_PLAY + '</button>' +
        '<input class="player__scrub" type="range" min="0" max="1000" value="0" aria-label="Seek">' +
        '<span class="player__time">0.0 / 5.1 s</span>' +
        '<button class="player__speed" type="button" aria-pressed="false">Half speed</button>';
      [this.btn, this.scrub, this.timeEl, this.speedBtn] = p.children;
      this.btn.addEventListener('click', () => {
        if (this.playing) { this.userPaused = true; this.pause(); } else { this.userPaused = false; this.play(); }
      });
      let resume = false;
      this.scrub.addEventListener('pointerdown', () => { resume = this.playing; this.pause(); });
      this.scrub.addEventListener('input', () => {
        const d = (this.videos[0] && this.videos[0].duration) || CLIP_DUR;
        this.seek(this.scrub.value / 1000 * d);
      });
      this.scrub.addEventListener('change', () => { if (resume) { resume = false; this.play(); } });
      this.speedBtn.addEventListener('click', () => {
        this.rate = this.rate === 1 ? 0.5 : 1;
        for (const v of this.videos) v.playbackRate = this.rate;
        this.speedBtn.setAttribute('aria-pressed', String(this.rate !== 1));
      });
    }
    updateBtn() {
      if (!this.btn) return;
      this.btn.innerHTML = this.playing ? ICON_PAUSE : ICON_PLAY;
      this.btn.setAttribute('aria-label', this.playing ? 'Pause' : 'Play');
    }
    emit(t, d) {
      if (this.scrub) {
        if (document.activeElement !== this.scrub) this.scrub.value = String(Math.round(t / d * 1000));
        this.scrub.style.setProperty('--p', (t / d * 100).toFixed(2) + '%');
        this.timeEl.textContent = t.toFixed(1) + ' / ' + d.toFixed(1) + ' s';
      }
      for (const f of this.listeners) f(t, d);
    }
    onTime(f) { this.listeners.push(f); }
  }

  /* ------------------------------------------------------------------ */
  /* Split compare                                                       */
  /* ------------------------------------------------------------------ */
  function makeSplit(root) {
    const handle = $('.split__handle', root);
    let pos = 50, drag = false;
    const set = (p) => {
      pos = clamp(p, 0, 100);
      root.style.setProperty('--pos', pos + '%');
      handle.setAttribute('aria-valuenow', String(Math.round(pos)));
    };
    const fromEvent = (e) => { const r = root.getBoundingClientRect(); set((e.clientX - r.left) / r.width * 100); };
    root.addEventListener('pointerdown', (e) => { drag = true; root.setPointerCapture(e.pointerId); fromEvent(e); });
    root.addEventListener('pointermove', (e) => { if (drag) fromEvent(e); });
    const end = () => { drag = false; };
    root.addEventListener('pointerup', end); root.addEventListener('pointercancel', end);
    handle.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') { set(pos - 5); e.preventDefault(); }
      if (e.key === 'ArrowRight') { set(pos + 5); e.preventDefault(); }
      if (e.key === 'Home') { set(0); e.preventDefault(); }
      if (e.key === 'End') { set(100); e.preventDefault(); }
    });
    set(50);
    return { set };
  }

  function renderPrompt(node, slug) {
    const c = D.clips[slug];
    if (!c) { node.innerHTML = ''; return; }
    node.innerHTML = '<span class="prompt__cat">' + c.cat + '</span>' +
      '<span class="prompt__k">Source</span><span class="prompt__src">' + c.from + '</span>' +
      '<span class="prompt__k">Edit</span><span class="prompt__tar">' + c.to + '</span>';
  }

  /* Accessible single-select control: radio groups (seg) and tab rows (clips). */
  function choice(container, items, current, onPick, opts = {}) {
    const role = opts.role || 'radio';
    const attr = role === 'tab' ? 'aria-selected' : 'aria-checked';
    container.querySelectorAll('button').forEach((b) => b.remove());
    const buttons = items.map(({ v, label }) => {
      const b = el('button', { type: 'button', role, 'data-v': v, [attr]: String(v === current), tabindex: v === current ? '0' : '-1' }, label);
      b.addEventListener('click', () => pick(v, true));
      container.appendChild(b);
      return b;
    });
    function pick(v, user) {
      buttons.forEach((b) => { const on = b.dataset.v === String(v); b.setAttribute(attr, String(on)); b.tabIndex = on ? 0 : -1; });
      onPick(v, user);
    }
    if (container._kd) container.removeEventListener('keydown', container._kd);
    container._kd = (e) => {
      const i = buttons.indexOf(document.activeElement); if (i < 0) return;
      let j = -1;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = (i + 1) % buttons.length;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = (i - 1 + buttons.length) % buttons.length;
      if (j >= 0) { e.preventDefault(); buttons[j].focus(); buttons[j].click(); }
    };
    container.addEventListener('keydown', container._kd);
    return { pick };
  }
  /* Wire up a seg whose buttons are already in the HTML. */
  function seg(container, onPick) {
    const buttons = $$('button', container);
    const items = buttons.map((b) => ({ v: b.dataset.v, label: b.innerHTML }));
    const cur = (buttons.find((b) => b.getAttribute('aria-checked') === 'true') || buttons[0]).dataset.v;
    return choice(container, items, cur, onPick);
  }

  function tile(label, cls) {
    const f = el('figure', { class: 'tile' + (cls ? ' ' + cls : '') });
    const m = el('div', { class: 'tile__media' });
    const v = el('video', { muted: true, playsinline: true, loop: true, preload: 'none', 'aria-label': strip(label) });
    m.appendChild(v); f.appendChild(m);
    f.appendChild(el('figcaption', {}, '<span>' + label + '</span>'));
    return { f, v };
  }

  /* ------------------------------------------------------------------ */
  /* Header                                                              */
  /* ------------------------------------------------------------------ */
  (function header() {
    const bar = $('#bar');
    const onScroll = () => bar.classList.toggle('is-stuck', window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
    const links = $$('.bar__nav a');
    const map = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
    const io = new IntersectionObserver((es) => {
      for (const e of es) if (e.isIntersecting) { links.forEach((a) => a.classList.remove('is-here')); const a = map.get(e.target.id); if (a) a.classList.add('is-here'); }
    }, { rootMargin: '-45% 0px -50% 0px' });
    ['asymmetry', 'reconstruction', 'method', 'results', 'numbers', 'ablations', 'cite'].forEach((id) => { const s = document.getElementById(id); if (s) io.observe(s); });
  })();

  /* ------------------------------------------------------------------ */
  /* Hero                                                                */
  /* ------------------------------------------------------------------ */
  (function hero() {
    const root = $('#hero');
    const split = $('[data-split]', root);
    makeSplit(split);
    const [va, vb] = $$('video', split);
    const group = new SyncGroup(split, null);
    const order = [['tiger-lion', 'rf'], ['cow', 'rf'], ['airplane', 'rf'], ['cat-dog', 'comp'], ['opera', 'comp'], ['jellyfish', 'rf'], ['pitcher', 'rf']];
    const promptEl = $('[data-prompt]', root);
    const picker = $('.picker', root);

    // chunk ribbon: 81 frames at 16 fps, 3 latent frames per chunk -> 9, then 12 frames per chunk
    // raw chunk boundaries [0, 9, 21, ..., 81], shifted by the frames cut from the start of every clip
    const bounds = [0, 9, 21, 33, 45, 57, 69, 81].map((b) => Math.max(0, b - TRIM));
    const cellsEl = $('.ribbon__cells', root);
    const cells = bounds.slice(1).map((b, i) => {
      const c = el('div', { class: 'ribbon__cell' }, '<i></i>'); c.style.flex = String(b - bounds[i]); cellsEl.appendChild(c); return c;
    });
    group.onTime((t) => {
      const f = clamp(t * 16, 0, NFRAMES - 0.001);
      let k = 0; while (k < 6 && f >= bounds[k + 1]) k++;
      cells.forEach((c, i) => {
        c.classList.toggle('is-done', i < k);
        c.firstChild.style.transform = i < k ? 'scaleX(1)' : i === k ? 'scaleX(' + ((f - bounds[k]) / (bounds[k + 1] - bounds[k])).toFixed(3) + ')' : 'scaleX(0)';
      });
    });

    const items = order.map(([c, set]) => ({
      v: c,
      label: '<img src="' + poster(D.media[set].clips[c].files.Ours) + '" alt="' + CLIP_NAME[c] + '" loading="lazy" width="832" height="480">',
    }));
    const pick = (c) => {
      const set = order.find((o) => o[0] === c)[1];
      const files = D.media[set].clips[c].files;
      group.set([{ v: va, src: files.Input }, { v: vb, src: files.Ours }]);
      renderPrompt(promptEl, c);
    };
    choice(picker, items, 'tiger-lion', pick, { role: 'tab' });
    pick('tiger-lion');
    // the chunk ribbon doubles as the timeline
    const btn = $('.ribbon__btn', root);
    group.btn = btn; group.updateBtn();
    btn.addEventListener('click', () => { if (group.playing) { group.userPaused = true; group.pause(); } else { group.userPaused = false; group.play(); } });
    let scrubbing = false, resume = false;
    const seekAt = (e) => { const r = cellsEl.getBoundingClientRect(); const d = (va.duration || CLIP_DUR); group.seek(clamp((e.clientX - r.left) / r.width, 0, 0.999) * d); };
    cellsEl.addEventListener('pointerdown', (e) => { scrubbing = true; resume = group.playing; group.pause(); cellsEl.setPointerCapture(e.pointerId); seekAt(e); });
    cellsEl.addEventListener('pointermove', (e) => { if (scrubbing) seekAt(e); });
    const stop = () => { if (!scrubbing) return; scrubbing = false; if (resume) group.play(); };
    cellsEl.addEventListener('pointerup', stop); cellsEl.addEventListener('pointercancel', stop);
  })();

  /* ------------------------------------------------------------------ */
  /* QKV explorer                                                        */
  /* ------------------------------------------------------------------ */
  (function qkvExplorer() {
    const root = $('#qkv');
    const range = $('#noise-range');
    const out = $('#noise-out');
    const ticksEl = $('.noise__ticks', root);
    const settingSeg = $('[data-qkv-setting]', root);
    const noteEl = $('[data-qkv-note]', root);
    const NOTES = {
      rf: 'Reward Forcing, a DMD-distilled autoregressive student of Wan 2.1, at its four training timesteps. Source videos noised to each timestep; mean over 120 benchmark videos.',
      rf_gen: 'Reward Forcing while generating from Gaussian noise with its native sampler, compared with the clean pass; mean over 120 videos.',
      rf_edit: 'Reward Forcing during AR-Edit editing with the default settings; mean over 120 videos.',
      causvid: 'CausVid, a second DMD-distilled autoregressive student, at its three training timesteps; mean over 120 videos.',
      causvid_bidi: 'The bidirectional DMD-distilled CausVid model: no causality, same asymmetry; mean over 120 videos.',
      teacher: 'The non-distilled bidirectional Wan 2.1 1.3B teacher across the full noise range; mean over 120 videos. The gap closes toward low noise, a region the distilled students never sample.',
      all: 'One row per model, every measured timestep in each plot, the slider\u2019s timestep in bold; mean over 120 videos. The big glyphs follow Reward Forcing.',
    };
    // what is being measured, per model / setting (the paragraph under the controls)
    const T0 = '<i>t</i>&nbsp;=&nbsp;0', CMP = 'compare the queries, keys and values of all 30 self-attention blocks with those of the clean pass at ' + T0 + '.';
    const DESC = {
      rf: 'We noise a clean source video to each training timestep of the distilled model, run a single forward pass, and ' + CMP + ' Drag the noise level and watch what survives.',
      rf_gen: 'We generate from pure Gaussian noise with the model&rsquo;s native few-step sampler and, at each denoising timestep, compare the queries, keys and values of all 30 self-attention blocks with those at ' + T0 + '. Drag the timestep and watch what survives.',
      rf_edit: 'We repeat the analysis during editing with AR-Edit. Drag the timestep and watch what survives.',
      causvid: 'The same protocol on CausVid, a second DMD-distilled autoregressive student: we noise a clean source video to each of its three training timesteps, run a single forward pass, and ' + CMP,
      causvid_bidi: 'The same protocol on the bidirectional DMD-distilled CausVid model, which removes causality: we noise a clean source video to each training timestep, run a single forward pass, and ' + CMP,
      teacher: 'The same protocol on the non-distilled bidirectional Wan&nbsp;2.1 teacher, over the full noise range: we noise a clean source video to each timestep, run a single forward pass, and ' + CMP,
      all: 'All four models under the same protocol, one row each: a clean source video noised to every timestep of the model, a single forward pass, and the ' + CMP.replace('compare the', 'comparison of the') + ' Read down a column: <b class="m">Q</b> and <b class="m">K</b> stay high in every model while <b class="m">V</b> drops. Only the teacher is measured below <i>t</i>&nbsp;=&nbsp;522.',
    };
    const descEl = $('[data-qkv-desc]', root);
    let model = 'rf', setting = '', idx = 0, userTouched = false, swept = false;
    const ALL = ['rf', 'causvid', 'causvid_bidi', 'teacher'];
    const ALL_NAME = { rf: 'Reward Forcing', causvid: 'CausVid', causvid_bidi: 'CausVid, bidirectional', teacher: 'Wan 2.1 teacher' };
    const compare = () => model === 'all';
    const key = () => (compare() ? 'rf' : model === 'rf' ? 'rf' + setting : model);
    // the compare view runs over the union of all models' timesteps
    const steps = () => [0].concat(compare()
      ? [...new Set(ALL.flatMap((m) => D.qkv[m].t))].sort((a, b) => a - b)
      : D.qkv[key()].t.slice().reverse());
    const nearestT = (m, t) => (t === 0 ? 0 : D.qkv[m].t.reduce((best, x) => (Math.abs(x - t) < Math.abs(best - t) ? x : best)));
    const seriesOf = (m, f, t) => (t === 0 ? new Array(30).fill(1) : D.qkv[m][f][String(t)]);
    const series = (f, t) => seriesOf(key(), f, compare() ? nearestT('rf', t) : t);
    const gridEl = $('[data-qkv-grid]', root), chartsEl = $('.qkv__charts', root), leftEl = $('.qkv__left', root);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const colorAt = (i, n) => ramp(n <= 0 ? 0 : i / n);

    // glyph filters
    const G = {};
    ['Q', 'K', 'V'].forEach((f) => {
      const filt = document.getElementById('noise-' + f);
      G[f] = {
        disp: filt.querySelector('feDisplacementMap'),
        cm: filt.querySelector('feColorMatrix'),
        text: $('.glyph[data-f="' + f + '"] text', root),
        val: $('.glyph[data-f="' + f + '"] [data-val]', root),
        cur: 1, target: 1,
      };
    });
    function drawGlyph(f, m) {
      const g = G[f], u = 1 - m;
      g.disp.setAttribute('scale', (40 * Math.pow(u, 1.8)).toFixed(2));
      const icpt = 4 - 3.25 * Math.pow(u, 1.25);
      g.cm.setAttribute('values', '0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -2.2 ' + icpt.toFixed(3));
      g.text.style.fill = ramp(clamp(Math.pow(u, 1.4) * 1.05, 0, 1));
    }
    let tween = 0;
    function animateGlyphs() {
      cancelAnimationFrame(tween);
      const t0 = performance.now(), from = {}, dur = RM.matches ? 0 : 380;
      ['Q', 'K', 'V'].forEach((f) => { from[f] = G[f].cur; });
      const step = (now) => {
        const p = dur ? clamp((now - t0) / dur, 0, 1) : 1, e = 1 - Math.pow(1 - p, 3);
        ['Q', 'K', 'V'].forEach((f) => { G[f].cur = lerp(from[f], G[f].target, e); drawGlyph(f, G[f].cur); });
        if (p < 1) tween = requestAnimationFrame(step);
      };
      tween = requestAnimationFrame(step);
    }

    // charts
    // same proportions as the paper's figure (plot area about 1.27 : 1), so V's drop reads at a glance
    const W = 270, H = 230, ml = 40, mr = 8, mt = 10, mb = 38;
    const X = (b) => ml + b / 29 * (W - ml - mr);
    const Y = (v) => mt + (1 - v) * (H - mt - mb);
    const charts = {};
    ['Q', 'K', 'V'].forEach((f) => {
      const svg = $('.chart[data-f="' + f + '"] svg', root);
      charts[f] = { svg, cap: $('.chart[data-f="' + f + '"] figcaption', root) };
    });
    function drawCharts(hoverBlock) {
      const ts = steps(), n = ts.length - 1, cur = ts[idx];
      ['Q', 'K', 'V'].forEach((f) => {
        const { svg } = charts[f];
        svg.innerHTML = '';
        const grid = sv('g', { class: 'grid' }, svg);
        [0, 0.25, 0.5, 0.75, 1].forEach((v) => sv('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v) }, grid));
        const ax = sv('g', { class: 'axis' }, svg);
        [0, 0.5, 1].forEach((v) => { const t = sv('text', { x: ml - 7, y: Y(v) + 3.5, 'text-anchor': 'end' }, ax); t.textContent = v === 0.5 ? '0.5' : String(v); });
        [0, 5, 10, 15, 20, 25, 29].forEach((b) => { const t = sv('text', { x: X(b), y: H - mb + 13, 'text-anchor': 'middle' }, ax); t.textContent = String(b); });
        sv('text', { x: (ml + W - mr) / 2, y: H - 6, 'text-anchor': 'middle', class: 'ttl' }, ax).textContent = 'Block index';
        sv('text', { x: 11, y: (mt + H - mb) / 2, 'text-anchor': 'middle', class: 'ttl', transform: 'rotate(-90 11 ' + ((mt + H - mb) / 2) + ')' }, ax).textContent = 'Cosine similarity';
        const path = (s) => s.map((v, b) => (b ? 'L' : 'M') + X(b).toFixed(1) + ' ' + Y(v).toFixed(1)).join('');
        // faint lines for every measured timestep
        ts.forEach((t, i) => {
          if (t === cur) return;
          const so = series(f, t), co = colorAt(i, n);
          sv('path', { d: path(so), fill: 'none', stroke: co, 'stroke-width': 1.4, opacity: 0.75, 'stroke-linejoin': 'round' }, svg);
          so.forEach((v, b) => sv('circle', { cx: X(b), cy: Y(v), r: 1.5, fill: co, opacity: 0.75 }, svg));
        });
        const s = series(f, cur), c = colorAt(idx, n);
        sv('path', { d: path(s), fill: 'none', stroke: c, 'stroke-width': 2.6, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
        s.forEach((v, b) => sv('circle', { cx: X(b), cy: Y(v), r: 2.1, fill: c }, svg));
        if (hoverBlock != null) {
          const hg = sv('g', { class: 'hover' }, svg);
          sv('line', { x1: X(hoverBlock), x2: X(hoverBlock), y1: mt, y2: H - mb }, hg);
          sv('circle', { cx: X(hoverBlock), cy: Y(s[hoverBlock]), r: 4.5, fill: 'none', stroke: PAL['--ink'], 'stroke-width': 1.5 }, hg);
          const left = hoverBlock > 18;
          const tx = sv('text', { x: X(hoverBlock) + (left ? -7 : 7), y: clamp(Y(s[hoverBlock]) - 9, mt + 10, H - mb - 4), 'text-anchor': left ? 'end' : 'start' }, hg);
          tx.textContent = 'block ' + hoverBlock + ': ' + s[hoverBlock].toFixed(2);
        }
      });
    }
    // compare view: a small Q / K / V plot for every model, all at the slider's timestep
    const ALL_SUB = { rf: 'DMD-distilled, autoregressive', causvid: 'DMD-distilled, autoregressive', causvid_bidi: 'DMD-distilled, bidirectional', teacher: 'not distilled, bidirectional' };
    const GW = 210, GH = 180, gml = 34, gmr = 6, gmt = 10, gmb = 32;
    const GX = (b) => gml + b / 29 * (GW - gml - gmr), GY = (v) => gmt + (1 - v) * (GH - gmt - gmb);
    function drawGrid() {
      // every tile shows all of the model's measured timesteps, like the paper's figure; the slider's timestep is drawn bold
      const cur = steps()[idx];
      gridEl.innerHTML = '';
      gridEl.appendChild(el('div', { class: 'gh' }, 'Model'));
      ['Q', 'K', 'V'].forEach((f) => gridEl.appendChild(el('div', { class: 'gh' }, '<b>' + f + '</b>')));
      ALL.forEach((m) => {
        const tsm = D.qkv[m].t.slice().sort((a, b) => b - a);   // high noise first, as in the paper's legend
        const n = tsm.length;
        const sel = nearestT(m, cur), off = cur !== 0 && sel !== cur;   // the timestep to highlight in this row
        const lab = el('div', { class: 'gm gm--' + m }, '<b>' + ALL_NAME[m] + '</b><small>' + ALL_SUB[m] + '</small>' +
          '<span class="gt">' + (cur === 0 ? 'clean pass, <i>t</i> = 0' : 'bold: <i>t</i> = ' + sel + (off ? ' <small>(nearest to ' + cur + ')</small>' : '')) + '</span>' +
          '<small><i>t</i> &isin; {' + tsm.join(', ') + '}</small>');
        gridEl.appendChild(lab);
        ['Q', 'K', 'V'].forEach((f) => {
          const fig = el('figure');
          const svg = sv('svg', { viewBox: '0 0 ' + GW + ' ' + GH, role: 'img', 'aria-label': ALL_NAME[m] + ', ' + f + ' cosine similarity against block index at every measured timestep' });
          const grid = sv('g', { class: 'grid' }, svg);
          [0, 0.5, 1].forEach((v) => sv('line', { x1: gml, x2: GW - gmr, y1: GY(v), y2: GY(v) }, grid));
          const ax = sv('g', { class: 'axis' }, svg);
          [0, 0.5, 1].forEach((v) => { const t = sv('text', { x: gml - 5, y: GY(v) + 3, 'text-anchor': 'end' }, ax); t.textContent = String(v); });
          [0, 10, 20, 29].forEach((b) => { const t = sv('text', { x: GX(b), y: GH - gmb + 11, 'text-anchor': 'middle' }, ax); t.textContent = String(b); });
          sv('text', { x: (gml + GW - gmr) / 2, y: GH - 5, 'text-anchor': 'middle', class: 'ttl' }, ax).textContent = 'Block index';
          sv('text', { x: 9, y: (gmt + GH - gmb) / 2, 'text-anchor': 'middle', class: 'ttl', transform: 'rotate(-90 9 ' + ((gmt + GH - gmb) / 2) + ')' }, ax).textContent = 'Cosine similarity';
          // one line per timestep in the noise-ramp colours: gray at t = 1000, orange toward the clean source
          const curves = tsm.map((t, i) => ({ t, i, hi: cur !== 0 && t === sel }));
          // the highlighted timestep is drawn last, thicker and at full strength; the others step back
          curves.filter((c) => !c.hi).concat(curves.filter((c) => c.hi)).forEach(({ t, i, hi }) => {
            const s = seriesOf(m, f, t), c = ramp(n <= 1 ? 1 : 1 - i / n), op = cur === 0 ? 0.9 : hi ? 1 : 0.35;
            sv('path', { d: s.map((v, bb) => (bb ? 'L' : 'M') + GX(bb).toFixed(1) + ' ' + GY(v).toFixed(1)).join(''), fill: 'none', stroke: c, 'stroke-width': hi ? 2.6 : 1.4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: op }, svg);
            s.forEach((v, bb) => sv('circle', { cx: GX(bb), cy: GY(v), r: hi ? 2 : 1.2, fill: c, opacity: op }, svg));
          });
          if (cur === 0) sv('line', { x1: gml, x2: GW - gmr, y1: GY(1), y2: GY(1), stroke: PAL['--orange'], 'stroke-width': 2.6, 'stroke-linecap': 'round' }, svg);
          fig.appendChild(svg);
          // the mean at the highlighted timestep
          const mv = cur === 0 ? 1 : mean(seriesOf(m, f, sel));
          fig.appendChild(el('span', { class: 'gv' }, mv.toFixed(2) + '<small>mean, t = ' + (cur === 0 ? 0 : sel) + '</small>'));
          gridEl.appendChild(fig);
        });
      });
      gridEl.appendChild(el('p', { class: 'gnote' }, 'Same protocol for every row: a clean source video noised to each of the model\u2019s timesteps, one forward pass, cosine similarity to the clean pass per block, mean over 120 videos. Every measured timestep is drawn, gray at pure noise (t = 1000) and orange toward the clean source; the slider\u2019s timestep is bold and its mean is shown. When a model was not measured at that exact timestep, its nearest one is highlighted.'));
    }
    let hover = null;
    Object.values(charts).forEach(({ svg }) => {
      svg.addEventListener('pointermove', (e) => {
        const r = svg.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width * W;
        const b = Math.round(clamp((x - ml) / (W - ml - mr) * 29, 0, 29));
        if (b !== hover) { hover = b; drawCharts(hover); }
      });
      svg.addEventListener('pointerleave', () => { hover = null; drawCharts(null); });
    });

    function update(animate) {
      const ts = steps();
      idx = clamp(idx, 0, ts.length - 1);
      const t = ts[idx];
      range.max = String(ts.length - 1); range.value = String(idx);
      out.innerHTML = t === 0 ? '<i>t</i> = 0 (clean)' : '<i>t</i> = ' + t;
      range.setAttribute('aria-valuetext', 't = ' + t);
      ['Q', 'K', 'V'].forEach((f) => {
        const m = mean(series(f, t));
        G[f].target = m; G[f].val.textContent = m.toFixed(2);
      });
      if (animate) animateGlyphs(); else ['Q', 'K', 'V'].forEach((f) => { G[f].cur = G[f].target; drawGlyph(f, G[f].cur); });
      chartsEl.hidden = compare(); gridEl.hidden = !compare();
      if (compare()) drawGrid(); else drawCharts(hover);
      noteEl.textContent = NOTES[compare() ? 'all' : key()];
      descEl.innerHTML = DESC[compare() ? 'all' : key()];
    }
    function buildTicks() {
      const ts = steps(), n = ts.length - 1;
      ticksEl.innerHTML = '';
      ts.forEach((t, i) => {
        const s = el('span', {}, String(t));
        s.style.left = (i / n * 100) + '%';
        if (n > 6 && i % 2 === 1 && i !== n) s.style.visibility = 'hidden';
        ticksEl.appendChild(s);
      });
    }
    range.addEventListener('input', () => { idx = +range.value; update(true); userTouched = true; });
    seg($('[data-qkv-model]', root), (v) => {
      const prevT = steps()[idx];
      model = v; settingSeg.hidden = v !== 'rf';
      if (v === 'all') setting = '';
      const ts = steps();
      // keep the closest timestep when switching models
      let best = 0; ts.forEach((t, i) => { if (Math.abs(t - prevT) < Math.abs(ts[best] - prevT)) best = i; });
      idx = best; buildTicks(); update(true); userTouched = true;
    });
    seg(settingSeg, (v) => { setting = v; update(true); userTouched = true; });
    buildTicks(); update(false);
    onTheme(() => { ['Q', 'K', 'V'].forEach((f) => drawGlyph(f, G[f].cur)); if (compare()) drawGrid(); else drawCharts(hover); });

    // one orchestrated moment: sweep from clean to pure noise the first time the explorer is seen
    new IntersectionObserver((es, obs) => {
      for (const e of es) {
        if (!e.isIntersecting || swept) continue;
        swept = true; obs.disconnect();
        const n = steps().length - 1;
        if (RM.matches) { idx = n; update(false); return; }
        let i = 0;
        const tick = () => { if (userTouched) return; i++; idx = i; update(true); if (i < n) setTimeout(tick, 650); };
        setTimeout(tick, 500);
      }
    }, { threshold: 0.45 }).observe($('.qkv__glyphs', root));
  })();

  /* ------------------------------------------------------------------ */
  /* Reconstruction recipes                                              */
  /* ------------------------------------------------------------------ */
  (function recon() {
    const root = $('#recon');
    const BLUE = (t) => PAL['--c-blue-' + t];
    const PURPLE = () => PAL['--c-purple'], GRAYM = () => PAL['--c-graym'];
    const FLAT = (v) => [v, v, v, v];
    // The recipes of the paper's table. LPIPS per pre-denoising NFE is read from the paper's figure
    // (benchmark mean over 120 videos); `clip` is the figure value for the run whose video is shown.
    const R = [
      { id: 'inversion', name: tupH('Inv', '&ndash;', '&ndash;'), plain: () => tupP('Inv', '\u2013', '\u2013'), shape: 'circle', fill: false, color: () => GRAYM(),
        z: 'Inv', vs: '&ndash;', vt: '&ndash;', img: () => 'inversion', clip: () => 0.31,
        series: [{ panel: 'L', nfe: [1, 2, 3, 4], v: [0.3058, 0.2881, 0.2866, 0.3257] }] },
      { id: 'inv-v', name: tupH('Inv', 'Inv', '{' + Vh('<i>t</i>') + '}'), plain: () => tupP('Inv', 'Inv', '{V' + SUP('t') + '}'), shape: 'diamond', fill: false, color: () => PURPLE(),
        z: 'Inv', vs: 'Inv', vt: '{' + Vh('<i>t</i>') + '}', img: () => 'inv-v', clip: () => 0.04,
        series: [{ panel: 'L', nfe: [1.12, 2, 3, 4], v: [0.0360, 0.0391, 0.0383, 0.0395] }] },
      { id: 'inv-fwd', name: tupH('Inv', 'Fwd', Vh('<i>t</i>')), at: (t) => tupH('Inv', 'Fwd', Vh(t)), plain: (t) => tupP('Inv', 'Fwd', 'V' + SUP(t == null ? 't' : t)), shape: 'square', fill: false,
        color: (t) => BLUE(t), ts: [0, 625, 833, 938, 1000], t0: 0, z: 'Inv', vs: 'Fwd', vt: Vh('<i>t</i>'),
        img: (t) => 'inv-fwd-' + t, clip: (t) => ({ 0: 0.04, 625: 0.07, 833: 0.11, 938: 0.18, 1000: 0.36 })[t],
        series: [{ panel: 'L', t: 0, nfe: [2, 3, 4, 5], v: FLAT(0.0353) }].concat([[0, 0.0353], [625, 0.0682], [833, 0.110], [938, 0.1782], [1000, 0.3615]]
          .map(([t, v]) => ({ panel: 'R', t, nfe: [2, 3, 4, 5], v: FLAT(v) }))) },
      { id: 'rand-v', name: tupH('Rand', 'Inv', '{' + Vh('<i>t</i>') + '}'), plain: () => tupP('Rand', 'Inv', '{V' + SUP('t') + '}'), shape: 'triup', fill: true, color: () => PURPLE(),
        z: 'Rand', vs: 'Inv', vt: '{' + Vh('<i>t</i>') + '}', img: () => 'rand-v-t', clip: () => 0.04,
        series: [{ panel: 'L', nfe: [1, 2, 3, 4], v: [0.0546, 0.0602, 0.0589, 0.0609] }] },
      { id: 'rand-fwd', name: tupH('Rand', 'Fwd', Vh('<i>t</i>')), at: (t) => tupH('Rand', 'Fwd', Vh(t)), plain: (t) => tupP('Rand', 'Fwd', 'V' + SUP(t == null ? 't' : t)), shape: 'tridown', fill: true,
        color: (t) => BLUE(t), ts: [625, 833, 938, 1000], t0: 625, z: 'Rand', vs: 'Fwd', vt: Vh('<i>t</i>'),
        img: (t) => 'rand-fwd-' + t, clip: (t) => ({ 625: 0.07, 833: 0.11, 938: 0.18, 1000: 0.36 })[t],
        series: [[625, 0.0681], [833, 0.110], [938, 0.1782], [1000, 0.3611]].map(([t, v]) => ({ panel: 'R', t, nfe: [1], v: [v] })) },
      { id: 'ours', name: 'Ours: ' + tupH('Rand', 'Fwd', Vh(0)), plain: () => 'Ours: ' + tupP('Rand', 'Fwd', 'V' + SUP(0)), shape: 'star', fill: true, color: () => PAL['--orange'], ours: true,
        z: 'Rand', vs: 'Fwd', vt: Vh(0), img: () => 'ours', clip: () => 0.04,
        series: [{ panel: 'L', nfe: [1], v: [0.0354] }, { panel: 'R', nfe: [1], v: [0.0354] }] },
    ];
    const OURS = 0.0354;
    function marker(shape, color, fill, size, parent, cx, cy, hollow = 'none') {
      const sw = 1.8, f = fill ? color : hollow, r = size / 2;
      const a = { fill: f, stroke: color, 'stroke-width': sw, 'stroke-linejoin': 'round' };
      let node;
      if (shape === 'circle') node = sv('circle', Object.assign({ cx, cy, r: r - 1 }, a), parent);
      else if (shape === 'square') node = sv('rect', Object.assign({ x: cx - r + 1.5, y: cy - r + 1.5, width: size - 3, height: size - 3 }, a), parent);
      else {
        let pts;
        if (shape === 'diamond') pts = [[0, -r], [r, 0], [0, r], [-r, 0]];
        if (shape === 'triup') pts = [[0, -r], [r, r * 0.8], [-r, r * 0.8]];
        if (shape === 'tridown') pts = [[0, r], [r, -r * 0.8], [-r, -r * 0.8]];
        if (shape === 'star') { pts = []; for (let i = 0; i < 10; i++) { const rr = i % 2 ? r * 0.45 : r * 1.05, ang = -Math.PI / 2 + i * Math.PI / 5; pts.push([Math.cos(ang) * rr, Math.sin(ang) * rr]); } }
        node = sv('polygon', Object.assign({ points: pts.map(([x, y]) => (cx + x).toFixed(1) + ',' + (cy + y).toFixed(1)).join(' ') }, a), parent);
      }
      return node;
    }
    const nameEl = $('[data-recon-name]', root), clipEl = $('[data-recon-clip]', root);
    // source and the selected recipe play in sync; switching recipe swaps only the right-hand video
    const vSrc = $('[data-recon-src]', root), vOut = $('[data-recon-out]', root), tile = $('[data-recon-tile]', root);
    const group = new SyncGroup($('.recon__pair', root), $('[data-player="recon"]', root));
    const tWrap = $('[data-recon-t]', root);
    let cur = 'ours', curT = {};
    R.forEach((r) => { if (r.ts) curT[r.id] = r.t0; });

    // ---- the paper's table: one column per recipe, the whole column is clickable
    const table = $('[data-recon-table]', root);
    const ROWS = [['Initial noise', 'z'], ['<b class="m">V</b> source', 'vs'], ['<b class="m">V</b> timestep', 'vt']];
    const oc = (r) => (r.ours ? ' class="is-ours"' : '');
    table.innerHTML = '<thead><tr><th scope="col">Recipe</th>' + R.map((r, i) => '<th scope="col" data-col="' + i + '"' + oc(r) + '><button type="button" class="rtable__pick" data-col="' + i + '" aria-pressed="false"><svg viewBox="0 0 18 18" aria-hidden="true"></svg><span>' + r.name + '</span></button></th>').join('') + '</tr></thead>' +
      '<tbody>' + ROWS.map(([lab, k]) => '<tr><th scope="row">' + lab + '</th>' + R.map((r, i) => '<td data-col="' + i + '"' + oc(r) + '>' + r[k] + '</td>').join('') + '</tr>').join('') + '</tbody>';
    const picks = $$('.rtable__pick', table);
    const drawIcons = () => picks.forEach((b, i) => { const ic = $('svg', b), r = R[i]; ic.innerHTML = ''; marker(r.shape, r.color(r.ts ? r.t0 : 0), r.fill, 14, ic, 9, 9); });
    table.addEventListener('click', (e) => { const c = e.target.closest('[data-col]'); if (c) select(R[+c.dataset.col].id); });
    table.addEventListener('keydown', (e) => {
      const i = picks.indexOf(document.activeElement); if (i < 0 || (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft')) return;
      e.preventDefault(); const j = (i + (e.key === 'ArrowRight' ? 1 : -1) + picks.length) % picks.length; picks[j].focus(); select(R[j].id);
    });

    // ---- the paper's figure: LPIPS (log) against pre-denoising NFE, two panels
    const plots = { L: $('[data-recon-plot="L"]', root), R: $('[data-recon-plot="R"]', root) };
    const TITLE = {
      L: [['{', 0], ['V', 'b'], ['t', 'supi'], ['} needs its inverted noise; ', 0], ['V', 'b'], ['0', 'sup'], [' does not', 0]],
      R: [['Forward-pass ', 0], ['V', 'b'], [': timestep matters, noise does not', 0]],
    };
    const W = 560, H = 360, ml = 58, mr = 12, mt = 44, mb = 48;
    const X = (n) => ml + (n - 0.6) / (5.4 - 0.6) * (W - ml - mr);
    const lo = Math.log10(0.027), hi = Math.log10(0.44);
    const Y = (v) => mt + (hi - Math.log10(v)) / (hi - lo) * (H - mt - mb);
    function title(svg, x, y, segs) {
      svgSegs(sv('text', { x, y, 'text-anchor': 'middle', 'font-size': 15, fill: PAL['--ink'], 'font-weight': 600 }, svg), segs);
    }
    function drawPlot(P) {
      const svg = plots[P]; svg.innerHTML = '';
      title(svg, (ml + W - mr) / 2, 18, TITLE[P]);
      [0.03, 0.05, 0.1, 0.2, 0.4].forEach((v) => {
        sv('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), stroke: PAL['--grid'] }, svg);
        sv('text', { x: ml - 9, y: Y(v) + 4, 'text-anchor': 'end', fill: PAL['--ink-3'], 'font-size': 12 }, svg).textContent = String(v);
      });
      [1, 2, 3, 4, 5].forEach((n) => sv('text', { x: X(n), y: H - mb + 20, 'text-anchor': 'middle', fill: PAL['--ink-3'], 'font-size': 12 }, svg).textContent = String(n));
      sv('text', { x: (ml + W - mr) / 2, y: H - 8, 'text-anchor': 'middle', fill: PAL['--ink-2'], 'font-size': 13 }, svg).textContent = 'Pre-denoising NFE';
      const yl = sv('text', { x: 16, y: (mt + H - mb) / 2, 'text-anchor': 'middle', fill: PAL['--ink-2'], 'font-size': 13, transform: 'rotate(-90 16 ' + ((mt + H - mb) / 2) + ')' }, svg);
      yl.textContent = 'LPIPS ↓ (log)';
      sv('line', { x1: ml, x2: W - mr, y1: Y(OURS), y2: Y(OURS), stroke: PAL['--orange'], 'stroke-dasharray': '5 4', 'stroke-width': 1.2, opacity: 0.7 }, svg);
      // dim everything but the selected recipe; the selected extraction timestep is drawn last, on top
      const items = [];
      R.forEach((r) => r.series.forEach((s) => {
        if (s.panel !== P) return;
        const mine = r.id === cur, on = mine && (!r.ts || s.t === curT[r.id]);
        items.push({ r, s, mine, on, z: r.ours ? 3 : on ? 2 : mine ? 1 : 0 });
      }));
      items.sort((a, b) => a.z - b.z).forEach(({ r, s, mine, on }) => {
        const c = r.color(s.t != null ? s.t : 0);
        // with Ours selected the figure reads as in the paper; otherwise the other recipes step back
        const g = sv('g', { class: 'pt', opacity: cur === 'ours' || on || r.ours ? 1 : mine ? 0.6 : 0.32 }, svg);
        if (s.nfe.length > 1) sv('polyline', { points: s.nfe.map((n, i) => X(n).toFixed(1) + ',' + Y(s.v[i]).toFixed(1)).join(' '), fill: 'none', stroke: c, 'stroke-width': on ? 2.6 : 1.8 }, g);
        s.nfe.forEach((n, i) => {
          const m = marker(r.shape, c, r.fill, r.ours ? (on ? 26 : 22) : on ? 16 : 13, g, X(n), Y(s.v[i]), PAL['--bg-raise']);
          const tt = document.createElementNS(NS, 'title');
          tt.textContent = r.plain(s.t) + ', ' + Math.round(n) + ' NFE: LPIPS ' + s.v[i].toFixed(3);
          m.appendChild(tt);
        });
        if (on && !r.ours) {
          const i0 = 0;
          sv('text', { x: X(s.nfe[i0]) + (P === 'R' && s.nfe.length === 1 ? -14 : 12), y: Y(s.v[i0]) - 10, 'text-anchor': P === 'R' && s.nfe.length === 1 ? 'end' : 'start', fill: PAL['--ink'], 'font-size': 12, 'font-weight': 600 }, g).textContent = s.v[i0].toFixed(3);
        }
        g.addEventListener('click', () => { if (r.ts && s.t != null && r.ts.includes(s.t)) curT[r.id] = s.t; select(r.id); });
      });
    }
    const legend = $('[data-recon-legend]', root);
    function drawLegend() {
      const ic = (shape, fill) => { const s = document.createElementNS(NS, 'svg'); s.setAttribute('viewBox', '0 0 14 14'); s.setAttribute('aria-hidden', 'true'); marker(shape, PAL['--ink-2'], fill, 12, s, 7, 7); return s.outerHTML; };
      legend.innerHTML = '<li>' + ic('circle', false) + 'inverted noise</li><li>' + ic('circle', true) + 'random noise</li>' +
        '<li><span class="m">V</span> extraction timestep <i>t</i>:</li>' + [0, 625, 833, 938, 1000].map((t) => '<li><span class="sw" style="background:var(--c-blue-' + t + ')"></span><span>' + t + '</span></li>').join('');
    }

    function select(id) {
      cur = id;
      const r = R.find((x) => x.id === id), i = R.indexOf(r);
      $$('th[data-col], td[data-col]', table).forEach((c) => c.classList.toggle('is-sel', +c.dataset.col === i));
      picks.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
      const t = r.ts ? curT[id] : null;
      group.set([{ v: vSrc, src: 'recon/source.mp4' }, { v: vOut, src: 'recon/' + r.img(t) + '.mp4' }]);
      vOut.setAttribute('aria-label', 'Reconstruction with ' + r.plain(t));
      tile.classList.toggle('tile--ours', !!r.ours);
      nameEl.innerHTML = t != null && r.at ? r.at(t) : r.name;
      // the figure's LPIPS is the benchmark mean (120 clips), not this clip's value
      clipEl.textContent = 'LPIPS ' + r.clip(t).toFixed(2) + ', benchmark mean';
      tWrap.hidden = !r.ts;
      $$('button', tWrap).forEach((b) => b.remove());
      if (r.ts) r.ts.forEach((tt) => {
        const b = el('button', { type: 'button', 'aria-pressed': String(tt === t) }, String(tt));
        b.addEventListener('click', () => { curT[id] = tt; select(id); });
        tWrap.appendChild(b);
      });
      drawPlot('L'); drawPlot('R');
    }
    drawIcons(); drawLegend();
    select('ours');
    onTheme(() => { drawIcons(); drawLegend(); drawPlot('L'); drawPlot('R'); });
  })();

  /* ------------------------------------------------------------------ */
  /* Method: six passes                                                  */
  /* ------------------------------------------------------------------ */
  (function passes() {
    const root = $('#passes');
    const row = $('.passes__row', root);
    const tEl = $('[data-pd-title]', root), bEl = $('[data-pd-body]', root), eqEl = $('[data-pd-eq]', root);
    const canvas = $('[data-pd-canvas]', root), ctx = canvas.getContext('2d');
    const depth = $('[data-pd-depth]', root), legend = $('[data-pd-legend]', root);
    const ATT = '\\operatorname{Attn} = \\operatorname{softmax}\\!\\Big(\\tfrac{\\mathbf{Q}_{\\mathrm{tar}}\\,\\mathbf{K}_{\\mathrm{tar}}^{\\top}}{\\sqrt{d}}\\Big)\\,\\mathbf{V}';
    const P = [
      { kind: 'clean', t: 'Clean pass', chip: '<i>t</i> = 0', s: 'Source chunk, source prompt. Cache <b class="m">V</b><sub>src</sub>.',
        body: 'When chunk <i>n</i> arrives, we encode it and run a single forward pass of the distilled model at <i>t</i>&nbsp;=&nbsp;0, the clean endpoint, conditioned on the source prompt. During this pass we cache the pre-attention <b class="m">V</b> tensor at every self-attention block <i>l</i>&nbsp;&gt;&nbsp;<i>b</i><sub>min</sub>, and the cross-attention of the source tokens absent from the target prompt at blocks <i>l</i>&nbsp;&le;&nbsp;<i>b</i><sub>CtA</sub>.',
        eq: ['\\mathbf{V}_{\\mathrm{src}}^{(l)} \\leftarrow \\mathbf{V}^{(l)}\\big(\\mathbf{x}_{\\mathrm{src}}^{(n)},\\, t{=}0,\\, c_{\\mathrm{src}}\\big), \\quad l > b_{\\min}'] },
      { kind: 'commit', t: 'Step <i>k</i> = 0', chip: 'Commit', s: 'Random noise, target prompt. Source values at every block <i>l</i>&nbsp;&gt;&nbsp;<i>b</i><sub>min</sub>.',
        body: 'We sample <b>z</b>&nbsp;~&nbsp;<i>N</i>(0,&nbsp;<i>I</i>) and denoise it with the distilled model&rsquo;s predict-then-renoise schedule, conditioned on the target prompt. At the first <i>S</i> steps and at every block <i>l</i>&nbsp;&gt;&nbsp;<i>b</i><sub>min</sub>, the target&rsquo;s value is replaced by the cached source value. The target&rsquo;s <b class="m">Q</b> and <b class="m">K</b> remain untouched, so the attention routing reflects the edit, while the injected <b class="m">V</b> controls what content is read out. Text boost: in every cross-attention layer, the unnormalized attention weights of the target tokens absent from the source prompt are scaled by a constant factor before the softmax.',
        eq: ['\\mathbf{V} \\leftarrow \\mathbf{V}_{\\mathrm{src}} \\quad (k < S = 2)', ATT] },
      { kind: 'commit', t: 'Step <i>k</i> = 1', chip: 'Commit', s: 'Renoised prediction. Source values everywhere again.',
        body: 'The prediction is renoised to the next timestep and denoised again with the same injection: replacing <b class="m">V</b> everywhere acts on the whole frame, so at the first <i>S</i>&nbsp;=&nbsp;2 steps we inject everywhere and commit to the edit.',
        eq: ['\\mathbf{V} \\leftarrow \\mathbf{V}_{\\mathrm{src}} \\quad (k < S = 2)', ATT] },
      { kind: 'anchor', t: 'Step <i>k</i> = 2', chip: 'Anchor', s: 'Mask from cross-attention. Source values outside it.',
        body: 'From step <i>S</i><sub>CtA</sub> onward we inject only outside the edited region. <i>M</i>&nbsp;&isin;&nbsp;[0,&nbsp;1] is a soft mask derived from the cross-attention of the tokens that differ between the source and target prompts, read from the blocks <i>l</i>&nbsp;&le;&nbsp;<i>b</i><sub>CtA</sub> with <i>b</i><sub>CtA</sub>&nbsp;&lt;&nbsp;<i>b</i><sub>min</sub>, so the mask is ready before the injected blocks of the same pass. This anchors everything outside <i>M</i> to the source while the edited region stays free.',
        eq: ['\\mathbf{V} \\leftarrow M \\odot \\mathbf{V} + (1 - M) \\odot \\mathbf{V}_{\\mathrm{src}} \\quad (k \\ge S_{\\mathrm{CtA}} = 2)', ATT] },
      { kind: 'anchor', t: 'Step <i>k</i> = 3', chip: 'Anchor', s: 'Last step. Source values outside the mask.',
        body: 'The last denoising step applies the same masked injection: outside <i>M</i> the source values, inside the target&rsquo;s own. By default <i>S</i><sub>CtA</sub>&nbsp;=&nbsp;<i>S</i>&nbsp;=&nbsp;2 with <i>K</i>&nbsp;=&nbsp;4 steps.',
        eq: ['\\mathbf{V} \\leftarrow M \\odot \\mathbf{V} + (1 - M) \\odot \\mathbf{V}_{\\mathrm{src}}', ATT] },
      { kind: 'ctx', t: 'Context rerun', chip: 'KV cache', s: 'Overwrite the chunk&rsquo;s slot in the cache with clean <b class="m">K</b>, <b class="m">V</b>.',
        body: 'In the autoregressive backbone, each chunk attends to past chunks via a rolling KV cache. The keys and values written into it during the editing pass are the native target-prompt outputs, so future chunks see representations consistent with the target prompt rather than a patchwork of source and target features.',
        eq: [] },
    ];
    // depth bar
    for (let b = 0; b < 30; b++) depth.appendChild(el('i', { title: 'block ' + b }));
    depth.appendChild(el('div', { class: 'depth__axis' }, '<span>block 0</span><span>mask read: 0&ndash;19</span><span>values injected: 23&ndash;29</span><span>29</span>'));
    const cells = $$('i', depth);

    // images for the token view
    const load = (src) => new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = src; });
    const files = D.media.rf_classic.clips.piglet.files;
    let srcImg = null, outImg = null, noise = null;
    const mask = (function () {
      if (!D.maskPiglet) return null;
      const bin = atob(D.maskPiglet); const a = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
      return a; // 21 latent frames x 30 x 52
    })();
    const POSTER_LAT = Math.floor((TRIM + 3) / 4); // the posters show raw frame TRIM
    function makeNoise() {
      const c = document.createElement('canvas'); c.width = 208; c.height = 120;
      const x = c.getContext('2d'), d = x.createImageData(208, 120);
      for (let i = 0; i < d.data.length; i += 4) { const g = 60 + Math.random() * 150; d.data[i] = g; d.data[i + 1] = g * 0.98; d.data[i + 2] = g * 0.95; d.data[i + 3] = 255; }
      x.putImageData(d, 0, 0); return c;
    }
    let cur = 0;
    function draw() {
      const p = P[cur], Wc = canvas.width, Hc = canvas.height;
      ctx.clearRect(0, 0, Wc, Hc);
      ctx.imageSmoothingEnabled = true;
      const bg = (im, a) => { if (im) { ctx.globalAlpha = a; ctx.drawImage(im, 0, 0, Wc, Hc); ctx.globalAlpha = 1; } };
      if (p.kind === 'clean') bg(srcImg, 1);
      else if (cur === 1) { bg(noise, 1); }
      else if (cur === 2) { bg(noise, 1); bg(outImg, 0.55); }
      else if (cur === 3) { bg(noise, 0.35); bg(outImg, 0.8); }
      else bg(outImg, 1);
      // token grid 52 x 30
      const cw = Wc / 52, ch = Hc / 30;
      for (let y = 0; y < 30; y++) for (let x = 0; x < 52; x++) {
        const m = mask ? mask[POSTER_LAT * 1560 + y * 52 + x] / 255 : 0;
        let a = 0;
        if (p.kind === 'commit') a = 0.5;
        if (p.kind === 'anchor') a = 0.5 * (1 - m);
        if (a > 0.01) { ctx.fillStyle = 'rgba(255,122,0,' + a.toFixed(3) + ')'; ctx.fillRect(x * cw + 0.6, y * ch + 0.6, cw - 1.2, ch - 1.2); }
        if (p.kind === 'clean') { ctx.strokeStyle = 'rgba(255,179,122,0.55)'; ctx.lineWidth = 0.8; ctx.strokeRect(x * cw + 1, y * ch + 1, cw - 2, ch - 2); }
        if (p.kind === 'ctx') { ctx.strokeStyle = 'rgba(241,238,233,0.32)'; ctx.lineWidth = 0.8; ctx.strokeRect(x * cw + 1, y * ch + 1, cw - 2, ch - 2); }
      }
      // depth bar state
      cells.forEach((c, b) => {
        c.className = '';
        if (b > 22 && p.kind !== 'ctx') c.classList.add('is-inj');
        if (b <= 19 && (p.kind === 'clean' || p.kind === 'anchor')) c.classList.add('is-mask', 'is-live');
      });
      const L = {
        clean: [['#ffb37a', 'values cached for every token'], ['var(--ink-2)', 'blocks reading cross-attention'], ['var(--orange)', 'blocks whose values are kept']],
        commit: [['#ff7a00', 'token reads source values'], ['var(--orange)', 'blocks receiving source values']],
        anchor: [['#ff7a00', 'outside M: source values'], ['transparent', 'inside M: own values'], ['var(--ink-2)', 'blocks building M']],
        ctx: [['#f1eee9', 'clean keys and values written to the cache']],
      }[p.kind];
      legend.innerHTML = L.map(([c, t]) => '<li><i style="background:' + c + (c === 'transparent' ? ';box-shadow:inset 0 0 0 1.5px var(--ink)' : '') + '"></i>' + t + '</li>').join('');
    }
    function tex(node, list) {
      node.innerHTML = '';
      list.forEach((s) => {
        const d = el('div');
        if (window.katex) { try { window.katex.render(s, d, { displayMode: true, throwOnError: false }); } catch (e) { d.textContent = s; } }
        else d.textContent = s;
        node.appendChild(d);
      });
    }
    function select(i) {
      cur = i; const p = P[i];
      btns.forEach((b, j) => { b.setAttribute('aria-selected', String(j === i)); b.tabIndex = j === i ? 0 : -1; });
      tEl.innerHTML = p.t; bEl.innerHTML = p.body; tex(eqEl, p.eq); draw();
    }
    const btns = P.map((p, i) => {
      const li = el('li');
      const b = el('button', { type: 'button', role: 'tab', class: 'pass pass--' + p.kind, 'aria-selected': 'false' },
        '<span class="pass__n">Pass ' + (i + 1) + ' of 6</span><span class="pass__t">' + p.t + '</span><span class="pass__chip">' + p.chip + '</span><span class="pass__s">' + p.s + '</span>');
      b.addEventListener('click', () => select(i));
      li.appendChild(b); row.appendChild(li); return b;
    });
    row.addEventListener('keydown', (e) => {
      const i = btns.indexOf(document.activeElement); if (i < 0) return;
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault(); const j = (i + (e.key === 'ArrowRight' ? 1 : -1) + 6) % 6; btns[j].focus(); select(j);
      }
    });
    noise = makeNoise();
    select(0);
    Promise.all([load(poster(files.Input)), load(poster(files.Ours))]).then(([a, b]) => { srcImg = a; outImg = b; draw(); });
    // KaTeX may arrive after this runs
    if (!window.katex) window.addEventListener('load', () => select(cur));
  })();

  /* ------------------------------------------------------------------ */
  /* Mask on a real stream                                               */
  /* ------------------------------------------------------------------ */
  (function maskDemo() {
    const root = $('#maskdemo');
    const [vIn, vOut] = $$('video', root);
    const cv = $('.tile__overlay', root), c2 = cv.getContext('2d');
    const toggle = $('[data-mask-toggle]', root);
    const group = new SyncGroup($('.maskdemo__pair', root), $('[data-player="mask"]', root));
    const f = D.media.rf_classic.clips.piglet.files;
    group.set([{ v: vIn, src: f.Input }, { v: vOut, src: f.Ours }]);
    if (!D.maskPiglet) return;
    const bin = atob(D.maskPiglet); const M = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) M[i] = bin.charCodeAt(i);
    const small = document.createElement('canvas'); small.width = 52; small.height = 30;
    const sctx = small.getContext('2d'); const img = sctx.createImageData(52, 30);
    let last = -1;
    function paint(lat) {
      if (lat === last) return; last = lat;
      const off = lat * 1560;
      for (let i = 0; i < 1560; i++) {
        const m = M[off + i] / 255;
        img.data[i * 4] = 255; img.data[i * 4 + 1] = 122; img.data[i * 4 + 2] = 0; img.data[i * 4 + 3] = Math.round((1 - m) * 0.5 * 255);
      }
      sctx.putImageData(img, 0, 0);
      c2.clearRect(0, 0, cv.width, cv.height);
      c2.imageSmoothingEnabled = true; c2.imageSmoothingQuality = 'high';
      c2.drawImage(small, 0, 0, cv.width, cv.height);
    }
    group.onTime((t) => {
      const fr = Math.floor(t * 16) + TRIM; // raw frame index
      const lat = clamp(Math.floor((fr + 3) / 4), 0, 20);
      paint(lat);
    });
    toggle.addEventListener('change', () => { cv.style.opacity = toggle.checked ? '1' : '0'; });
    paint(Math.floor((TRIM + 3) / 4));
  })();

  /* ------------------------------------------------------------------ */
  /* Network-pass budget                                                 */
  /* ------------------------------------------------------------------ */
  (function budget() {
    const rows = $('#budget .budget__rows');
    const B = [
      { n: 'SDEdit', nfe: 5, fps: 11.20 }, { n: 'AR-Edit', nfe: 6, fps: 10.50, ours: true },
      { n: 'FlowEdit', nfe: 9, fps: 6.72 }, { n: 'Inversion w/ V', nfe: 9, fps: 9.07 }, { n: 'DynaEdit', nfe: 13, fps: 4.66 },
    ];
    B.forEach((b) => {
      const r = el('div', { class: 'brow' + (b.ours ? ' brow--ours' : '') });
      r.appendChild(el('span', { class: 'brow__name' }, METHOD[b.n] ? METHOD[b.n].html : b.n));
      const c = el('span', { class: 'brow__cells', 'aria-label': b.nfe + ' network passes' });
      for (let i = 0; i < b.nfe; i++) c.appendChild(el('i', { class: i === b.nfe - 1 ? 'is-ctx' : null }));
      r.appendChild(c);
      r.appendChild(el('span', { class: 'brow__fps' }, '<b>' + b.fps.toFixed(1) + '</b> fps'));
      rows.appendChild(r);
    });
    rows.appendChild(el('p', { class: 'note' }, 'Each square is one forward pass of the network for a chunk; the outlined square is the context rerun. Throughput in frames per second on one H100.'));
  })();

  /* ------------------------------------------------------------------ */
  /* Results: grid and slider                                            */
  /* ------------------------------------------------------------------ */
  (function results() {
    const root = $('#compare');
    const clipsEl = $('[data-clips]', root), promptEl = $('[data-prompt]', root);
    const grid = $('[data-grid]', root), splitwrap = $('[data-splitwrap]', root);
    const splitEl = $('[data-split]', splitwrap); makeSplit(splitEl);
    const [sa, sb] = $$('video', splitEl);
    const leftName = $('[data-left-name]', splitwrap);
    const group = new SyncGroup(root, $('[data-player="compare"]', root));
    const CL = ['tiger-lion', 'cow', 'train', 'jellyfish', 'pitcher', 'piglet', 'rabbit', 'zebra', 'coffee'];
    const setOf = (c) => (D.media.rf.clips[c] ? 'rf' : 'rf_classic');
    const ROWS = ['Input', 'SDEdit', 'FlowEdit', 'DynaEdit', 'Inversion w/ V', 'Ours'];
    let clip = 'tiger-lion', layout = 'grid', left = 'FlowEdit';
    const tiles = ROWS.map((r) => tile(r === 'Ours' ? 'AR-Edit' : METHOD[r].html, r === 'Ours' ? 'tile--ours' : r === 'Input' ? 'tile--input' : ''));
    tiles.forEach((t) => grid.appendChild(t.f));
    function apply() {
      const files = D.media[setOf(clip)].clips[clip].files;
      renderPrompt(promptEl, clip);
      grid.hidden = layout !== 'grid'; splitwrap.hidden = layout !== 'split';
      if (layout === 'grid') group.set(ROWS.map((r, i) => ({ v: tiles[i].v, src: files[r] })));
      else { leftName.innerHTML = METHOD[left].html; group.set([{ v: sa, src: files[left] }, { v: sb, src: files.Ours }]); }
    }
    choice(clipsEl, CL.map((c) => ({ v: c, label: CLIP_NAME[c] })), clip, (v) => { clip = v; apply(); }, { role: 'tab' });
    seg($('[data-layout]', root), (v) => { layout = v; apply(); });
    choice($('[data-left]', splitwrap), ROWS.slice(0, 5).map((r) => ({ v: r, label: METHOD[r].html })), left, (v) => { left = v; apply(); });
    apply();
  })();

  (function streamEdit() {
    const root = $('#streamedit');
    const clipsEl = $('[data-clips]', root), promptEl = $('[data-prompt]', root), grid = $('[data-grid]', root);
    const group = new SyncGroup(root, $('[data-player="se"]', root));
    const ROWS = ['Input', 'StreamEdit', 'Ours'];
    const tiles = ROWS.map((r) => tile(r === 'Ours' ? 'AR-Edit' : r, r === 'Ours' ? 'tile--ours' : r === 'Input' ? 'tile--input' : ''));
    tiles.forEach((t) => grid.appendChild(t.f));
    const CL = Object.keys(D.media.sf.clips);
    const order = ['cow', 'tiger-lion', 'jellyfish', 'pitcher'].filter((c) => CL.includes(c));
    function apply(c) {
      const files = D.media.sf.clips[c].files;
      renderPrompt(promptEl, c);
      group.set(ROWS.map((r, i) => ({ v: tiles[i].v, src: files[r] })));
    }
    choice(clipsEl, order.map((c) => ({ v: c, label: CLIP_NAME[c] })), order[0], apply, { role: 'tab' });
    apply(order[0]);
  })();

  /* ------------------------------------------------------------------ */
  /* Numbers                                                             */
  /* ------------------------------------------------------------------ */
  (function numbers() {
    const tbody = $('[data-table]');
    const note = $('[data-backbone-note]');
    // [method, src, text, vq, clip, tf, tfu, nfe, fps, latency, mem] and paper marks (b = bold, u = underline)
    // [method, src, text, vq, clip, tf, tfu, steps, nfe, fps, latency, mem, marks (b = bold, u = underline, per column), muted]
    const T = {
      rf: { note: 'Reward Forcing: DMD-distilled causal student of Wan 2.1 with EMA-sink attention; four editing steps for every method.', rows: [
        ['SDEdit', '3.81', '6.54', '7.92', '0.81', '0.27', '0.23', '4', '5', '11.20', '0.91', '25.7 / 28.9', '-b-----bbbb'],
        ['FlowEdit', '6.77', '5.98', '7.89', '0.89', '0.38', '0.34', '4', '9', '6.72', '1.22', '30.8 / 34.2', '---u-----u-'],
        ['DynaEdit', '7.17', '5.96', '7.85', '0.89', '0.39', '0.36', '4', '13', '4.66', '1.63', '30.8 / 34.2', 'u--buu-----'],
        ['Inversion w/ V', '4.93', '5.85', '7.93', '0.85', '0.38', '0.32', '4', '9', '9.07', '1.49', '27.3 / 31.0', '--u--------'],
        ['Ours', '7.28', '6.11', '7.96', '0.89', '0.47', '0.41', '4', '6', '10.50', '1.22', '27.2 / 30.5', 'bub-bb-uuuu'],
      ] },
      causvid: { note: 'CausVid: DMD-distilled causal student of Wan 2.1; three editing steps for every method.', rows: [
        ['SDEdit', '2.78', '6.45', '8.15', '0.79', '0.25', '0.20', '3', '4', '15.30', '0.91', '29.5 / 34.0', '--b----bbbb'],
        ['FlowEdit', '3.98', '6.88', '7.70', '0.81', '0.36', '0.28', '3', '7', '11.30', '1.37', '35.2 / 39.6', '-b---------'],
        ['DynaEdit', '4.83', '6.82', '7.67', '0.83', '0.38', '0.31', '3', '11', '8.08', '1.73', '35.2 / 39.6', 'uu-uuu-----'],
        ['Inversion w/ V', '3.06', '6.42', '8.12', '0.81', '0.29', '0.23', '3', '7', '11.16', '1.38', '31.1 / 35.4', '--u-------u'],
        ['Ours', '6.99', '6.67', '8.01', '0.84', '0.42', '0.39', '3', '5', '11.91', '1.27', '31.1 / 35.4', 'b--bbb-uuuu'],
      ] },
      sf: { note: 'Self Forcing: the native backbone of the concurrent StreamEdit, run with four editing steps to match our budget and all other hyperparameters as in its paper. The gray row is StreamEdit at its default 15 editing steps and is not ranked.', rows: [
        ['StreamEdit', '8.11', '5.56', '7.91', '0.91', '0.42', '0.39', '15', '32', '2.27', '3.22', '41.3 / 43.4', '-----------', true],
        ['StreamEdit', '7.17', '5.87', '7.56', '0.88', '0.42', '0.38', '4', '10', '6.12', '1.38', '41.3 / 42.8', 'ubuuuu-uuuu'],
        ['Ours', '7.22', '5.77', '7.91', '0.89', '0.46', '0.42', '4', '6', '10.25', '0.85', '31.8 / 35.0', 'bubbbb-bbbb'],
      ] },
      bidir: { note: 'The non-distilled bidirectional Wan 2.1 1.3B with 45 editing steps over the whole clip. This is the offline setting the velocity-based editors were designed for, not a streaming one.', rows: [
        ['SDEdit', '5.01', '6.12', '7.98', '0.84', '0.39', '0.31', '45', '45', '0.85', '94.9', '24.8 / 28.3', '--b----bbbb'],
        ['FlowEdit', '7.12', '6.22', '7.86', '0.88', '0.49', '0.44', '45', '90', '0.45', '181.8', '24.9 / 31.1', 'buuubu-u--u'],
        ['DynaEdit', '6.97', '6.26', '7.80', '0.88', '0.48', '0.45', '45', '94', '0.42', '195.1', '25.0 / 31.1', 'ub-uub-----'],
        ['Inversion w/ V', '6.89', '5.13', '7.77', '0.89', '0.42', '0.35', '45', '90', '0.52', '155.5', '27.8 / 30.3', '---b---uuu-'],
      ] },
    };
    function render(k) {
      const d = T[k]; note.textContent = d.note;
      tbody.innerHTML = d.rows.map((r) => {
        const marks = r[12], muted = !!r[13];
        const m = METHOD[r[0]];
        const cells = r.slice(1, 12).map((v, i) => {
          const c = marks[i] === 'b' ? 'is-best' : marks[i] === 'u' ? 'is-second' : '';
          const g = (i === 2 || i === 6) ? ' gstart' : '';
          return '<td class="' + c + g + '">' + v + '</td>';
        }).join('');
        return '<tr class="' + (r[0] === 'Ours' ? 'is-ours' : muted ? 'is-muted' : '') + '"><td><span class="sw" style="background:' + m.css + '"></span>' + m.html + '</td>' + cells + '</tr>';
      }).join('');
    }
    seg($('[data-backbone]'), render);
    render('rf');
    // header group starts
    const ths = $$('.results thead tr:last-child th');
    [2, 6].forEach((i) => ths[i] && ths[i].classList.add('gstart'));

    /* scatter: source vs text adherence over 2, 3, 4 editing steps (Reward Forcing) */
    const S = {
      'SDEdit': [[3.85, 3.53], [4.22, 4.79], [3.81, 6.54]],
      'FlowEdit': [[8.62, 4.74], [8.29, 5.10], [6.77, 5.98]],
      'DynaEdit': [[8.74, 4.71], [8.34, 5.08], [7.17, 5.96]],
      'Inversion w/ V': [[5.89, 5.47], [5.58, 5.51], [5.75, 5.40]],
      'Ours': [[6.93, 5.93], [7.12, 5.98], [7.28, 6.11]],
    };
    const svg = $('[data-scatter]');
    const W = 560, H = 400, ml = 44, mr = 14, mt = 12, mb = 44;
    const FULL = { x: [3.5, 9], y: [3, 7], dx: 1, dy: 1, f: 0 }, ZOOM = { x: [6.5, 7.5], y: [5.6, 6.3], dx: 0.2, dy: 0.1, f: 1 };
    const LBL = { 'SDEdit': [10, 4, 'start'], 'FlowEdit': [-12, -10, 'end'], 'DynaEdit': [12, 18, 'start'], 'Inversion w/ V': [0, -16, 'middle'], 'Ours': [12, -10, 'start'] };
    function drawScatter(dom) {
      svg.innerHTML = '';
      const X = (v) => ml + (v - dom.x[0]) / (dom.x[1] - dom.x[0]) * (W - ml - mr);
      const Y = (v) => mt + (dom.y[1] - v) / (dom.y[1] - dom.y[0]) * (H - mt - mb);
      const cp = sv('clipPath', { id: 'sc-clip' }, sv('defs', {}, svg));
      sv('rect', { x: ml, y: mt, width: W - ml - mr, height: H - mt - mb }, cp);
      for (let v = dom.x[0]; v <= dom.x[1] + 1e-9; v += dom.dx) {
        if (!dom.f && v < 4) continue;
        sv('line', { x1: X(v), x2: X(v), y1: mt, y2: H - mb, stroke: PAL['--grid'] }, svg);
        sv('text', { x: X(v), y: H - mb + 18, 'text-anchor': 'middle', fill: PAL['--ink-3'], 'font-size': 12 }, svg).textContent = dom.f ? v.toFixed(1) : String(Math.round(v));
      }
      for (let v = dom.y[0]; v <= dom.y[1] + 1e-9; v += dom.dy) {
        sv('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), stroke: PAL['--grid'] }, svg);
        sv('text', { x: ml - 10, y: Y(v) + 4, 'text-anchor': 'end', fill: PAL['--ink-3'], 'font-size': 12 }, svg).textContent = dom.f ? v.toFixed(1) : String(Math.round(v));
      }
      sv('text', { x: (ml + W - mr) / 2, y: H - 6, 'text-anchor': 'middle', fill: PAL['--ink-2'], 'font-size': 13 }, svg).textContent = 'Source adherence ↑';
      const yl = sv('text', { x: 14, y: (mt + H - mb) / 2, 'text-anchor': 'middle', fill: PAL['--ink-2'], 'font-size': 13, transform: 'rotate(-90 14 ' + ((mt + H - mb) / 2) + ')' }, svg); yl.textContent = 'Text adherence ↑';
      const plot = sv('g', { 'clip-path': 'url(#sc-clip)' }, svg);
      const inside = (p) => p[0] >= dom.x[0] && p[0] <= dom.x[1] && p[1] >= dom.y[0] && p[1] <= dom.y[1];
      Object.entries(S).forEach(([k, pts]) => {
        const c = METHOD[k].color, ours = k === 'Ours';
        sv('path', { d: pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(''), fill: 'none', stroke: c, 'stroke-width': ours ? 2.6 : 1.6, 'stroke-dasharray': ours ? '' : '5 4', opacity: 0.9 }, plot);
        pts.forEach((p, i) => {
          const cc = sv('circle', { cx: X(p[0]), cy: Y(p[1]), r: (3.6 + i * 2.2) * (dom.f ? 1.25 : 1), fill: c, stroke: PAL['--bg-raise'], 'stroke-width': 1.5 }, plot);
          const tt = document.createElementNS(NS, 'title'); tt.textContent = METHOD[k].name + ', ' + (i + 2) + ' steps: source ' + p[0].toFixed(2) + ', text ' + p[1].toFixed(2); cc.appendChild(tt);
          if (dom.f && inside(p)) sv('text', { x: X(p[0]), y: Y(p[1]) + (k === 'DynaEdit' ? 22 : -13), 'text-anchor': 'middle', fill: PAL['--ink-2'], 'font-size': 11 }, plot).textContent = (i + 2) + ' steps';
        });
        const lp = dom.f ? (pts.slice().reverse().find(inside) || null) : pts[2];
        if (!lp) return;
        const [dx, dy, anchor] = dom.f ? { 'FlowEdit': [0, 26, 'middle'], 'DynaEdit': [14, 4, 'start'], 'Ours': [14, -6, 'start'] }[k] || [10, 4, 'start'] : LBL[k];
        svgSegs(sv('text', { x: X(lp[0]) + dx, y: Y(lp[1]) + dy, 'text-anchor': anchor, fill: ours ? PAL['--orange-2'] : c, 'font-size': 13, 'font-weight': ours ? 700 : 550 }, plot), METHOD[k].svg);
      });
    }
    let dom = FULL;
    drawScatter(dom);
    const zb = $('[data-zoom]');
    zb.addEventListener('click', () => {
      const on = zb.getAttribute('aria-pressed') !== 'true';
      zb.setAttribute('aria-pressed', String(on)); zb.textContent = on ? 'Show all methods' : 'Zoom into the cluster';
      dom = on ? ZOOM : FULL; drawScatter(dom);
    });
    onTheme(() => drawScatter(dom));
    $('[data-scatter-legend]').innerHTML = '<li><svg width="34" height="14" aria-hidden="true"><circle cx="5" cy="7" r="3.2" style="fill:var(--ink-2)"/><circle cx="15" cy="7" r="4.8" style="fill:var(--ink-2)"/><circle cx="27" cy="7" r="6.4" style="fill:var(--ink-2)"/></svg>2, 3 and 4 editing steps</li>';

    /* user study: best-pick rate per criterion */
    const U = {
      'Source adherence': [3.9, 20.3, 20.8, 13.3, 41.7],
      'Text adherence': [19.2, 21.7, 15.8, 13.3, 30.0],
      'Video quality': [9.4, 21.4, 17.4, 19.5, 32.3],
      'Tonal fidelity': [6.4, 17.4, 16.8, 16.7, 42.7],
    };
    const UM = ['SDEdit', 'FlowEdit', 'DynaEdit', 'Inversion w/ V', 'Ours'];
    const bars = $('[data-bars]');
    bars.innerHTML = Object.entries(U).map(([crit, vals]) => '<div class="bars__group"><h4>' + crit + '</h4>' + vals.map((v, i) => {
      const m = METHOD[UM[i]];
      return '<div class="bars__row' + (UM[i] === 'Ours' ? ' is-ours' : '') + '"><span title="' + m.name + '">' + m.html + '</span><span class="bars__track"><span class="bars__fill" style="width:' + (v / 50 * 100).toFixed(1) + '%;background:' + m.css + '"></span></span><b>' + Math.round(v) + '%</b></div>';
    }).join('') + '</div>').join('');
  })();

  /* ------------------------------------------------------------------ */
  /* Ablations                                                           */
  /* ------------------------------------------------------------------ */
  (function ablations() {
    const root = $('#abl');
    const findingEl = $('[data-abl-finding]', root), clipsEl = $('[data-clips]', root), promptEl = $('[data-prompt]', root), grid = $('[data-grid]', root);
    const group = new SyncGroup(root, $('[data-player="abl"]', root));
    const F = {
      sched: 'Masked injection from the very first step re-renders the scene around the edit; full injection on three of four steps reproduces the source and drops the edit. With S = 2, the edit’s layout forms under full injection and the last two steps keep the rest anchored.',
      blocks: 'Injecting source values from early blocks leaves the source nearly unchanged; injecting only in the last blocks strengthens the edit at the cost of source appearance. The default b_min = 22, b_CtA = 19 sits between the two.',
      comp: 'Without commit-then-anchor the edit spreads beyond its target; without the text boost it is only partly applied. The full method changes the subject and the sky and leaves the rest alone.',
      boost: 'At s = 0 the edit is incomplete, s = 4 completes it, and s = 8 adds little beyond it.',
    };
    const fmt = (s) => s.replace(/S = 2/g, '<i>S</i>&nbsp;=&nbsp;2').replace(/b_min = 22/g, '<i>b</i><sub>min</sub>&nbsp;=&nbsp;22').replace(/b_CtA = 19/g, '<i>b</i><sub>CtA</sub>&nbsp;=&nbsp;19').replace(/s = (\d)/g, '<i>s</i>&nbsp;=&nbsp;$1');
    const LAB = (r) => (D.labels[r] || r).replace(/(^|[ ,(])S(?= |<)/g, '$1<i>S</i>').replace(/(^|[ ,])S<sub>/g, '$1<i>S</i><sub>').replace(/(^|[ ,])b<sub>/g, '$1<i>b</i><sub>');
    let which = 'sched', tiles = [];
    function build(clip) {
      const set = D.media[which];
      const rows = set.rows;
      grid.className = rows.length === 4 ? 'row5 row4' : 'row5';
      grid.innerHTML = '';
      tiles = rows.map((r) => {
        const isDef = r === set.default;
        let label = r === 'Input' ? 'Input' : r === 'Ours' ? 'AR-Edit, full' : LAB(r);
        if (isDef && r !== 'Ours') label += ' <small>default</small>';
        const t = tile(label, isDef ? 'tile--ours' : r === 'Input' ? 'tile--input' : '');
        grid.appendChild(t.f); return t;
      });
      const files = set.clips[clip].files;
      renderPrompt(promptEl, clip);
      group.set(rows.map((r, i) => ({ v: tiles[i].v, src: files[r] })));
    }
    let clipCtl = null;
    function pickSet(v) {
      which = v;
      findingEl.innerHTML = fmt(F[v]);
      const clips = Object.keys(D.media[v].clips);
      clipCtl = choice(clipsEl, clips.map((c) => ({ v: c, label: CLIP_NAME[c] || c })), clips[0], (c) => build(c), { role: 'tab' });
      build(clips[0]);
    }
    seg($('[data-abl]', root), pickSet);
    pickSet('sched');
  })();

  /* ------------------------------------------------------------------ */
  /* Copy BibTeX                                                         */
  /* ------------------------------------------------------------------ */
  $$('[data-copy]').forEach((b) => b.addEventListener('click', async () => {
    const txt = $(b.dataset.copy).textContent;
    try { await navigator.clipboard.writeText(txt); b.textContent = 'Copied'; }
    catch (e) {
      const r = document.createRange(); r.selectNodeContents($(b.dataset.copy));
      const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = 'Selected, press Ctrl+C';
    }
    setTimeout(() => { b.textContent = 'Copy BibTeX'; }, 2200);
  }));
})();
