/* MindScreen front-end: no inline scripts (CSP). Drag-and-drop, confirmations, mobile menu, Chart.js charts. */
(function () {
  "use strict";
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const COLORS = ["#0f9d9a", "#2563eb", "#d99a12", "#7c3aed", "#d6453d"];

  // mobile sidebar
  $$(".menu-btn").forEach(b => b.addEventListener("click", () => $$(".sidebar").forEach(s => s.classList.toggle("open"))));

  $$("[data-print]").forEach(b => b.addEventListener("click", () => window.print()));

  // confirmation dialogs
  $$("form[data-confirm]").forEach(f => f.addEventListener("submit", e => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));

  // drag & drop upload
  $$(".drop").forEach(z => {
    const input = z.querySelector("input[type=file]"), label = z.querySelector(".filename");
    if (!input) return;
    const show = () => { if (label) label.textContent = input.files.length ? input.files[0].name : ""; };
    ["dragenter", "dragover"].forEach(ev => z.addEventListener(ev, e => { e.preventDefault(); z.classList.add("over"); }));
    ["dragleave", "drop"].forEach(ev => z.addEventListener(ev, e => { e.preventDefault(); z.classList.remove("over"); }));
    z.addEventListener("drop", e => { if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(); } });
    input.addEventListener("change", show);
  });

  // busy state on prediction button
  $$("form[data-busy]").forEach(f => f.addEventListener("submit", () => { const b = f.querySelector("button"); if (b) { b.disabled = true; b.textContent = "Working…"; } }));

  if (typeof Chart === "undefined") return;
  Chart.defaults.font.family = "Inter, system-ui, sans-serif"; Chart.defaults.color = "#62708a";

  function lineChart(canvas, chart) {
    const ds = chart.datasets.map((d, i) => ({ label: d.label, data: d.data, borderColor: COLORS[i % 5], backgroundColor: COLORS[i % 5] + "22",
      yAxisID: d.axis === "y2" ? "y2" : "y", spanGaps: false, tension: .25, pointRadius: 2, borderWidth: 2 }));
    const hasY2 = chart.datasets.some(d => d.axis === "y2");
    const scales = { x: { ticks: { maxTicksLimit: 8 }, grid: { display: false } }, y: { beginAtZero: true, grid: { color: "#eef2f6" } } };
    if (hasY2) scales.y2 = { position: "right", beginAtZero: true, grid: { drawOnChartArea: false } };
    return new Chart(canvas, { type: "line", data: { labels: chart.labels, datasets: ds },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } }, scales } });
  }

  function renderCharts(container, payload) {
    container.innerHTML = "";
    payload.charts.forEach(c => {
      const w = document.createElement("div"); w.className = "chartwrap";
      const h = document.createElement("h3"); h.textContent = c.title; w.appendChild(h);
      if (!c.has_data) { const p = document.createElement("p"); p.className = "mute small"; p.textContent = "No data submitted for this category in the selected range."; w.appendChild(p); container.appendChild(w); return; }
      const box = document.createElement("div"); box.className = "chartbox"; const cv = document.createElement("canvas"); box.appendChild(cv); w.appendChild(box);
      const m = document.createElement("div"); m.className = "meta";
      m.textContent = "Gaps are shown as breaks, never filled in. Missing days: " + c.datasets.map(d => d.label + " " + d.missing).join(" · ");
      w.appendChild(m); container.appendChild(w); lineChart(cv, c);
    });
  }

  // trends (student) + overview mini-chart
  $$("[data-trends-src]").forEach(el => {
    const form = document.getElementById(el.dataset.form || "");
    const load = () => {
      let url = el.dataset.trendsSrc;
      if (form) url += (url.includes("?") ? "&" : "?") + new URLSearchParams(new FormData(form)).toString().replace(/csrf=[^&]*&?/, "");
      fetch(url, { credentials: "same-origin" }).then(r => r.ok ? r.json() : Promise.reject(r.status)).then(p => renderCharts(el, p))
        .catch(() => { el.innerHTML = '<p class="mute">Could not load the chart data.</p>'; });
    };
    if (form) form.addEventListener("submit", e => { e.preventDefault(); load(); });
    load();
  });

  // researcher aggregate trends
  $$("[data-aggregate-src]").forEach(el => {
    fetch(el.dataset.aggregateSrc, { credentials: "same-origin" }).then(r => r.json()).then(p => {
      const withData = p.series.filter(s => s.points.length);
      if (!withData.length) { el.innerHTML = '<p class="mute">Not enough students per day (minimum cohort ' + p.min_cohort + ') to display aggregate trends.</p>'; return; }
      const cv = document.createElement("canvas"); el.appendChild(cv);
      new Chart(cv, { type: "line", data: { datasets: withData.map((s, i) => ({ label: s.label, data: s.points, borderColor: COLORS[i % 5], tension: .25, pointRadius: 1, borderWidth: 2, yAxisID: i === 2 ? "y2" : "y" })) },
        options: { parsing: { xAxisKey: "x", yAxisKey: "y" }, responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } },
          scales: { x: { type: "category", ticks: { maxTicksLimit: 8 } }, y: { beginAtZero: true }, y2: { position: "right", grid: { drawOnChartArea: false } } } } });
    });
  });

  // SHAP bar chart + global importance
  $$("canvas[data-bars]").forEach(cv => {
    const items = JSON.parse(cv.dataset.bars), signed = cv.dataset.signed === "1";
    new Chart(cv, { type: "bar", data: { labels: items.map(i => i.label), datasets: [{ data: items.map(i => i.value),
      backgroundColor: items.map(i => signed ? (i.value >= 0 ? "#d6453d" : "#0f9d9a") : "#2563eb") }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: cv.dataset.xlabel || "" }, grid: { color: "#eef2f6" } }, y: { grid: { display: false } } } } });
  });
})();
