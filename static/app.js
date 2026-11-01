/**
 * Loaf Ledger - Modern Vanilla JavaScript Frontend
 * Handles Plan, Today, Reports, and Items screens via plain fetch calls to /api endpoints.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    currentTab: "plan",
    theme: "light",
    items: [],
    todayDate: new Date().toISOString().split("T")[0],
    tomorrowDate: getTomorrowDateStr(),
    planDate: getTomorrowDateStr(),
    reportDays: 7,
  };

  function getTomorrowDateStr() {
    const d = new Date();
    d.setDate(d.getDate() + 1);
    return d.toISOString().split("T")[0];
  }

  // DOM Elements
  const elements = {
    themeToggleBtn: document.getElementById("theme-toggle-btn"),
    navTabBtns: document.querySelectorAll(".nav-tab-btn, .mobile-nav-btn"),
    screens: document.querySelectorAll(".tab-screen"),
    weatherWidget: document.getElementById("weather-widget"),
    weatherRefreshBtn: document.getElementById("weather-refresh-btn"),
    toastContainer: document.getElementById("toast-container"),

    // Plan Screen
    planDateInput: document.getElementById("plan-date-input"),
    planTomorrowBtn: document.getElementById("plan-tomorrow-btn"),
    planRefreshBtn: document.getElementById("plan-refresh-btn"),
    planContainer: document.getElementById("plan-container"),
    planWeatherBanner: document.getElementById("plan-weather-banner"),

    // Today Screen
    todayForm: document.getElementById("today-log-form"),
    todayItemSelect: document.getElementById("today-item-select"),
    todayDateInput: document.getElementById("today-date-input"),
    todaySoldInput: document.getElementById("today-sold-input"),
    todayBakedInput: document.getElementById("today-baked-input"),
    todayWastedInput: document.getElementById("today-wasted-input"),
    todaySoldOutCheckbox: document.getElementById("today-sold-out-checkbox"),
    todaySubmitBtn: document.getElementById("today-submit-btn"),
    recentLogsContainer: document.getElementById("recent-logs-container"),
    refreshLogsBtn: document.getElementById("refresh-logs-btn"),

    // Reports Screen
    reportPeriodSelect: document.getElementById("report-period-select"),
    reportsRefreshBtn: document.getElementById("reports-refresh-btn"),
    reportsKpisContainer: document.getElementById("reports-kpis-container"),
    reportsChartWrapper: document.getElementById("reports-chart-wrapper"),
    reportsTableContainer: document.getElementById("reports-table-container"),

    // Items Screen
    itemsContainer: document.getElementById("items-container"),
    addItemBtn: document.getElementById("add-item-btn"),
    itemModal: document.getElementById("item-modal"),
    itemModalTitle: document.getElementById("item-modal-title"),
    itemModalForm: document.getElementById("item-modal-form"),
    itemIdInput: document.getElementById("item-id-input"),
    itemNameInput: document.getElementById("item-name-input"),
    itemStockInput: document.getElementById("item-stock-input"),
    itemShelfLifeInput: document.getElementById("item-shelf-life-input"),
    itemModalCloseBtn: document.getElementById("item-modal-close-btn"),
    itemModalCancelBtn: document.getElementById("item-modal-cancel-btn"),
    itemModalSaveBtn: document.getElementById("item-modal-save-btn"),
  };

  // -------------------------------------------------------------------------
  // Helper: Toast Notifications
  // -------------------------------------------------------------------------
  function showToast(message, type = "info") {
    if (!elements.toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    const icon = type === "success" ? "✓" : type === "error" ? "⚠️" : "ℹ️";
    toast.innerHTML = `<span>${icon}</span> <span>${escapeHtml(message)}</span>`;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // -------------------------------------------------------------------------
  // Helper: Theme Management
  // -------------------------------------------------------------------------
  function initTheme() {
    const saved = localStorage.getItem("loaf_theme");
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme = saved ? saved : prefersDark ? "dark" : "light";
    setTheme(initialTheme);

    if (elements.themeToggleBtn) {
      elements.themeToggleBtn.addEventListener("click", () => {
        setTheme(state.theme === "dark" ? "light" : "dark");
      });
    }
  }

  function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("loaf_theme", theme);
    if (elements.themeToggleBtn) {
      elements.themeToggleBtn.innerHTML = theme === "dark" ? "☀️" : "🌙";
      elements.themeToggleBtn.setAttribute(
        "title",
        theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
      );
    }
  }

  // -------------------------------------------------------------------------
  // Helper: Tab Navigation & Routing
  // -------------------------------------------------------------------------
  function initNavigation() {
    elements.navTabBtns.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const tab = btn.dataset.tab;
        switchTab(tab);
      });
    });

    window.addEventListener("hashchange", () => {
      const hashTab = window.location.hash.replace("#", "");
      if (["plan", "today", "reports", "items"].includes(hashTab)) {
        switchTab(hashTab, false);
      }
    });

    const initialTab = window.location.hash.replace("#", "");
    if (["plan", "today", "reports", "items"].includes(initialTab)) {
      switchTab(initialTab, false);
    } else {
      switchTab("plan", true);
    }
  }

  function switchTab(tabName, updateHash = true) {
    state.currentTab = tabName;
    if (updateHash) {
      window.location.hash = `#${tabName}`;
    }

    elements.navTabBtns.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });

    elements.screens.forEach((screen) => {
      screen.classList.toggle("active", screen.id === `screen-${tabName}`);
    });

    // Load data for screen
    if (tabName === "plan") {
      loadPlan(state.planDate);
    } else if (tabName === "today") {
      loadTodayScreen();
    } else if (tabName === "reports") {
      loadReports();
    } else if (tabName === "items") {
      loadItems();
    }
  }

  // -------------------------------------------------------------------------
  // Header Weather
  // -------------------------------------------------------------------------
  async function loadHeaderWeather() {
    if (!elements.weatherWidget) return;
    try {
      const res = await fetch(`/api/weather?date=${state.tomorrowDate}`);
      if (!res.ok) {
        elements.weatherWidget.innerHTML = `
          <span class="weather-icon">☁️</span>
          <span class="weather-temp">Weather offline</span>
          <button id="weather-refresh-btn" class="weather-refresh-btn" title="Refresh weather data">↻</button>
        `;
        setupWeatherRefreshBtn();
        return;
      }
      const data = await res.json();
      renderHeaderWeather(data);
    } catch {
      elements.weatherWidget.innerHTML = `
        <span class="weather-icon">☁️</span>
        <span class="weather-temp">Weather offline</span>
        <button id="weather-refresh-btn" class="weather-refresh-btn" title="Refresh weather data">↻</button>
      `;
      setupWeatherRefreshBtn();
    }
  }

  function renderHeaderWeather(w) {
    const temp = w.temp_mean_c !== null && w.temp_mean_c !== undefined ? `${w.temp_mean_c}°C` : "—";
    const condition = w.is_rainy ? "Rainy" : "Dry";
    const rain = w.precip_mm ? ` (${w.precip_mm} mm)` : "";
    const icon = w.is_rainy ? "🌧️" : "☀️";

    elements.weatherWidget.innerHTML = `
      <span class="weather-icon">${icon}</span>
      <span class="weather-temp">Tomorrow: ${temp}</span>
      <span class="weather-condition">· ${condition}${rain}</span>
      <button id="weather-refresh-btn" class="weather-refresh-btn" title="Refresh Open-Meteo weather">↻</button>
    `;
    setupWeatherRefreshBtn();
  }

  function setupWeatherRefreshBtn() {
    const btn = document.getElementById("weather-refresh-btn");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      btn.classList.add("spinning");
      try {
        const res = await fetch("/api/weather/refresh", { method: "POST" });
        if (res.ok) {
          showToast("Weather data refreshed from Open-Meteo", "success");
          await loadHeaderWeather();
          if (state.currentTab === "plan") {
            loadPlan(state.planDate);
          }
        } else {
          showToast("Failed to refresh weather. Using cache.", "error");
        }
      } catch {
        showToast("Network error refreshing weather.", "error");
      } finally {
        btn.classList.remove("spinning");
      }
    });
  }

  // -------------------------------------------------------------------------
  // SCREEN 1: PLAN (Tomorrow's Bake Plan & Demand Forecast)
  // -------------------------------------------------------------------------
  function initPlanScreen() {
    if (elements.planDateInput) {
      elements.planDateInput.value = state.planDate;
      elements.planDateInput.addEventListener("change", (e) => {
        state.planDate = e.target.value || state.tomorrowDate;
        loadPlan(state.planDate);
      });
    }

    if (elements.planTomorrowBtn) {
      elements.planTomorrowBtn.addEventListener("click", () => {
        state.planDate = state.tomorrowDate;
        if (elements.planDateInput) elements.planDateInput.value = state.tomorrowDate;
        loadPlan(state.planDate);
      });
    }

    if (elements.planRefreshBtn) {
      elements.planRefreshBtn.addEventListener("click", () => {
        loadPlan(state.planDate);
      });
    }
  }

  async function loadPlan(targetDate) {
    if (!elements.planContainer) return;
    renderPlanLoading();

    try {
      const res = await fetch(`/api/forecast?date=${encodeURIComponent(targetDate)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        renderPlanError(err?.error?.message || "Could not load the bake forecast. Please try again.");
        return;
      }
      const data = await res.json();
      renderPlanContent(data, targetDate);
    } catch {
      renderPlanError("Network error connecting to the forecast service.");
    }
  }

  function renderPlanLoading() {
    elements.planContainer.innerHTML = `
      <div class="state-box">
        <div class="spinner"></div>
        <div class="state-title">Calculating bake plan...</div>
        <div class="state-desc">Retrieving historical sales patterns and local weather forecasts.</div>
      </div>
      <div class="grid-2">
        <div class="card skeleton skeleton-card"></div>
        <div class="card skeleton skeleton-card"></div>
      </div>
    `;
    if (elements.planWeatherBanner) elements.planWeatherBanner.style.display = "none";
  }

  function renderPlanError(msg) {
    elements.planContainer.innerHTML = `
      <div class="state-box">
        <div class="state-icon">⚠️</div>
        <div class="state-title">Could not generate bake plan</div>
        <div class="state-desc">${escapeHtml(msg)}</div>
        <button class="btn btn-primary" onclick="window.LoafLedger.reloadPlan()">Try Again</button>
      </div>
    `;
    if (elements.planWeatherBanner) elements.planWeatherBanner.style.display = "none";
  }

  function renderPlanContent(data, targetDate) {
    const forecasts = data.forecasts || [];

    if (forecasts.length === 0) {
      elements.planContainer.innerHTML = `
        <div class="state-box">
          <div class="state-icon">🥐</div>
          <div class="state-title">No bakery items found</div>
          <div class="state-desc">Add your bakery items in the Items screen to start generating daily bake plans.</div>
          <button class="btn btn-primary" onclick="window.LoafLedger.goToTab('items')">Go to Items</button>
        </div>
      `;
      if (elements.planWeatherBanner) elements.planWeatherBanner.style.display = "none";
      return;
    }

    // Weather Banner
    const firstW = forecasts[0]?.weather || {};
    const wSource = forecasts[0]?.weather_source || "forecast";
    const temp = firstW.temp_mean_c !== null && firstW.temp_mean_c !== undefined ? `${firstW.temp_mean_c}°C` : "—";
    const condition = firstW.is_rainy ? "🌧️ Rainy" : "☀️ Dry";
    const rainMm = firstW.precip_mm ? ` (${firstW.precip_mm} mm)` : "";
    const sourceLabel = wSource === "fallback" ? "Seasonal Fallback Weather" : "Live Weather Forecast";
    const sourceBadge = wSource === "fallback" ? "badge-alert" : "badge-neutral";

    if (elements.planWeatherBanner) {
      elements.planWeatherBanner.style.display = "flex";
      elements.planWeatherBanner.innerHTML = `
        <div class="info-banner-icon">${firstW.is_rainy ? "🌧️" : "☀️"}</div>
        <div style="flex:1;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
            <strong>Target Date: ${escapeHtml(targetDate)}</strong>
            <span class="badge ${sourceBadge}">${escapeHtml(sourceLabel)}</span>
          </div>
          <div style="margin-top:0.25rem;">
            Expected weather: <strong>${temp}</strong> · ${condition}${rainMm}.
            ${wSource === "fallback" ? "<em>Live forecast unavailable; seasonal historical temperature average is applied.</em>" : ""}
          </div>
        </div>
      `;
    }

    // Item Cards
    let html = `<div class="grid-2">`;
    for (const f of forecasts) {
      const pred = Math.round(f.predicted_qty * 10) / 10;
      const stock = f.current_stock ?? 0;
      const bakeQty = f.suggested_bake_qty ?? Math.max(0, Math.round(pred - stock));
      const rangeStr = f.range_str || `${Math.max(0, Math.round(pred * 0.85))}–${Math.round(pred * 1.15)}`;

      const bakeClass = bakeQty > 0 ? "bake" : "sufficient";
      const bakeText = bakeQty > 0 ? `Bake ${bakeQty} units` : "Stock is sufficient (0 needed)";
      const bakeIcon = bakeQty > 0 ? "🥖" : "✓";

      // Warnings
      let warningsHtml = "";
      if (f.warnings && f.warnings.length > 0) {
        for (const w of f.warnings) {
          warningsHtml += `<div class="warning-box">⚠️ ${escapeHtml(w)}</div>`;
        }
      } else {
        if (f.cold_start) {
          warningsHtml += `<div class="warning-box">⚠️ New item: Under 60 days of sales history recorded. Using 7-day rolling average.</div>`;
        }
        if (f.lag_features_stale) {
          warningsHtml += `<div class="warning-box">⏳ Recent sales not logged: The last logged sales entry is more than 1 day old.</div>`;
        }
        if (f.weather_source === "fallback") {
          warningsHtml += `<div class="warning-box">☁️ Weather offline: Live forecast unavailable; using seasonal temperature average.</div>`;
        }
      }

      html += `
        <div class="card plan-item-card">
          <div>
            <div class="plan-card-header">
              <div>
                <div class="plan-item-name">${escapeHtml(f.name)}</div>
                <div style="font-size:0.78rem; color:var(--text-muted); margin-top:2px;">
                  Shelf life: ${f.shelf_life_days ?? 1} ${f.shelf_life_days === 1 ? "day" : "days"}
                </div>
              </div>
              <div class="stock-tag">In Stock: <strong>${stock}</strong></div>
            </div>

            <div class="bake-suggestion-banner ${bakeClass}">
              <span>${bakeIcon} ${bakeText}</span>
              <span style="font-size:0.8rem; font-weight:normal; opacity:0.85;">Based on stock: ${stock}</span>
            </div>

            <div class="plan-metrics-box">
              <div>
                <div class="metric-label">Predicted Demand</div>
                <div class="metric-val-lg">${pred} <span style="font-size:0.9rem; font-weight:normal;">units</span></div>
              </div>
              <div>
                <div class="metric-label">Expected Range</div>
                <div class="metric-val-lg">${escapeHtml(rangeStr)}</div>
              </div>
            </div>

            <div class="plan-reason-text">
              <strong>Why this number?</strong> ${escapeHtml(f.reason || "Estimated from day of week, weather, and recent sales trends.")}
            </div>
          </div>

          <div>
            ${warningsHtml}
          </div>
        </div>
      `;
    }
    html += `</div>`;
    elements.planContainer.innerHTML = html;
  }

  // -------------------------------------------------------------------------
  // SCREEN 2: TODAY (Log Production & Sales, View Recent Logs)
  // -------------------------------------------------------------------------
  function initTodayScreen() {
    if (elements.todayDateInput) {
      elements.todayDateInput.value = state.todayDate;
    }

    if (elements.todayForm) {
      elements.todayForm.addEventListener("submit", handleTodayFormSubmit);
    }

    if (elements.refreshLogsBtn) {
      elements.refreshLogsBtn.addEventListener("click", loadDailyLogs);
    }
  }

  async function loadTodayScreen() {
    await populateItemSelect();
    await loadDailyLogs();
  }

  async function populateItemSelect() {
    if (!elements.todayItemSelect) return;
    try {
      const res = await fetch("/api/items");
      if (!res.ok) return;
      const items = await res.json();
      state.items = items;

      const currentVal = elements.todayItemSelect.value;
      let html = `<option value="">Select an item...</option>`;
      items.forEach((item) => {
        html += `<option value="${item.id}">${escapeHtml(item.name)} (Stock: ${item.current_stock})</option>`;
      });
      elements.todayItemSelect.innerHTML = html;
      if (currentVal && items.some((i) => String(i.id) === String(currentVal))) {
        elements.todayItemSelect.value = currentVal;
      }
    } catch {
      // ignore
    }
  }

  async function handleTodayFormSubmit(e) {
    e.preventDefault();
    const itemId = elements.todayItemSelect.value;
    const dateVal = elements.todayDateInput.value;
    const soldVal = elements.todaySoldInput.value.trim();
    const bakedVal = elements.todayBakedInput.value.trim();
    const wastedVal = elements.todayWastedInput.value.trim();
    const soldOutVal = elements.todaySoldOutCheckbox.checked;

    if (!itemId) {
      showToast("Please choose an item to log.", "error");
      return;
    }
    if (!dateVal) {
      showToast("Please choose a date.", "error");
      return;
    }

    const payload = {
      item_id: parseInt(itemId, 10),
      date: dateVal,
      sold_out: soldOutVal,
    };

    if (soldVal !== "") payload.sold = parseInt(soldVal, 10);
    if (bakedVal !== "") payload.baked = parseInt(bakedVal, 10);
    if (wastedVal !== "") payload.wasted = parseInt(wastedVal, 10);

    elements.todaySubmitBtn.disabled = true;
    elements.todaySubmitBtn.innerHTML = `Saving...`;

    try {
      const res = await fetch("/api/daily-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showToast(err?.error?.message || "Failed to save daily log.", "error");
        return;
      }

      const itemName = elements.todayItemSelect.options[elements.todayItemSelect.selectedIndex]?.text || "item";
      showToast(`Logged day for ${itemName}. Tomorrow's bake plan has been updated!`, "success");

      // Reset numeric inputs
      elements.todaySoldInput.value = "";
      elements.todayBakedInput.value = "";
      elements.todayWastedInput.value = "";
      elements.todaySoldOutCheckbox.checked = false;

      await loadDailyLogs();
      await populateItemSelect();
    } catch {
      showToast("Network error submitting daily log.", "error");
    } finally {
      elements.todaySubmitBtn.disabled = false;
      elements.todaySubmitBtn.innerHTML = `Save Daily Log`;
    }
  }

  async function loadDailyLogs() {
    if (!elements.recentLogsContainer) return;
    elements.recentLogsContainer.innerHTML = `
      <div class="state-box">
        <div class="spinner"></div>
        <div class="state-title">Loading recent logs...</div>
      </div>
    `;

    try {
      const res = await fetch("/api/daily-log");
      if (!res.ok) {
        elements.recentLogsContainer.innerHTML = `
          <div class="state-box">
            <div class="state-icon">⚠️</div>
            <div class="state-title">Could not load recent logs</div>
            <button class="btn btn-secondary btn-sm" onclick="window.LoafLedger.reloadLogs()">Retry</button>
          </div>
        `;
        return;
      }

      const logs = await res.json();
      renderDailyLogsTable(logs);
    } catch {
      elements.recentLogsContainer.innerHTML = `
        <div class="state-box">
          <div class="state-icon">⚠️</div>
          <div class="state-title">Network error loading logs</div>
          <button class="btn btn-secondary btn-sm" onclick="window.LoafLedger.reloadLogs()">Retry</button>
        </div>
      `;
    }
  }

  function renderDailyLogsTable(logs) {
    if (!logs || logs.length === 0) {
      elements.recentLogsContainer.innerHTML = `
        <div class="state-box">
          <div class="state-icon">📋</div>
          <div class="state-title">No daily logs recorded yet</div>
          <div class="state-desc">Use the form above to log sales and production numbers for today.</div>
        </div>
      `;
      return;
    }

    const recent = logs.slice(0, 20);
    let html = `
      <div class="table-responsive">
        <table class="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Item</th>
              <th>Sold</th>
              <th>Baked</th>
              <th>Wasted</th>
              <th>Sold Out</th>
            </tr>
          </thead>
          <tbody>
    `;

    for (const log of recent) {
      const soldBadge = log.sold !== null ? `<strong>${log.sold}</strong>` : `<span style="color:var(--text-muted)">—</span>`;
      const bakedBadge = log.baked !== null ? `${log.baked}` : `<span style="color:var(--text-muted)">—</span>`;
      const wastedBadge = log.wasted !== null && log.wasted > 0
        ? `<span class="badge badge-alert">${log.wasted}</span>`
        : log.wasted === 0 ? "0" : `<span style="color:var(--text-muted)">—</span>`;

      const soldOutBadge = log.sold_out
        ? `<span class="badge badge-accent">Yes</span>`
        : `<span class="badge badge-neutral">No</span>`;

      html += `
        <tr>
          <td><strong>${escapeHtml(log.date)}</strong></td>
          <td><strong>${escapeHtml(log.item_name || "Item #" + log.item_id)}</strong></td>
          <td>${soldBadge}</td>
          <td>${bakedBadge}</td>
          <td>${wastedBadge}</td>
          <td>${soldOutBadge}</td>
        </tr>
      `;
    }

    html += `
          </tbody>
        </table>
      </div>
    `;

    elements.recentLogsContainer.innerHTML = html;
  }

  // -------------------------------------------------------------------------
  // SCREEN 3: REPORTS (Forecast vs Actual for the last 7 days + Inline SVG Chart)
  // -------------------------------------------------------------------------
  function initReportsScreen() {
    if (elements.reportPeriodSelect) {
      elements.reportPeriodSelect.addEventListener("change", (e) => {
        state.reportDays = parseInt(e.target.value, 10) || 7;
        loadReports();
      });
    }

    if (elements.reportsRefreshBtn) {
      elements.reportsRefreshBtn.addEventListener("click", loadReports);
    }
  }

  async function loadReports() {
    if (!elements.reportsKpisContainer || !elements.reportsChartWrapper || !elements.reportsTableContainer) return;

    elements.reportsKpisContainer.innerHTML = `<div class="card skeleton skeleton-card" style="height:100px; grid-column:1/-1;"></div>`;
    elements.reportsChartWrapper.innerHTML = `
      <div class="state-box">
        <div class="spinner"></div>
        <div class="state-title">Calculating forecast accuracy...</div>
        <div class="state-desc">Comparing historical bake forecasts against actual registered sales.</div>
      </div>
    `;
    elements.reportsTableContainer.innerHTML = "";

    try {
      const res = await fetch(`/api/reports/forecast-vs-actual?days=${state.reportDays}`);
      if (!res.ok) {
        renderReportsError("Failed to calculate report numbers.");
        return;
      }
      const data = await res.json();
      renderReportsContent(data);
    } catch {
      renderReportsError("Network error loading report data.");
    }
  }

  function renderReportsError(msg) {
    elements.reportsKpisContainer.innerHTML = "";
    elements.reportsChartWrapper.innerHTML = `
      <div class="state-box">
        <div class="state-icon">⚠️</div>
        <div class="state-title">Could not load reports</div>
        <div class="state-desc">${escapeHtml(msg)}</div>
        <button class="btn btn-primary" onclick="window.LoafLedger.reloadReports()">Try Again</button>
      </div>
    `;
    elements.reportsTableContainer.innerHTML = "";
  }

  function renderReportsContent(data) {
    const summary = data.summary || {};
    const daily = data.daily || [];
    const records = data.records || [];

    // 1. KPI Cards
    const maeText = summary.mae !== null ? `${summary.mae} <span style="font-size:0.85rem; font-weight:normal;">units avg</span>` : "—";
    elements.reportsKpisContainer.innerHTML = `
      <div class="kpi-card">
        <div class="kpi-title">Actual Sold</div>
        <div class="kpi-value">${summary.total_sold ?? 0}</div>
        <div class="kpi-sub">Total units past ${data.period?.days ?? 7} days</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Forecasted Demand</div>
        <div class="kpi-value">${summary.total_forecast ?? 0}</div>
        <div class="kpi-sub">Predicted across all items</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Total Wasted</div>
        <div class="kpi-value" style="color:var(--alert);">${summary.total_wasted ?? 0}</div>
        <div class="kpi-sub">Spoiled or discarded units</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Forecast Error (MAE)</div>
        <div class="kpi-value" style="color:var(--accent);">${maeText}</div>
        <div class="kpi-sub">Mean absolute variance per item</div>
      </div>
    `;

    // 2. Inline SVG Chart
    renderSvgForecastChart(daily);

    // 3. Comparison Table
    renderReportsTable(records);
  }

  function renderSvgForecastChart(dailyData) {
    if (!dailyData || dailyData.length === 0) {
      elements.reportsChartWrapper.innerHTML = `
        <div class="state-box">
          <div class="state-icon">📊</div>
          <div class="state-title">No data recorded for this period</div>
          <div class="state-desc">Generate forecasts and record daily sales to view the accuracy comparison chart.</div>
        </div>
      `;
      return;
    }

    // Chart dimensions
    const width = 800;
    const height = 280;
    const padding = { top: 30, right: 25, bottom: 40, left: 55 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Find max value
    let maxVal = 10;
    dailyData.forEach((d) => {
      maxVal = Math.max(maxVal, d.forecast_qty || 0, d.actual_sold || 0, d.wasted_qty || 0);
    });
    maxVal = Math.ceil(maxVal * 1.15); // Add headroom

    // Scales
    const n = dailyData.length;
    const groupW = chartW / n;
    const barW = Math.max(8, Math.min(24, groupW * 0.26));
    const gap = 3;

    function getY(val) {
      return padding.top + chartH - (val / maxVal) * chartH;
    }

    function getH(val) {
      return (val / maxVal) * chartH;
    }

    // Grid lines & Y ticks
    let gridLinesSvg = "";
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const val = Math.round((maxVal / yTicks) * i);
      const y = getY(val);
      gridLinesSvg += `
        <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="var(--chart-grid)" stroke-dasharray="${i === 0 ? "none" : "3,3"}" />
        <text x="${padding.left - 8}" y="${y + 4}" font-size="11" fill="var(--text-muted)" text-anchor="end">${val}</text>
      `;
    }

    // Bars & X labels
    let barsSvg = "";
    dailyData.forEach((d, idx) => {
      const groupCenter = padding.left + idx * groupW + groupW / 2;
      const xForecast = groupCenter - barW - gap / 2;
      const xActual = groupCenter + gap / 2;
      const xDateLabel = groupCenter;

      const fH = getH(d.forecast_qty || 0);
      const aH = getH(d.actual_sold || 0);
      const fY = getY(d.forecast_qty || 0);
      const aY = getY(d.actual_sold || 0);

      // Bars
      barsSvg += `
        <g class="chart-group">
          <!-- Forecast bar (Amber) -->
          <rect x="${xForecast}" y="${fY}" width="${barW}" height="${fH}" fill="var(--chart-forecast)" rx="3">
            <title>Forecast: ${d.forecast_qty} units on ${d.date}</title>
          </rect>
          <!-- Actual Sold bar (Crust) -->
          <rect x="${xActual}" y="${aY}" width="${barW}" height="${aH}" fill="var(--chart-actual)" rx="3">
            <title>Actual Sold: ${d.actual_sold} units on ${d.date}</title>
          </rect>
          <!-- X Axis Date Label -->
          <text x="${xDateLabel}" y="${height - 12}" font-size="11" fill="var(--text-secondary)" text-anchor="middle" font-weight="500">
            ${escapeHtml(d.day_name)} ${escapeHtml(d.date.slice(5))}
          </text>
        </g>
      `;
    });

    const svg = `
      <div class="chart-svg-wrapper">
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
          ${gridLinesSvg}
          ${barsSvg}
        </svg>
      </div>
    `;

    elements.reportsChartWrapper.innerHTML = svg;
  }

  function renderReportsTable(records) {
    if (!records || records.length === 0) {
      elements.reportsTableContainer.innerHTML = `
        <div class="state-box">
          <div class="state-title">No daily records for this period</div>
          <div class="state-desc">When you generate bake forecasts and record daily sales, the breakdown table will populate here.</div>
        </div>
      `;
      return;
    }

    let html = `
      <h3 style="margin: 1.5rem 0 0.8rem; font-family:var(--font-serif); font-size:1.15rem;">Daily Breakdown (Forecast vs Actual)</h3>
      <div class="table-responsive">
        <table class="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Item</th>
              <th>Forecast</th>
              <th>Actual Sold</th>
              <th>Variance</th>
              <th>Baked</th>
              <th>Wasted</th>
              <th>Sold Out</th>
            </tr>
          </thead>
          <tbody>
    `;

    for (const r of records) {
      const fText = r.forecast_qty !== null ? `<strong>${r.forecast_qty}</strong>` : `<span style="color:var(--text-muted)">—</span>`;
      const sText = r.actual_sold !== null ? `<strong>${r.actual_sold}</strong>` : `<span style="color:var(--text-muted)">—</span>`;
      const bText = r.baked_qty !== null ? `${r.baked_qty}` : `<span style="color:var(--text-muted)">—</span>`;
      const wText = r.wasted_qty !== null && r.wasted_qty > 0
        ? `<span class="badge badge-alert">${r.wasted_qty}</span>`
        : r.wasted_qty === 0 ? "0" : `<span style="color:var(--text-muted)">—</span>`;

      let diffBadge = `<span style="color:var(--text-muted)">—</span>`;
      if (r.difference !== null) {
        if (r.difference > 0) {
          diffBadge = `<span class="badge badge-accent">+${r.difference} (under-forecast)</span>`;
        } else if (r.difference < 0) {
          diffBadge = `<span class="badge badge-alert">${r.difference} (over-forecast)</span>`;
        } else {
          diffBadge = `<span class="badge badge-ok">Exact match (0)</span>`;
        }
      }

      const soldOutBadge = r.sold_out
        ? `<span class="badge badge-accent">Yes</span>`
        : `<span class="badge badge-neutral">No</span>`;

      html += `
        <tr>
          <td><strong>${escapeHtml(r.date)}</strong></td>
          <td>${escapeHtml(r.item_name)}</td>
          <td>${fText}</td>
          <td>${sText}</td>
          <td>${diffBadge}</td>
          <td>${bText}</td>
          <td>${wText}</td>
          <td>${soldOutBadge}</td>
        </tr>
      `;
    }

    html += `
          </tbody>
        </table>
      </div>
    `;

    elements.reportsTableContainer.innerHTML = html;
  }

  // -------------------------------------------------------------------------
  // SCREEN 4: ITEMS (List, Add, Edit, Delete through /api/items)
  // -------------------------------------------------------------------------
  function initItemsScreen() {
    if (elements.addItemBtn) {
      elements.addItemBtn.addEventListener("click", openAddItemModal);
    }

    if (elements.itemModalCloseBtn) {
      elements.itemModalCloseBtn.addEventListener("click", closeItemModal);
    }

    if (elements.itemModalCancelBtn) {
      elements.itemModalCancelBtn.addEventListener("click", closeItemModal);
    }

    if (elements.itemModalForm) {
      elements.itemModalForm.addEventListener("submit", handleItemFormSubmit);
    }
  }

  async function loadItems() {
    if (!elements.itemsContainer) return;
    elements.itemsContainer.innerHTML = `
      <div class="state-box">
        <div class="spinner"></div>
        <div class="state-title">Loading bakery items...</div>
      </div>
    `;

    try {
      const res = await fetch("/api/items");
      if (!res.ok) {
        renderItemsError("Failed to retrieve bakery items.");
        return;
      }
      const items = await res.json();
      state.items = items;
      renderItemsGrid(items);
    } catch {
      renderItemsError("Network error loading items.");
    }
  }

  function renderItemsError(msg) {
    elements.itemsContainer.innerHTML = `
      <div class="state-box">
        <div class="state-icon">⚠️</div>
        <div class="state-title">Could not load items</div>
        <div class="state-desc">${escapeHtml(msg)}</div>
        <button class="btn btn-primary" onclick="window.LoafLedger.reloadItems()">Try Again</button>
      </div>
    `;
  }

  function renderItemsGrid(items) {
    if (!items || items.length === 0) {
      elements.itemsContainer.innerHTML = `
        <div class="state-box">
          <div class="state-icon">🥐</div>
          <div class="state-title">No bakery items yet</div>
          <div class="state-desc">Add your first item to start forecasting demand and tracking inventory.</div>
          <button class="btn btn-primary" onclick="window.LoafLedger.openAddItem()">Add First Item</button>
        </div>
      `;
      return;
    }

    let html = `<div class="grid-2">`;
    for (const item of items) {
      html += `
        <div class="card" style="display:flex; flex-direction:column; justify-content:space-between;">
          <div>
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.75rem;">
              <h3 style="font-family:var(--font-serif); font-size:1.25rem; font-weight:600; color:var(--text-primary); margin:0;">
                ${escapeHtml(item.name)}
              </h3>
              <span class="stock-tag">ID: #${item.id}</span>
            </div>

            <div class="plan-metrics-box">
              <div>
                <div class="metric-label">Current Stock</div>
                <div class="metric-val-lg">${item.current_stock ?? 0} <span style="font-size:0.85rem; font-weight:normal;">units</span></div>
              </div>
              <div>
                <div class="metric-label">Shelf Life</div>
                <div class="metric-val-lg">${item.shelf_life_days ?? 1} <span style="font-size:0.85rem; font-weight:normal;">${item.shelf_life_days === 1 ? "day" : "days"}</span></div>
              </div>
            </div>
          </div>

          <div style="display:flex; justify-content:flex-end; gap:0.6rem; margin-top:0.75rem;">
            <button class="btn btn-secondary btn-sm" onclick="window.LoafLedger.openEditItem(${item.id})">
              ✏️ Edit
            </button>
            <button class="btn btn-danger btn-sm" onclick="window.LoafLedger.confirmDeleteItem(${item.id}, '${escapeHtml(item.name)}')">
              🗑️ Delete
            </button>
          </div>
        </div>
      `;
    }
    html += `</div>`;
    elements.itemsContainer.innerHTML = html;
  }

  function openAddItemModal() {
    elements.itemModalTitle.innerText = "Add New Bakery Item";
    elements.itemIdInput.value = "";
    elements.itemNameInput.value = "";
    elements.itemStockInput.value = "0";
    elements.itemShelfLifeInput.value = "1";
    elements.itemModal.classList.add("active");
    elements.itemNameInput.focus();
  }

  function openEditItemModal(itemId) {
    const item = state.items.find((i) => i.id === itemId);
    if (!item) return;

    elements.itemModalTitle.innerText = `Edit ${item.name}`;
    elements.itemIdInput.value = item.id;
    elements.itemNameInput.value = item.name;
    elements.itemStockInput.value = item.current_stock ?? 0;
    elements.itemShelfLifeInput.value = item.shelf_life_days ?? 1;
    elements.itemModal.classList.add("active");
    elements.itemNameInput.focus();
  }

  function closeItemModal() {
    elements.itemModal.classList.remove("active");
  }

  async function handleItemFormSubmit(e) {
    e.preventDefault();
    const id = elements.itemIdInput.value;
    const name = elements.itemNameInput.value.trim();
    const stock = parseInt(elements.itemStockInput.value, 10);
    const shelfLife = parseInt(elements.itemShelfLifeInput.value, 10);

    if (!name) {
      showToast("Item name is required.", "error");
      return;
    }
    if (isNaN(stock) || stock < 0) {
      showToast("Current stock must be a non-negative integer.", "error");
      return;
    }
    if (isNaN(shelfLife) || shelfLife < 1) {
      showToast("Shelf life must be at least 1 day.", "error");
      return;
    }

    const payload = {
      name,
      current_stock: stock,
      shelf_life_days: shelfLife,
    };

    elements.itemModalSaveBtn.disabled = true;
    elements.itemModalSaveBtn.innerText = "Saving...";

    try {
      const url = id ? `/api/items/${id}` : "/api/items";
      const method = id ? "PUT" : "POST";

      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showToast(err?.error?.message || "Failed to save item.", "error");
        return;
      }

      showToast(`Item '${name}' ${id ? "updated" : "created"} successfully.`, "success");
      closeItemModal();
      await loadItems();
      await populateItemSelect();
    } catch {
      showToast("Network error saving item.", "error");
    } finally {
      elements.itemModalSaveBtn.disabled = false;
      elements.itemModalSaveBtn.innerText = "Save Item";
    }
  }

  async function confirmDeleteItem(id, name) {
    if (!confirm(`Are you sure you want to delete '${name}'?\n\nThis will remove the item and its forecasts.`)) {
      return;
    }

    try {
      const res = await fetch(`/api/items/${id}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showToast(err?.error?.message || "Failed to delete item.", "error");
        return;
      }
      showToast(`Item '${name}' deleted.`, "success");
      await loadItems();
      await populateItemSelect();
    } catch {
      showToast("Network error deleting item.", "error");
    }
  }

  // -------------------------------------------------------------------------
  // Global API Exposure for inline onclick handlers
  // -------------------------------------------------------------------------
  window.LoafLedger = {
    reloadPlan: () => loadPlan(state.planDate),
    reloadLogs: loadDailyLogs,
    reloadReports: loadReports,
    reloadItems: loadItems,
    goToTab: (tab) => switchTab(tab),
    openAddItem: openAddItemModal,
    openEditItem: (id) => openEditItemModal(id),
    confirmDeleteItem,
  };

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initNavigation();
    initPlanScreen();
    initTodayScreen();
    initReportsScreen();
    initItemsScreen();
    loadHeaderWeather();
  });
})();

