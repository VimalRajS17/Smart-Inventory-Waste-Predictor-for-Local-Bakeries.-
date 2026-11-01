/**
 * Loaf Ledger - Shared JavaScript API & UI Helper
 * Provides robust error handling, loading states, empty states, toasts, and theme management.
 */

(function () {
  "use strict";

  const LoafAPI = {
    /**
     * Shared fetch helper with standardized JSON error parsing and network resilience.
     */
    async fetch(url, options = {}) {
      const method = (options.method || "GET").toUpperCase();
      const defaultHeaders = {
        "Accept": "application/json",
      };
      if (options.body && typeof options.body === "string") {
        defaultHeaders["Content-Type"] = "application/json";
      }

      // Add CSRF token for state-changing requests (POST, PUT, DELETE, PATCH)
      if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
        const csrfMeta = document.querySelector('meta[name="csrf-token"]');
        if (csrfMeta) {
          const csrfToken = csrfMeta.getAttribute("content");
          if (csrfToken) {
            defaultHeaders["X-CSRFToken"] = csrfToken;
          }
        }
      }
      options.headers = { ...defaultHeaders, ...(options.headers || {}) };

      try {
        const response = await window.fetch(url, options);
        let data = null;
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          data = await response.json().catch(() => null);
        } else {
          data = await response.text().catch(() => null);
        }

        if (!response.ok) {
          if (response.status === 401 && !url.includes("/login")) {
            // Unauthenticated API request - redirect to login
            window.location.href = "/login?next=" + encodeURIComponent(window.location.pathname);
          }
          const msg = data?.error?.message || (typeof data === "string" ? data : `Request failed with status ${response.status}`);
          const err = new Error(msg);
          err.status = response.status;
          err.code = data?.error?.code || "request_failed";
          throw err;
        }

        return data;
      } catch (err) {
        if (!err.status && err.name === "TypeError") {
          err.message = "Network error. Please check your connection and try again.";
        }
        throw err;
      }
    },

    /**
     * Injects a clean loading state into container.
     */
    showLoading(container, message = "Loading...") {
      if (!container) return;
      container.innerHTML = `
        <div class="state-box">
          <div class="spinner"></div>
          <div class="state-title">${this.escape(message)}</div>
          <div class="state-desc">Fetching latest information from the bakery database.</div>
        </div>
      `;
    },

    /**
     * Injects an error state with retry button.
     */
    showError(container, message, onRetry = null) {
      if (!container) return;
      const retryHtml = onRetry
        ? `<button class="btn btn-primary" id="state-retry-btn" style="margin-top:0.75rem;">Try Again</button>`
        : "";

      container.innerHTML = `
        <div class="state-box">
          <div class="state-icon">⚠️</div>
          <div class="state-title">Something went wrong</div>
          <div class="state-desc">${this.escape(message)}</div>
          ${retryHtml}
        </div>
      `;

      if (onRetry) {
        const btn = container.querySelector("#state-retry-btn");
        if (btn) btn.addEventListener("click", onRetry);
      }
    },

    /**
     * Injects a helpful empty state.
     */
    showEmpty(container, title, message, actionText = null, onAction = null) {
      if (!container) return;
      const actionHtml = actionText && onAction
        ? `<button class="btn btn-primary" id="state-action-btn" style="margin-top:0.75rem;">${this.escape(actionText)}</button>`
        : "";

      container.innerHTML = `
        <div class="state-box">
          <div class="state-icon">🥐</div>
          <div class="state-title">${this.escape(title)}</div>
          <div class="state-desc">${this.escape(message)}</div>
          ${actionHtml}
        </div>
      `;

      if (actionText && onAction) {
        const btn = container.querySelector("#state-action-btn");
        if (btn) btn.addEventListener("click", onAction);
      }
    },

    /**
     * Shows a non-blocking toast message.
     */
    showToast(message, type = "info") {
      const container = document.getElementById("toast-container");
      if (!container) return;
      const toast = document.createElement("div");
      toast.className = `toast ${type}`;
      const icon = type === "success" ? "✓" : type === "error" ? "⚠️" : "ℹ️";
      toast.innerHTML = `<span>${icon}</span> <span>${this.escape(message)}</span>`;
      container.appendChild(toast);
      setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        setTimeout(() => toast.remove(), 250);
      }, 3500);
    },

    /**
     * HTML escaping utility.
     */
    escape(str) {
      if (str === null || str === undefined) return "";
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    },

    /**
     * Theme management (light/dark mode).
     */
    initTheme() {
      const saved = localStorage.getItem("loaf_theme");
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      const theme = saved ? saved : prefersDark ? "dark" : "light";
      this.setTheme(theme);

      const btn = document.getElementById("theme-toggle-btn");
      if (btn) {
        btn.addEventListener("click", () => {
          const current = document.documentElement.getAttribute("data-theme") || "light";
          this.setTheme(current === "dark" ? "light" : "dark");
        });
      }
    },

    setTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem("loaf_theme", theme);
      const btn = document.getElementById("theme-toggle-btn");
      if (btn) {
        btn.innerHTML = theme === "dark" ? "☀️" : "🌙";
        const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
        btn.setAttribute("title", label);
        btn.setAttribute("aria-label", label);
      }
    },

    /**
     * Sticky Header Weather Widget
     */
    async loadHeaderWeather() {
      const widget = document.getElementById("weather-widget");
      if (!widget) return;

      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      const tomorrowStr = tomorrow.toISOString().split("T")[0];

      try {
        const data = await this.fetch(`/api/weather?date=${tomorrowStr}`);
        const temp = data.temp_mean_c !== null && data.temp_mean_c !== undefined ? `${data.temp_mean_c}°C` : "—";
        const cond = data.is_rainy ? "Rainy" : "Dry";
        const rain = data.precip_mm ? ` (${data.precip_mm} mm)` : "";
        const icon = data.is_rainy ? "🌧️" : "☀️";

        widget.innerHTML = `
          <span class="weather-icon">${icon}</span>
          <span class="weather-temp">Tomorrow: ${temp}</span>
          <span class="weather-condition">· ${cond}${rain}</span>
          <button id="weather-refresh-btn" class="weather-refresh-btn" type="button" title="Refresh Open-Meteo weather" aria-label="Refresh Open-Meteo weather">↻</button>
        `;
        this.bindWeatherRefresh();
      } catch {
        widget.innerHTML = `
          <span class="weather-icon">☁️</span>
          <span class="weather-temp">Weather offline</span>
          <button id="weather-refresh-btn" class="weather-refresh-btn" type="button" title="Refresh weather data" aria-label="Refresh weather data">↻</button>
        `;
        this.bindWeatherRefresh();
      }
    },

    bindWeatherRefresh() {
      const btn = document.getElementById("weather-refresh-btn");
      if (!btn) return;
      btn.addEventListener("click", async () => {
        btn.classList.add("spinning");
        try {
          await this.fetch("/api/weather/refresh", { method: "POST" });
          this.showToast("Weather refreshed from Open-Meteo", "success");
          await this.loadHeaderWeather();
          if (typeof window.onWeatherRefreshed === "function") {
            window.onWeatherRefreshed();
          }
        } catch {
          this.showToast("Could not refresh live weather. Using cached data.", "error");
        } finally {
          btn.classList.remove("spinning");
        }
      });
    },
  };

  // Expose globally
  window.LoafAPI = LoafAPI;

  // Auto-init theme and weather on DOMContentLoaded
  document.addEventListener("DOMContentLoaded", () => {
    LoafAPI.initTheme();
    LoafAPI.loadHeaderWeather();
  });
})();

