// static/app.js
document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.getElementById("overlay");
  const overlayText = document.getElementById("overlay-text");

  const restaurantsTbody = document.getElementById("restaurants-table-body");
  const phase1ThreadsTbody = document.getElementById(
    "phase1-threads-table-body"
  );

  const filterInput = document.getElementById("filter-input");
  const statTotalReviews = document.getElementById("stat-total-reviews");
  const statTotalRestaurants = document.getElementById("stat-total-restaurants");
  const statTotalThreads = document.getElementById("stat-total-threads");

  const analysisPlaceholder = document.getElementById("analysis-placeholder");
  const analysisContent = document.getElementById("analysis-content");

  let restaurants = [];
  let phase1Threads = [];

  // ----------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------

  function showOverlay(text) {
    overlayText.textContent = text || "Loading…";
    overlay.classList.remove("hidden");
  }

  function hideOverlay() {
    overlay.classList.add("hidden");
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

  // ----------------------------------------------------------
  // Phase 1 – Load restaurants list
  // ----------------------------------------------------------

  async function loadRestaurants() {
    showOverlay("Loading restaurants from dataset…");

    try {
      const res = await fetch("/api/v1/getAllRestaurents");
      const data = await res.json();

      if (!res.ok) {
        restaurantsTbody.innerHTML = `<tr><td colspan="3" class="error">${escapeHtml(
          data.error || data.message || "Failed to load restaurants."
        )}</td></tr>`;
        phase1ThreadsTbody.innerHTML = "";
        return;
      }

      restaurants = data.restaurants || [];
      phase1Threads = data.threads || [];

      statTotalReviews.textContent =
        data.total_reviews != null ? data.total_reviews : "–";
      statTotalRestaurants.textContent = restaurants.length;
      statTotalThreads.textContent = phase1Threads.length;

      renderRestaurantsTable();
      renderPhase1Threads();
    } catch (err) {
      restaurantsTbody.innerHTML = `<tr><td colspan="3" class="error">Request failed: ${escapeHtml(
        String(err)
      )}</td></tr>`;
      phase1ThreadsTbody.innerHTML = "";
    } finally {
      hideOverlay();
    }
  }

  function renderRestaurantsTable() {
    const filterTerm = (filterInput.value || "").toLowerCase().trim();
    restaurantsTbody.innerHTML = "";

    const filtered = restaurants.filter((r) => {
      const name = (r.name || "").toLowerCase();
      const slug = (r.slug || "").toLowerCase();
      if (!filterTerm) return true;
      return name.includes(filterTerm) || slug.includes(filterTerm);
    });

    if (filtered.length === 0) {
      restaurantsTbody.innerHTML =
        '<tr><td colspan="3" class="muted">No restaurants match this filter.</td></tr>';
      return;
    }

    filtered.forEach((rest) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>
          <div class="cell-main">
            <span class="cell-title">${escapeHtml(rest.name || rest.slug)}</span>
            <span class="cell-sub">${escapeHtml(rest.slug || "")}</span>
          </div>
        </td>
        <td class="cell-center">
          <span class="badge badge-light">${rest.review_count}</span>
        </td>
        <td class="cell-right">
          <button
            class="btn btn-check"
            type="button"
            data-name="${encodeURIComponent(rest.name || rest.slug)}"
          >
            Check
          </button>
        </td>
      `;
      restaurantsTbody.appendChild(tr);
    });

    restaurantsTbody.querySelectorAll(".btn-check").forEach((btn) => {
      btn.addEventListener("click", () => {
        const encoded = btn.getAttribute("data-name") || "";
        const name = decodeURIComponent(encoded);
        analyzeRestaurant(name);
      });
    });
  }

  function renderPhase1Threads() {
    phase1ThreadsTbody.innerHTML = "";

    if (!phase1Threads.length) {
      phase1ThreadsTbody.innerHTML =
        '<tr><td colspan="4" class="muted">No thread traces available yet.</td></tr>';
      return;
    }

    phase1Threads
      .slice()
      .sort((a, b) => (a.thread_index ?? 0) - (b.thread_index ?? 0))
      .forEach((t) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${t.thread_index ?? "-"}</td>
          <td>${t.rows_processed ?? "-"}</td>
          <td>${t.unique_restaurants ?? "-"}</td>
          <td>${t.duration_ms ?? "-"}</td>
        `;
        phase1ThreadsTbody.appendChild(tr);
      });
  }

  // ----------------------------------------------------------
  // Phase 2 & 3 – Analyze one restaurant
  // ----------------------------------------------------------

  async function analyzeRestaurant(name) {
    if (!name) return;

    analysisPlaceholder.classList.add("hidden");
    analysisContent.classList.remove("hidden");
    analysisContent.innerHTML = "";

    showOverlay(`Analyzing "${name}"…`);

    try {
      const url = `/api/v1/getrestaurantrate?name=${encodeURIComponent(name)}`;
      const res = await fetch(url);
      const data = await res.json();

      if (!res.ok) {
        analysisContent.innerHTML = `<p class="error">${escapeHtml(
          data.error || data.message || "Analysis failed."
        )}</p>`;
        return;
      }

      renderAnalysis(data);
    } catch (err) {
      analysisContent.innerHTML = `<p class="error">Request failed: ${escapeHtml(
        String(err)
      )}</p>`;
    } finally {
      hideOverlay();
    }
  }

  function renderAnalysis(data) {
    const restaurantName = data.restaurant || data.search_query || "";
    const globalRate = data.global_rate;
    const ratingsByYear = data.ratings_by_year || {};
    const years = Object.keys(ratingsByYear).sort();
    const totalForRestaurant = data.restaurant_total_reviews ?? 0;

    const phase2 = data.phase2 || {};
    const phase2Threads = phase2.threads || [];
    const phase2TotalThreads = phase2.total_threads ?? phase2Threads.length;
    const phase2TotalReviews =
      phase2.total_reviews_processed ?? data.dataset_total_reviews ?? 0;

    const phase3 = data.phase3 || {};
    const phase3Threads = phase3.threads || [];
    const phase3TotalYearThreads =
      phase3.total_year_threads ?? phase3Threads.length;

    let html = `
      <div class="analysis-header">
        <div>
          <h3>${escapeHtml(restaurantName)}</h3>
          <p class="muted">
            Reviews found for this restaurant: <strong>${totalForRestaurant}</strong>
          </p>
        </div>
        <div class="global-rate">
          <span class="badge badge-label">Global rate</span>
          <span class="global-rate-value">${
            globalRate != null ? Number(globalRate).toFixed(2) : "N/A"
          }</span>
          <span class="global-rate-scale">/ 5</span>
        </div>
      </div>
    `;

    // Yearly ratings (Phase 3 result)
    html += `<section class="analysis-section">
      <h4>Yearly ratings (last 10 years)</h4>
    `;

    if (!years.length) {
      html += `<p class="muted">No yearly ratings could be computed (no reviews in the last 10 years).</p>`;
    } else {
      html += `
        <div class="table-wrapper compact">
          <table class="table thread-table">
            <thead>
              <tr>
                <th>Year</th>
                <th>Average rating (0–5)</th>
              </tr>
            </thead>
            <tbody>
      `;
      years.forEach((year) => {
        const rating = ratingsByYear[year];
        const ratingStr =
          rating != null ? Number(rating).toFixed(2) : "N/A";
        html += `
          <tr>
            <td>${escapeHtml(year)}</td>
            <td>${ratingStr}</td>
          </tr>
        `;
      });
      html += `
            </tbody>
          </table>
        </div>
      `;
    }

    html += `</section>`;

    // Phase 2 traces
    html += `
      <section class="analysis-section">
        <h4>Phase 2 – Review processing threads</h4>
        <p class="muted">
          Threads: <strong>${phase2TotalThreads}</strong>,
          dataset rows processed: <strong>${phase2TotalReviews}</strong>
        </p>
    `;

    if (!phase2Threads.length) {
      html += `<p class="muted">No thread traces available for Phase 2.</p>`;
    } else {
      html += `
        <div class="table-wrapper compact">
          <table class="table thread-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Rows processed</th>
                <th>Matched reviews</th>
                <th>Duration (ms)</th>
              </tr>
            </thead>
            <tbody>
      `;
      phase2Threads
        .slice()
        .sort((a, b) => (a.thread_index ?? 0) - (b.thread_index ?? 0))
        .forEach((t) => {
          html += `
            <tr>
              <td>${t.thread_index ?? "-"}</td>
              <td>${t.rows_processed ?? "-"}</td>
              <td>${t.matched_reviews ?? "-"}</td>
              <td>${t.duration_ms ?? "-"}</td>
            </tr>
          `;
        });
      html += `
            </tbody>
          </table>
        </div>
      `;
    }

    html += `</section>`;

    // Phase 3 traces (per-year threads)
    html += `
      <section class="analysis-section">
        <h4>Phase 3 – Yearly rating threads</h4>
        <p class="muted">
          Year threads created: <strong>${phase3TotalYearThreads}</strong>
        </p>
    `;

    if (!phase3Threads.length) {
      html += `<p class="muted">No thread traces available for Phase 3.</p>`;
    } else {
      html += `
        <div class="table-wrapper compact">
          <table class="table thread-table">
            <thead>
              <tr>
                <th>Year</th>
                <th>Rows processed</th>
                <th>Duration (ms)</th>
              </tr>
            </thead>
            <tbody>
      `;
      phase3Threads
        .slice()
        .sort((a, b) => (a.year ?? 0) - (b.year ?? 0))
        .forEach((t) => {
          html += `
            <tr>
              <td>${t.year ?? "-"}</td>
              <td>${t.rows_processed ?? "-"}</td>
              <td>${t.duration_ms ?? "-"}</td>
            </tr>
          `;
        });
      html += `
            </tbody>
          </table>
        </div>
      `;
    }

    html += `</section>`;

    analysisContent.innerHTML = html;
  }

  // ----------------------------------------------------------
  // Events
  // ----------------------------------------------------------

  filterInput.addEventListener("input", () => {
    renderRestaurantsTable();
  });

  loadRestaurants();
});
