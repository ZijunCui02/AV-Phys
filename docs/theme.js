/* Theme toggle + auto-hide navbar + small interactions for AV-Phys site */

(function () {
  const STORAGE_KEY = "color-scheme";
  const ORDER = ["auto", "light", "dark"];
  const root = document.documentElement;
  const mql = window.matchMedia("(prefers-color-scheme: dark)");

  function current() {
    return localStorage.getItem(STORAGE_KEY) || "auto";
  }

  function apply(s) {
    root.classList.remove("theme-auto", "theme-light", "theme-dark");
    root.classList.add("theme-" + s);
    const useDark = s === "dark" || (s === "auto" && mql.matches);
    root.classList.toggle("dark", useDark);
  }

  apply(current());

  const toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const cur = current();
      const next = ORDER[(ORDER.indexOf(cur) + 1) % ORDER.length];
      localStorage.setItem(STORAGE_KEY, next);
      apply(next);
    });
  }

  mql.addEventListener("change", () => {
    if (current() === "auto") apply("auto");
  });
})();

/* Auto-hide navbar */
(function () {
  const nav = document.getElementById("navbar");
  if (!nav) return;
  const H = nav.offsetHeight || 56;
  let lastY = window.scrollY;
  let fixed = false;

  function update() {
    const y = window.scrollY;
    const dir = y > lastY ? "down" : "up";
    if (dir === "down") {
      nav.classList.remove("is-visible");
      if (y > H) {
        fixed = true;
        nav.classList.add("is-fixed");
      }
    } else {
      if (y > 0 && fixed) {
        nav.classList.add("is-visible");
      } else {
        fixed = false;
        nav.classList.remove("is-visible", "is-fixed");
      }
    }
    lastY = y;
  }

  let ticking = false;
  window.addEventListener("scroll", () => {
    if (!ticking) {
      requestAnimationFrame(() => { update(); ticking = false; });
      ticking = true;
    }
  }, { passive: true });
})();

/* Copy buttons inside .code-block */
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".copy-btn");
  if (!btn) return;
  const block = btn.closest(".code-block");
  if (!block) return;
  const code = block.querySelector("code") || block;
  const text = code.textContent.trim();
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => (btn.textContent = orig), 1500);
  });
});
