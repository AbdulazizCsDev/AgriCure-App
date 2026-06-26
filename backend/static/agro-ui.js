/* AgroCure shared UI behaviour: theme toggle (persisted) + presentation mode.
   The initial theme/mode is set by a tiny inline script in each page's <head>
   (before paint, to avoid flicker). This file wires the toggle buttons. */
(function(){
  var root = document.documentElement;
  var DESKTOP = function(){ return window.matchMedia("(min-width:1024px)").matches; };

  var SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>';
  var PRESENT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>';

  function theme(){ return root.dataset.theme === "light" ? "light" : "dark"; }
  function isPresent(){ return root.dataset.mode === "present"; }

  function applyTheme(t){ root.dataset.theme = t; try{ localStorage.setItem("agro-theme", t); }catch(e){} updateIcons(); }
  function toggleTheme(){
    root.classList.add("theming");
    applyTheme(theme() === "light" ? "dark" : "light");
    setTimeout(function(){ root.classList.remove("theming"); }, 460);
  }
  function applyMode(present){
    if (present && DESKTOP()){ root.dataset.mode = "present"; try{ localStorage.setItem("agro-mode","present"); }catch(e){} }
    else { delete root.dataset.mode; try{ localStorage.setItem("agro-mode","personal"); }catch(e){} }
    updateIcons();
  }
  function toggleMode(){ applyMode(!isPresent()); }

  function updateIcons(){
    document.querySelectorAll("[data-theme-toggle]").forEach(function(b){
      b.innerHTML = theme() === "light" ? MOON : SUN;
      b.setAttribute("aria-label", theme() === "light" ? "Switch to dark mode" : "Switch to light mode");
      b.title = b.getAttribute("aria-label");
    });
    document.querySelectorAll("[data-mode-toggle]").forEach(function(b){
      b.innerHTML = PRESENT;
      b.setAttribute("aria-pressed", isPresent() ? "true" : "false");
      b.setAttribute("aria-label", isPresent() ? "Exit presentation mode" : "Presentation mode");
      b.title = b.getAttribute("aria-label");
    });
  }

  function init(){
    document.querySelectorAll("[data-theme-toggle]").forEach(function(b){ b.addEventListener("click", toggleTheme); });
    document.querySelectorAll("[data-mode-toggle]").forEach(function(b){ b.addEventListener("click", toggleMode); });
    if (!DESKTOP() && isPresent()) delete root.dataset.mode;   // never present on mobile
    window.addEventListener("resize", function(){ if (!DESKTOP() && isPresent()){ delete root.dataset.mode; updateIcons(); } });
    updateIcons();
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);

  window.AgroUI = { toggleTheme: toggleTheme, toggleMode: toggleMode, theme: theme };
})();
