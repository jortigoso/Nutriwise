
/* ---------------------------------------------------------------- */
/*                          Initialization                          */
/* ---------------------------------------------------------------- */

// DOM elements
const chatbox = document.getElementById("chatbox");
const themeToggle = document.getElementById("theme-toggle");
document.getElementById("welcome-time").innerText = getTimestamp();

// Backend URL
const backendUrl = "http://127.0.0.1:5000/query";

/* ---------------------------------------------------------------- */
/*                         Theme Management                         */
/* ---------------------------------------------------------------- */

// Apply selected theme to the page
function applyTheme(mode) {
  document.documentElement.classList.toggle("dark-mode", mode === "dark");
  themeToggle.textContent = mode === "dark" ? "☀️" : "🌙";
}

// Load theme preference from local storage or use system default
function loadTheme() {
  const stored = localStorage.getItem("theme");
  if (stored) {
    applyTheme(stored);
  } else {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(prefersDark ? "dark" : "light");
  }
}

// Toggle theme on button click
themeToggle.addEventListener("click", () => {
  const isDark = document.documentElement.classList.toggle("dark-mode");
  localStorage.setItem("theme", isDark ? "dark" : "light");
  themeToggle.textContent = isDark ? "☀️" : "🌙";
});

// Load theme when DOM is ready
document.addEventListener("DOMContentLoaded", loadTheme);


/* ---------------------------------------------------------------- */
/*                             Utilities                            */
/* ---------------------------------------------------------------- */

// Return current timestamp in HH:MM format
function getTimestamp() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Generate color based on confidence (0-1) from red to green
function getColorFromConfidence(conf) {
  const norm = Math.min(Math.max(conf, 0), 1);
  const hue = norm * 120;
  return `hsl(${hue}, 100%, 50%)`;
}

/* ---------------------------------------------------------------- */
/*                          Message Display                         */
/* ---------------------------------------------------------------- */

// Add a message to the chatbox
function addMessage(text, sender, confidence = null, summary = null, metrics = null, lang = "en") {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender);

  const avatar = document.createElement("img");
  avatar.classList.add("avatar");
  avatar.src = sender === "bot" ? "frontend/assets/images/chatbot.jpg" : "frontend/assets/images/avatar.png";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.innerHTML = text;

  // Add timestamp
  const ts = document.createElement("span");
  ts.classList.add("timestamp");
  ts.innerText = getTimestamp();
  bubble.appendChild(ts);

  const labels = labelTranslations[lang] || labelTranslations["en"];

  // Add summary if available
  if (summary) {
    const sumEl = document.createElement("div");
    sumEl.style.fontStyle = "italic";
    sumEl.style.marginTop = "6px";
    sumEl.innerText = `${labels.summary}: ${summary}`;
    bubble.appendChild(sumEl);
  }

  // Add confidence bar if available
  if (confidence !== null && sender === "bot") {
    const percent = Math.round(confidence * 100);
    const barColor = getColorFromConfidence(confidence);

    const cBar = document.createElement("div");
    cBar.classList.add("confidence-bar");

    const fill = document.createElement("div");
    fill.classList.add("confidence-fill");
    fill.style.width = `${percent}%`;
    fill.style.background = barColor;

    cBar.appendChild(fill);
    bubble.appendChild(cBar);

    const label = document.createElement("div");
    label.classList.add("confidence-label");
    label.innerText = `${percent}% ${labels.confidence}`;
    bubble.appendChild(label);
  }

  // Add metrics if available
  if (metrics && sender === "bot") {
    const panel = document.createElement("div");
    panel.classList.add("metrics-panel");

    const rLabel = document.createElement("div");
    const bLabel = document.createElement("div");
    rLabel.innerHTML = `<b>${labels.recall}:</b> ${(metrics.recall_at_k * 100).toFixed(1)}%`;
    bLabel.innerHTML = `<b>${labels.f1}:</b> ${(metrics.bert_score_f1 * 100).toFixed(1)}%`;
    panel.appendChild(rLabel);
    panel.appendChild(bLabel);
    bubble.appendChild(panel);
  }

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(bubble);
  chatbox.appendChild(msgDiv);
  chatbox.scrollTop = chatbox.scrollHeight;
}

/* ---------------------------------------------------------------- */
/*                        User Input Handler                        */
/* ---------------------------------------------------------------- */

// Handle user question submission
async function ask() {
  const input = document.getElementById("question");
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";

  // Show loading bubble
  const loadDiv = document.createElement("div");
  loadDiv.classList.add("message", "bot");
  const loadAv = document.createElement("img");
  loadAv.classList.add("avatar");
  loadAv.src = "frontend/assets/images/chatbot.jpg";
  const loadBub = document.createElement("div");
  loadBub.classList.add("bubble");
  loadDiv.appendChild(loadAv);
  loadDiv.appendChild(loadBub);
  chatbox.appendChild(loadDiv);
  chatbox.scrollTop = chatbox.scrollHeight;

  // Animated loading dots
  let dots = 1, active = true;
  const loader = setInterval(() => {
    if (!active) return;
    loadBub.innerText = ".".repeat(dots);
    dots = (dots % 3) + 1;
  }, 500);

  // Retry logic for backend request
  let retries = 3;
  while (retries > 0) {
    try {
      const controller = new AbortController();

      const res = await fetch(backendUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
        mode: "cors",
      });

      const data = await res.json();
      active = false;
      clearInterval(loader);
      loadDiv.remove();

      // Show response or error
      if (data.error) {
        addMessage(`⚠️ Error: ${data.error}`, "bot");
      } else {
        let html = data.answer;
        if (data.sources?.length) {
          const labels = labelTranslations[data.lang] || labelTranslations["en"];
          html += "<br><small><b>" + labels.sources + ":</b><br>" +
            data.sources.map(s => `<a href="${s}" target="_blank">${s}</a>`).join("<br>") +
            "</small>";
        }
        addMessage(html, "bot", data.confidence, data.summary, data.metrics, data.lang);
      }
      return;
    } catch (err) {
      retries--;
      console.error(`Fetch attempt ${4 - retries} failed:`, err.message, err);
      if (retries === 0) {
        active = false;
        clearInterval(loader);
        loadDiv.remove();
        addMessage("⚠️ Could not contact server. Please check your network and try again.", "bot");
      } else {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }
}

// Listen for Enter key to submit question
document.getElementById("question").addEventListener("keydown", e => {
  if (e.key === "Enter") ask();
});


/* ---------------------------------------------------------------- */
/*             Predefined labels for different languages            */
/* ---------------------------------------------------------------- */

const labelTranslations = {
  en: {
    sources: "Sources",
    summary: "Summary",
    confidence: "Confidence",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  es: {
    sources: "Fuentes",
    summary: "Resumen",
    confidence: "Confianza",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  it: {
    sources: "Fonti",
    summary: "Sommario",
    confidence: "Affidabilità",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  ru: {
    sources: "Источники",
    summary: "Резюме",
    confidence: "Уверенность",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  fr: {
    sources: "Sources",
    summary: "Résumé",
    confidence: "Confiance",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  de: {
    sources: "Quellen",
    summary: "Zusammenfassung",
    confidence: "Vertrauen",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  pt: {
    sources: "Fontes",
    summary: "Resumo",
    confidence: "Confiança",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  nl: {
    sources: "Bronnen",
    summary: "Samenvatting",
    confidence: "Vertrouwen",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  pl: {
    sources: "Źródła",
    summary: "Podsumowanie",
    confidence: "Pewność",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  sv: {
    sources: "Källor",
    summary: "Sammanfattning",
    confidence: "Tillförlitlighet",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  fi: {
    sources: "Lähteet",
    summary: "Yhteenveto",
    confidence: "Luottamus",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  cs: {
    sources: "Zdroje",
    summary: "Shrnutí",
    confidence: "Důvěra",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  hu: {
    sources: "Források",
    summary: "Összefoglaló",
    confidence: "Bizalom",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  tr: {
    sources: "Kaynaklar",
    summary: "Özet",
    confidence: "Güven",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  ar: {
    sources: "المصادر",
    summary: "الملخص",
    confidence: "الثقة",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  zh: {
    sources: "来源",
    summary: "摘要",
    confidence: "置信度",
    recall: "Recall@3",
    f1: "BERT F1"
  },
  hi: {
    sources: "स्रोत",
    summary: "सारांश",
    confidence: "विश्वास",
    recall: "Recall@3",
    f1: "BERT F1"
  }
};

