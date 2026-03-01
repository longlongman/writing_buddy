const contextText = document.getElementById("contextText");
const generateBtn = document.getElementById("generateBtn");
const regenerateBtn = document.getElementById("regenerateBtn");
const clearBtn = document.getElementById("clearBtn");
const resultList = document.getElementById("resultList");
const statusEl = document.getElementById("status");

const CACHE_KEY = "writing_buddy_context";
const LAST_RESULTS_KEY = "writing_buddy_last_results";

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function saveContext() {
  localStorage.setItem(CACHE_KEY, contextText.value);
}

function loadCachedData() {
  const cached = localStorage.getItem(CACHE_KEY);
  if (cached) contextText.value = cached;

  const lastResults = localStorage.getItem(LAST_RESULTS_KEY);
  if (lastResults) {
    try {
      renderResults(JSON.parse(lastResults));
    } catch {
      localStorage.removeItem(LAST_RESULTS_KEY);
    }
  }
}

function insertAtCursor(text) {
  const start = contextText.selectionStart;
  const end = contextText.selectionEnd;
  const before = contextText.value.slice(0, start);
  const after = contextText.value.slice(end);
  const spacer = before.endsWith(" ") || before.length === 0 ? "" : " ";
  contextText.value = `${before}${spacer}${text} ${after}`.trim();
  contextText.focus();
  const pos = (before + spacer + text + " ").length;
  contextText.setSelectionRange(pos, pos);
  saveContext();
}

function renderResults(sentences) {
  resultList.innerHTML = "";
  if (!sentences || sentences.length === 0) return;

  sentences.forEach((sentence) => {
    const li = document.createElement("li");
    const text = document.createElement("p");
    text.textContent = sentence;
    const btn = document.createElement("button");
    btn.className = "secondary";
    btn.textContent = "Insert";
    btn.addEventListener("click", () => insertAtCursor(sentence));
    li.appendChild(text);
    li.appendChild(btn);
    resultList.appendChild(li);
  });
}

async function generate() {
  const text = contextText.value.trim();
  if (text.length < 30) {
    setStatus("Please provide at least 30 characters of context.", true);
    return;
  }

  generateBtn.disabled = true;
  regenerateBtn.disabled = true;
  setStatus("Generating suggestions...");
  saveContext();

  try {
    const res = await fetch("/api/recommend-next-sentences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context_text: text, candidate_count: 5 }),
    });

    const data = await res.json();
    if (!res.ok) {
      setStatus(data.detail || "Request failed.", true);
      return;
    }

    renderResults(data.sentences);
    localStorage.setItem(LAST_RESULTS_KEY, JSON.stringify(data.sentences));
    setStatus(
      `Returned ${data.meta.returned} suggestions. Filtered: ${data.meta.filtered}.`
    );
  } catch (err) {
    setStatus("Network error. Check if server is running.", true);
  } finally {
    generateBtn.disabled = false;
    regenerateBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", generate);
regenerateBtn.addEventListener("click", generate);
clearBtn.addEventListener("click", () => {
  contextText.value = "";
  resultList.innerHTML = "";
  localStorage.removeItem(CACHE_KEY);
  localStorage.removeItem(LAST_RESULTS_KEY);
  setStatus("Cleared.");
});

contextText.addEventListener("input", saveContext);
contextText.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    generate();
  }
});

loadCachedData();
