/* cortex-agent web UI: stream agent events over SSE and render them live. */

const els = {
  goal: document.getElementById("goal"),
  send: document.getElementById("send"),
  events: document.getElementById("events"),
  plan: document.getElementById("plan"),
  timeline: document.getElementById("timeline"),
  status: document.getElementById("status"),
  backend: document.getElementById("backend"),
  maxSteps: document.getElementById("maxSteps"),
  placeholder: document.getElementById("placeholder"),
};

const LABELS = {
  plan: "📋 Plan",
  thought: "💭 Thought",
  tool_call: "🔧 Tool call",
  observation: "👁 Observation",
  answer: "✅ Answer",
  error: "❌ Error",
};

function setStatus(state, text) {
  els.status.className = "status " + state;
  els.status.textContent = text;
}

function clearOutput() {
  els.events.innerHTML = "";
  els.plan.innerHTML = "";
  els.timeline.innerHTML = "";
  if (els.placeholder) els.placeholder.style.display = "none";
}

function addTimeline(type) {
  const item = document.createElement("div");
  item.className = "tl-item";
  const dot = document.createElement("span");
  dot.className = "tl-dot " + type;
  const label = document.createElement("span");
  label.textContent = (LABELS[type] || type).replace(/^\S+\s/, "");
  item.appendChild(dot);
  item.appendChild(label);
  els.timeline.appendChild(item);
}

function renderEvent(ev) {
  const card = document.createElement("div");
  card.className = "event " + ev.type;

  const label = document.createElement("div");
  label.className = "label";
  label.innerHTML =
    `<span>${LABELS[ev.type] || ev.type}</span>` +
    (ev.step ? `<span class="step">step ${ev.step}</span>` : "");

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = ev.content || "(empty)";

  card.appendChild(label);
  card.appendChild(body);
  els.events.appendChild(card);
  card.scrollIntoView({ behavior: "smooth", block: "end" });
}

function renderPlan(steps) {
  els.plan.innerHTML = "";
  (steps || []).forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    els.plan.appendChild(li);
  });
}

function handleEvent(ev) {
  if (ev.type === "plan") {
    renderPlan(ev.data && ev.data.steps ? ev.data.steps : []);
  }
  renderEvent(ev);
  addTimeline(ev.type);
}

async function run(goal) {
  clearOutput();
  setStatus("running", "running");
  els.send.disabled = true;

  let resp;
  try {
    resp = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        goal,
        backend: els.backend.value,
        max_steps: parseInt(els.maxSteps.value, 10) || 8,
      }),
    });
  } catch (err) {
    setStatus("error", "error");
    renderEvent({ type: "error", content: String(err) });
    els.send.disabled = false;
    return;
  }

  // Parse the SSE stream manually from the fetch body reader.
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const lines = chunk.split("\n");
      let eventName = "message";
      let dataLine = "";
      for (const line of lines) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }
      if (eventName === "done") {
        setStatus("done", "done");
        continue;
      }
      if (!dataLine) continue;
      try {
        const ev = JSON.parse(dataLine);
        handleEvent(ev);
      } catch (e) {
        /* ignore malformed chunk */
      }
    }
  }

  if (els.status.className.indexOf("error") < 0) setStatus("done", "done");
  els.send.disabled = false;
}

function submit() {
  const goal = els.goal.value.trim();
  if (!goal) return;
  run(goal);
}

els.send.addEventListener("click", submit);
els.goal.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submit();
});

document.querySelectorAll(".examples li").forEach((li) => {
  li.addEventListener("click", () => {
    els.goal.value = li.dataset.goal;
    submit();
  });
});
