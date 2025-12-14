// renderer.js (module)
console.log("Renderer loaded");

const attachBtn = document.getElementById("attachBtn");
const fileInput = document.getElementById("fileInput");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const chatBox = document.getElementById("chatBox");
// ---- LMARS state ----
let originalQuery = "";
let followupQuestions = [];
let followupAnswers = [];
let currentFollowupIndex = 0;
let inFollowupMode = false;


// small helper to create message bubble (returns the bubble DOM element)
function addMessageBubble(role, htmlContent = "") {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = htmlContent;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

// render markdown into a bubble and apply syntax highlighting for code blocks
function renderMarkdownToElement(text, targetDiv) {
  // marked available globally from CDN
  targetDiv.innerHTML = marked.parse(text);

  // highlight code blocks (highlight.js global)
  const codeBlocks = targetDiv.querySelectorAll("pre code");
  codeBlocks.forEach((block) => {
    try {
      hljs.highlightElement(block);
    } catch (err) {
      // ignore highlight errors
    }
  });
}

// send a message — uses streaming endpoint for nicer UX
// Debounce helper
function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// Incremental markdown renderer (debounced)
const incrementalRender = debounce((buffered, streamP) => {
  try {
    // Convert markdown -> HTML
    streamP.innerHTML = marked.parse(buffered);

    // highlight code blocks inside this bubble
    const codeBlocks = streamP.querySelectorAll("pre code");
    codeBlocks.forEach((block) => {
      try { hljs.highlightElement(block); } catch (e) { /* ignore */ }
    });

    // keep scroll at bottom
    chatBox.scrollTop = chatBox.scrollHeight;
  } catch (err) {
    // fallback: just show plain text
    streamP.innerText = buffered;
  }
}, 120); // render at most ~8 times/sec

async function sendMessage() {
  const msg = input.value.trim();
  if (!msg) return;

  // If we are in follow-up mode: this msg is an answer to a follow-up question
  if (inFollowupMode) {
    // Add user answer bubble
    addMessageBubble("user", `<p>${escapeHtml(msg)}</p>`);
    followupAnswers.push(msg);
    input.value = "";
    input.style.height = "auto";

    currentFollowupIndex++;

    if (currentFollowupIndex < followupQuestions.length) {
      // Ask next follow-up question
      addMessageBubble("bot", `<strong>Follow-up (${currentFollowupIndex+1}/${followupQuestions.length}):</strong> ${escapeHtml(followupQuestions[currentFollowupIndex])}`);
      return;
    }

    // All follow-ups answered → call backend again with query + Q/A
    const thinkingBubble = addMessageBubble("bot", `<p><em>Generating final answer…</em></p>`);

    const res = await window.api.sendLmarsQuery(
      originalQuery,
      followupQuestions,
      followupAnswers
    );

    if (res.error) {
      thinkingBubble.innerHTML = `<p><strong>Error:</strong> ${escapeHtml(
        res.error
      )}</p>`;
    } else if (res.status === "answer") {
      // Replace the "thinking" bubble with final answer
      const finalHtml = `
        <p><strong>Final query used:</strong></p>
        <pre>${escapeHtml(res.final_query || originalQuery)}</pre>
        <div class="markdown"></div>
      `;
      thinkingBubble.innerHTML = finalHtml;

      const mdDiv = thinkingBubble.querySelector(".markdown");
      if (mdDiv && res.summary) {
        renderMarkdownToElement(res.summary, mdDiv);
      }
    } 
    else {
      thinkingBubble.innerHTML = `<p>Unexpected response from backend.</p>`;
    }

    // Reset follow-up state so user can ask a new main query
    inFollowupMode = false;
    followupQuestions = [];
    followupAnswers = [];
    currentFollowupIndex = 0;

    return;
  }

  // ---------- Normal mode: initial query ----------
  originalQuery = msg;
  followupQuestions = [];
  followupAnswers = [];
  currentFollowupIndex = 0;

  // Add user bubble
  addMessageBubble("user", `<p>${escapeHtml(msg)}</p>`);
  input.value = "";
  input.style.height = "auto";

  // "Thinking" bubble
  const botDiv = addMessageBubble("bot", `<p><em>Thinking…</em></p>`);

  const res = await window.api.sendLmarsQuery(msg);

   if (res.status === "memory_updated") {
    botDiv.innerHTML = `<p>Memory updated</p>`;
    return;   // stop further processing
  }


  if (res.error) {
    botDiv.innerHTML = `<p><strong>Error:</strong> ${escapeHtml(res.error)}</p>`;
    return;
  }

  if (res.status === "answer") {
    // Show final answer directly
    botDiv.innerHTML = `<div class="markdown"></div>
    `;
    const mdDiv = botDiv.querySelector(".markdown");
    if (mdDiv && res.summary) {
      renderMarkdownToElement(res.summary, mdDiv);
    }
  } else if (res.status === "followup_needed") {
  followupQuestions = res.follow_up_questions || [];

  if (!followupQuestions.length) {
    botDiv.innerHTML = `<p>
      Query not sufficient, and no follow-up questions were provided.
    </p>`;
    return;
  }

  inFollowupMode = true;
  currentFollowupIndex = 0;

  const total = followupQuestions.length;

  // 👇 Heads-up message
  botDiv.innerHTML = `
    Can you answer these <strong>${total}</strong> follow-up question${total === 1 ? "" : "s"}.?
    <strong>Follow-up (1/${total}):</strong> ${escapeHtml(followupQuestions[0])}
  `;
}

}



// helper to escape HTML for user-sent text
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (m) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
}

// send on click
sendBtn.addEventListener("click", (e) => {
  sendMessage();
});

// send on Enter (Shift+Enter => newline)
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// make textarea auto-resize
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
});

// When the attach button is clicked → open file picker
attachBtn.addEventListener("click", () => {
  fileInput.click();
});

// When a file is selected
fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files); // convert FileList → array

  if (files.length === 0) return;

  // Display list of selected files in chat
  const fileListHtml = files
    .map(f => `<div class="file-item">📄 ${f.name} (${Math.round(f.size / 1024)} KB)</div>`)
    .join("");

  addMessageBubble("user", `<div class="file-list">${fileListHtml}</div>`);

  // Send files to backend (example using FormData)
  const formData = new FormData();
  files.forEach((file, index) => {
    formData.append("files", file); // backend must accept "files" as a list
  });

  // Upload asynchronously (customize URL)
  fetch("http://127.0.0.1:8000/upload", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      addMessageBubble("bot", `Uploaded ${files.length} file(s) successfully!`);
    })
    .catch(err => {
      addMessageBubble("bot", `**Upload error:** ${err}`);
    });

  // Reset the input so same files can be selected again
  fileInput.value = "";
});
