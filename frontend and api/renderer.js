// renderer.js (module)
console.log("Renderer loaded");

const attachBtn = document.getElementById("attachBtn");
const fileInput = document.getElementById("fileInput");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const chatBox = document.getElementById("chatBox");

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

  // add user bubble (escape HTML)
  addMessageBubble("user", `<p>${escapeHtml(msg)}</p>`);

  // create an initially-empty bot bubble and a plain-text streaming element
  const botDiv = addMessageBubble("bot", `<div class="streaming-plain" style="white-space:pre-wrap;"></div>`);
  const plainEl = botDiv.querySelector(".streaming-plain");

  input.value = "";
  input.style.height = "auto";

  try {
    const url = "http://127.0.0.1:8000/stream?message=" + encodeURIComponent(msg);
    const resp = await fetch(url);

    if (!resp.ok || !resp.body) {
      // If streaming endpoint fails, show error text
      plainEl.innerText = "Error: streaming endpoint returned " + resp.status;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    // As chunks arrive, append to buffer and update the plain text immediately.
    // Also schedule the debounced markdown rendering so markup appears progressively.
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      buffer += chunk;

      // immediate plain-text streaming for instant feedback
      plainEl.innerText = buffer;
      chatBox.scrollTop = chatBox.scrollHeight;

      // schedule markdown rendering (debounced)
      incrementalRender(buffer, plainEl);
    }

    // final render (ensure final markdown/hightlight)
    incrementalRender(buffer, plainEl);
  } catch (err) {
    console.error("stream error", err);
    // show error message in bubble
    botDiv.querySelector(".streaming-plain").innerText = "Streaming Error: " + (err.message || err);
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

