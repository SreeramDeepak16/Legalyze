const { contextBridge } = require("electron");

// Expose a simple sendMessage API that calls the local backend /chat (non-streaming)
contextBridge.exposeInMainWorld("api", {
  sendMessage: async (msg) => {
    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
      });
      return await res.json();
    } catch (err) {
      return { error: err.message || String(err) };
    }
  }
});
