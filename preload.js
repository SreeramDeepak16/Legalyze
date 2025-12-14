// preload.js
const { contextBridge } = require("electron");

// Expose APIs to the renderer
contextBridge.exposeInMainWorld("api", {
  // // (keep this if you still want /chat later – currently unused)
  // sendMessage: async (msg) => {
  //   try {
  //     const res = await fetch("http://127.0.0.1:8000/chat", {
  //       method: "POST",
  //       headers: { "Content-Type": "application/json" },
  //       body: JSON.stringify({ message: msg }),
  //     });
  //     return await res.json();
  //   } catch (err) {
  //     return { error: err.message || String(err) };
  //   }
  // },

  sendLmarsQuery: async (query, followupQuestions = null, followupAnswers = null) => {
    try {
      const body = { query };

      if (followupQuestions && followupAnswers) {
        body.followup_questions = followupQuestions;
        body.followup_answers = followupAnswers;
      }

      const res = await fetch("http://127.0.0.1:8000/lmars/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      return await res.json();
    } catch (err) {
      return { error: err.message || String(err) };
    }
  },
});