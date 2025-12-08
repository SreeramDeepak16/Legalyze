const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const net = require("net");

// -------------------------------------------------------------
// WAIT FOR FASTAPI BACKEND
// -------------------------------------------------------------
function waitForBackend(port = 8000, retries = 100, delay = 200) {
  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = new net.Socket();
      socket.setTimeout(200);

      socket.on("connect", () => {
        socket.destroy();
        resolve(true);
      });

      socket.on("error", () => {
        socket.destroy();
        if (retries <= 0) reject(new Error("Backend did not start"));
        else setTimeout(() => tryConnect(--retries), delay);
      });

      socket.on("timeout", () => {
        socket.destroy();
        if (retries <= 0) reject(new Error("Backend timeout"));
        else setTimeout(() => tryConnect(--retries), delay);
      });

      socket.connect(port, "127.0.0.1");
    };

    tryConnect();
  });
}

// -------------------------------------------------------------
// PYTHON BACKEND PROCESS
// -------------------------------------------------------------
let pyProcess;

function createPyProcess() {
  const script = path.join(__dirname, "backend.py");

  pyProcess = spawn("python", [script], { stdio: ["pipe", "pipe", "pipe"] });

  pyProcess.stdout.on("data", (data) => {
    console.log("PYTHON stdout:", data.toString());
  });

  pyProcess.stderr.on("data", (data) => {
    console.error("PYTHON stderr:", data.toString());
  });

  pyProcess.on("exit", (code) => {
    console.log("PYTHON exited with", code);
  });
}

// -------------------------------------------------------------
// ELECTRON WINDOWS
// -------------------------------------------------------------

let loadingWindow = null;
let mainWindow = null;

// Show loading screen immediately
function createLoadingWindow() {
  loadingWindow = new BrowserWindow({
    width: 400,
    height: 280,
    frame: false,
    transparent: false,
    resizable: false,
    alwaysOnTop: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  loadingWindow.loadFile("loading.html");  // <-- YOU MUST CREATE THIS FILE
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    show: false,  // hide until ready
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile("index.html");
  mainWindow.once("ready-to-show", () => {
    if (loadingWindow) loadingWindow.close();
    mainWindow.show();
  });
}

// -------------------------------------------------------------
// APP FLOW
// -------------------------------------------------------------

app.whenReady().then(async () => {
  createLoadingWindow();     // show buffering UI instantly
  createPyProcess();         // start backend in background

  console.log("Waiting for FastAPI backend to start...");

  try {
    await waitForBackend(8000);   // wait until server responds
    console.log("Backend ready. Launching UI...");
    createMainWindow();
  } catch (err) {
    console.error("Backend failed to start:", err);
    loadingWindow.webContents.executeJavaScript(
      `document.body.innerHTML = "<h2 style='text-align:center;margin-top:50px;'>Backend failed to start</h2>"`
    );
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (pyProcess) pyProcess.kill();
});
