const { app, BrowserWindow, shell } = require("electron");
const { spawn, exec } = require("child_process");
const path = require("path");
const net = require("net");

// -------------------------------------------------------------
// KILL ANY PROCESS USING A PORT (NO EXTERNAL MODULES)
// -------------------------------------------------------------
function killProcessOnPort(port) {
  return new Promise((resolve) => {
    const platform = process.platform;
    let cmd;

    if (platform === "win32") {
      // Windows
      cmd = `for /f "tokens=5" %a in ('netstat -ano ^| findstr :${port}') do taskkill /PID %a /F`;
    } else {
      // macOS/Linux
      cmd = `lsof -ti :${port} | xargs kill -9`;
    }

    exec(cmd, () => resolve()); // ignore errors, always resolve
  });
}

// -------------------------------------------------------------
// WAIT FOR BACKEND TO START
// -------------------------------------------------------------
function waitForBackend(port = 8000, retries = 600, delay = 500) {
  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = new net.Socket();
      socket.setTimeout(300);

      socket.on("connect", () => {
        socket.destroy();
        resolve(true);
      });

      const fail = () => {
        socket.destroy();
        retries--;
        if (retries <= 0) return reject(new Error("Backend did not start"));
        setTimeout(tryConnect, delay);
      };

      socket.on("error", fail);
      socket.on("timeout", fail);

      socket.connect(port, "127.0.0.1");
    };

    tryConnect();
  });
}

// -------------------------------------------------------------
// PYTHON BACKEND PROCESS
// -------------------------------------------------------------
let pyProcess = null;

function createPyProcess() {
  const scriptPath = path.join(__dirname, "backend.py");

  pyProcess = spawn("python", [scriptPath], {
    cwd: __dirname,
    stdio: ["pipe", "pipe", "pipe"]
  });

  pyProcess.stdout.on("data", (data) => {
    console.log("PYTHON stdout:", data.toString());
  });

  pyProcess.stderr.on("data", (data) => {
    console.error("PYTHON stderr:", data.toString());
  });

  pyProcess.on("exit", (code) => {
    console.log("PYTHON exited with code:", code);
  });
}

// -------------------------------------------------------------
// ELECTRON WINDOWS
// -------------------------------------------------------------
let loadingWindow = null;
let mainWindow = null;

function createLoadingWindow() {
  loadingWindow = new BrowserWindow({
    width: 400,
    height: 280,
    frame: false,
    resizable: false,
    alwaysOnTop: true,
    minimizable: true,        
    showInTaskbar: true,      
    webPreferences: {
      contextIsolation: true
    }
  });

  loadingWindow.loadFile("loading.html");
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true
    }
  });

  mainWindow.setMenu(null);

  // -----------------------------
  // Fix for opening external links
  // -----------------------------
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (url !== mainWindow.webContents.getURL()) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  mainWindow.loadFile("index.html");

  mainWindow.once("ready-to-show", () => {
    if (loadingWindow) loadingWindow.close();
    mainWindow.maximize();
    mainWindow.show();
  });
}

// -------------------------------------------------------------
// APP FLOW
// -------------------------------------------------------------
app.whenReady().then(async () => {
  createLoadingWindow();

  console.log("Clearing port 8000 (if occupied)...");
  await killProcessOnPort(8000);

  console.log("Starting FastAPI backend...");
  createPyProcess();

  console.log("Waiting for backend...");

  try {
    await waitForBackend(8000);
    console.log("Backend ready. Launching UI...");
    createMainWindow();
  } catch (err) {
    console.error("Backend failed:", err);
    loadingWindow.webContents.executeJavaScript(
      `document.body.innerHTML =
        "<h2 style='text-align:center;margin-top:50px;'>Backend failed to start</h2>"`
    );
  }
});

app.on("before-quit", () => {
  if (pyProcess) pyProcess.kill();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
