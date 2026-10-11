const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const elements = {
  form: $("#recognition-form"),
  input: $("#image-input"),
  dropZone: $("#drop-zone"),
  selectedFile: $("#selected-file"),
  preview: $("#image-preview"),
  fileName: $("#file-name"),
  fileMeta: $("#file-meta"),
  changeImage: $("#change-image"),
  recognizeButton: $("#recognize-button"),
  deviceInputs: $$("input[name=\"device\"]"),
  deviceCpu: $("#device-cpu"),
  deviceGpu: $("#device-gpu"),
  cpuDetail: $("#cpu-detail"),
  gpuDetail: $("#gpu-detail"),
  deviceNote: $("#device-note"),
  jobPanel: $("#job-panel"),
  jobTitle: $("#job-title"),
  jobState: $("#job-state"),
  jobStage: $("#job-stage"),
  jobLog: $("#job-log"),
  progressBar: $("#progress-bar"),
  results: $("#results-section"),
  reportFrame: $("#report-frame"),
  reportStatus: $("#report-status"),
  resultSummary: $("#result-summary"),
  rawReport: $("#raw-report-link"),
  openDirectory: $("#open-report-directory"),
  toast: $("#toast"),
};

let selectedImage = null;
let previewUrl = null;
let pollTimer = null;
let reportResizeObserver = null;
let openDirectoryUrl = null;
let runtimeLoaded = false;
let taskBusy = false;

function formatNumber(value) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("zh-CN").format(value);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function selectedDevice() {
  return elements.deviceInputs.find((input) => input.checked && !input.disabled)?.value || null;
}

function deviceLabel(device) {
  return String(device || "").startsWith("cuda") ? "GPU" : "CPU";
}

function updateRecognizeAvailability() {
  elements.recognizeButton.disabled = !selectedImage || !selectedDevice() || !runtimeLoaded || taskBusy;
}

function updateDeviceNote() {
  elements.deviceNote.hidden = Boolean(selectedDevice());
  elements.deviceNote.textContent = "当前没有可用的运行设备。";
}

function renderExecutionRuntime(execution) {
  const cpu = execution?.cpu || {};
  const gpu = execution?.gpu || {};
  runtimeLoaded = true;
  elements.deviceCpu.disabled = !cpu.available;
  elements.deviceGpu.disabled = !gpu.available;
  elements.cpuDetail.textContent = [cpu.device_name || "CPU 型号未识别",
    execution?.logical_processors ? `${execution.logical_processors} 个逻辑处理器` : "逻辑处理器数量未知"].join(" · ");
  elements.gpuDetail.textContent = gpu.available
    ? gpu.device_name || "CUDA 可用"
    : gpu.reason || "当前环境不可用";
  $("[data-device-option=\"cuda\"]").classList.toggle("unavailable", !gpu.available);

  if (!selectedDevice()) {
    if (gpu.available) elements.deviceGpu.checked = true;
    else if (cpu.available) elements.deviceCpu.checked = true;
  }
  updateDeviceNote();
  updateRecognizeAvailability();
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) throw new Error(payload.error || `请求失败（${response.status}）`);
  return payload;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 4200);
}

async function loadStatus() {
  try {
    const payload = await api("/api/status");
    const version = payload.version_info || {};
    const code = version.release || `开发版 · ${version.code_commit || "未记录"}`;
    $("#release-info").textContent = `项目代码版本：${code}`;
    $("#release-info").title = `代码：${version.code_commit || "未记录"}`;
    renderDatasetStatus(payload.dataset);
    renderExecutionRuntime(payload.execution);
    const active = payload.jobs.find((job) => ["queued", "running"].includes(job.status));
    taskBusy = Boolean(active);
    updateRecognizeAvailability();
    if (active) {
      renderJob(active);
      startPolling(active.id);
    }
  } catch (error) {
    runtimeLoaded = false;
    elements.deviceCpu.disabled = true;
    elements.deviceGpu.disabled = true;
    elements.deviceNote.textContent = "无法读取运行环境，请检查本地服务日志。";
    elements.deviceNote.hidden = false;
    updateRecognizeAvailability();
    $("#dataset-ready").textContent = "读取失败";
    $("#dataset-ready").classList.add("unready");
    $("#dataset-updated").textContent = error.message;
  }
}

function renderDatasetStatus(dataset) {
  const ready = $("#dataset-ready");
  ready.textContent = dataset.ready ? "可用" : "未就绪";
  ready.classList.toggle("unready", !dataset.ready);
  $("#metric-pieces").textContent = formatNumber(dataset.gear_pieces);
  $("#metric-images").textContent = formatNumber(dataset.ready_images);
  $("#metric-missing").textContent = formatNumber(dataset.missing_images);
  if (dataset.last_updated) {
    $("#dataset-updated").textContent = `本地资料库更新时间：${dataset.last_updated.slice(0, 10)}`;
  } else {
    $("#dataset-updated").textContent = "本地资料库更新时间：未记录";
  }
}

function selectImage(file) {
  if (!file) return false;
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
    showToast("请选择 PNG、JPG/JPEG 或 WebP 图片。");
    return false;
  }
  if (file.size > 25 * 1024 * 1024) {
    showToast("图片不能超过 25 MB。");
    return false;
  }
  selectedImage = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.preview.src = previewUrl;
  elements.fileName.textContent = file.name;
  elements.fileMeta.textContent = `${formatBytes(file.size)} · 保存在本地任务目录`;
  elements.dropZone.hidden = true;
  elements.selectedFile.hidden = false;
  updateRecognizeAvailability();
  return true;
}

function clipboardImage(clipboardData) {
  const imageItem = Array.from(clipboardData?.items || []).find(
    (item) => item.kind === "file" && item.type.startsWith("image/"),
  );
  return imageItem?.getAsFile()
    || Array.from(clipboardData?.files || []).find((file) => file.type.startsWith("image/"))
    || null;
}

for (const dropTarget of [elements.dropZone, elements.selectedFile]) {
  for (const eventName of ["dragenter", "dragover"]) {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropTarget.classList.add("dragover");
    });
  }

  for (const eventName of ["dragleave", "drop"]) {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropTarget.classList.remove("dragover");
    });
  }

  dropTarget.addEventListener("drop", (event) => {
    selectImage(event.dataTransfer.files[0]);
  });
}

elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.input.click();
  }
});

elements.input.addEventListener("change", () => selectImage(elements.input.files[0]));
elements.changeImage.addEventListener("click", () => elements.input.click());
document.addEventListener("paste", (event) => {
  const file = clipboardImage(event.clipboardData);
  if (!file) return;
  event.preventDefault();
  if (selectImage(file)) showToast("已从剪贴板粘贴图片。");
});
for (const input of elements.deviceInputs) {
  input.addEventListener("change", () => {
    updateDeviceNote();
    updateRecognizeAvailability();
  });
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedImage) return;
  const slots = $$('input[name="slot"]:checked').map((input) => input.value);
  if (!slots.length) {
    showToast("请至少选择一个识别部位。");
    return;
  }
  const form = new FormData();
  form.append("image", selectedImage, selectedImage.name);
  form.append("gender", $("#gender").value);
  form.append("top_k", $("#top-k").value);
  form.append("slots", slots.join(","));
  form.append("profile", "current-best");
  form.append("device", selectedDevice());
  taskBusy = true;
  updateRecognizeAvailability();
  elements.results.hidden = true;
  try {
    const payload = await api("/api/recognize", { method: "POST", body: form });
    renderJob(payload.job);
    startPolling(payload.job.id);
    elements.jobPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    showToast(error.message);
    taskBusy = false;
    updateRecognizeAvailability();
  }
});

function renderJob(job) {
  elements.jobPanel.hidden = false;
  elements.jobTitle.textContent = job.device
    ? `${job.title} · ${deviceLabel(job.device)}`
    : job.title;
  elements.jobStage.textContent = job.error || job.stage || "等待中";
  const performanceSummary = $("#recognition-performance");
  const totalSeconds = job.performance?.elapsed_seconds;
  performanceSummary.hidden = job.status !== "succeeded" || !Number.isFinite(totalSeconds);
  performanceSummary.textContent = Number.isFinite(totalSeconds) ? `· 总耗时 ${totalSeconds.toFixed(1)} 秒` : "";
  elements.jobLog.textContent = (job.logs || []).join("\n");
  elements.jobLog.scrollTop = elements.jobLog.scrollHeight;
  elements.jobState.className = `job-state ${job.status}`;
  const stateLabels = {
    queued: "等待中",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
  };
  elements.jobState.textContent = stateLabels[job.status] || job.status;
  elements.progressBar.className = job.status === "running" ? "indeterminate" : "";
  elements.progressBar.style.width = job.status === "succeeded" ? "100%" : job.status === "failed" ? "100%" : "";
}

function startPolling(jobId) {
  window.clearTimeout(pollTimer);
  const poll = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      renderJob(job);
      if (["queued", "running"].includes(job.status)) {
        pollTimer = window.setTimeout(poll, 1400);
        return;
      }
      taskBusy = false;
      updateRecognizeAvailability();
      await loadStatus();
      if (job.status === "succeeded" && job.kind === "recognition") {
        const result = await api(job.result_url);
        renderResults(result, job.report_url, job.open_directory_url);

      } else if (job.status === "failed") {
        showToast(job.error || "任务运行失败，请展开日志查看。");
      }
    } catch (error) {
      showToast(error.message);
      taskBusy = false;
      updateRecognizeAvailability();
    }
  };
  poll();
}

// Display the generated report itself so styles, content, and interactions stay in sync.
elements.reportFrame.addEventListener("load", () => {
  if (!elements.reportFrame.hasAttribute("src")) return;
  reportResizeObserver?.disconnect();
  const reportDocument = elements.reportFrame.contentDocument;
  const reportContent = reportDocument?.querySelector("main");
  elements.reportFrame.setAttribute("aria-busy", "false");
  if (!reportContent) {
    elements.reportStatus.textContent = "报告加载失败，请点击“打开原始报告”重试。";
    elements.reportStatus.hidden = false;
    return;
  }
  const resizeReport = () => {
    // Measure content rather than the viewport, allowing the frame to shrink on collapse.
    const height = Math.ceil(reportContent.getBoundingClientRect().height);
    if (height > 0) elements.reportFrame.style.height = `${height}px`;
  };
  // Defer viewport changes so the report's responsive layout can settle first.
  reportResizeObserver = new ResizeObserver(() => window.requestAnimationFrame(resizeReport));
  reportResizeObserver.observe(reportContent);
  resizeReport();
  elements.reportStatus.hidden = true;
});

function renderResults(result, reportUrl, directoryUrl) {
  reportResizeObserver?.disconnect();
  elements.results.hidden = false;
  elements.rawReport.href = reportUrl;
  openDirectoryUrl = directoryUrl || null;
  elements.openDirectory.disabled = !openDirectoryUrl;
  elements.resultSummary.textContent = `${deviceLabel(result.device)} 推断 · ${result.slots.length} 个部位 · 每个部位最多 ${result.top_k} 件候选 · 姿态质量 ${Math.round(result.pose_quality * 100)}%`;
  elements.reportStatus.textContent = "正在加载报告…";
  elements.reportStatus.hidden = false;
  elements.reportFrame.style.height = "600px";
  elements.reportFrame.setAttribute("aria-busy", "true");
  elements.reportFrame.src = reportUrl;
  elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

elements.openDirectory.addEventListener("click", async () => {
  if (!openDirectoryUrl) return;
  elements.openDirectory.disabled = true;
  try {
    await api(openDirectoryUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Prism-Local-Action": "open-directory" },
      body: JSON.stringify({}),
    });
    showToast("已打开报告所在的本地目录。");
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.openDirectory.disabled = !openDirectoryUrl;
  }
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

loadStatus();
