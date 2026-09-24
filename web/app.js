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
  visualGrid: $("#visual-grid"),
  notice: $("#result-notice"),
  slotTabs: $("#slot-tabs"),
  slotResults: $("#slot-results"),
  resultSummary: $("#result-summary"),
  rawReport: $("#raw-report-link"),
  updateDialog: $("#update-dialog"),
  updateForm: $("#update-form"),
  updateMode: $("#update-mode"),
  modeNote: $("#mode-note"),
  startUpdate: $("#start-update"),
  toast: $("#toast"),
};

const slotLabels = {
  HEAD: "头部",
  BODY: "身体",
  HANDS: "手部",
  LEGS: "腿部",
  FEET: "脚部",
};

const updateNotes = {
  standard: "发现新增装备图片 → 下载与质量校验 → 关联装备标签 → 更新检索索引；随后补充中文名和染色信息。首次补齐图片与索引可能耗时较长。",
  "local-index": "不采集远程图片或资料，使用已有图片重新校验、关联标签，并补齐本地检索索引。训练需要另行审核与执行。",
};

let selectedImage = null;
let previewUrl = null;
let pollTimer = null;
let activeResult = null;
let activeSlot = null;
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

function fileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
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
  const device = selectedDevice();
  if (device === "cuda") {
    const gpuName = elements.gpuDetail.dataset.deviceName;
    elements.deviceNote.innerHTML = `本次任务将固定使用 <strong>GPU</strong>${gpuName ? ` · ${escapeHtml(gpuName)}` : ""}。`;
  } else if (device === "cpu") {
    elements.deviceNote.innerHTML = "本次任务将固定使用 <strong>CPU</strong>，并对该子进程隐藏 CUDA。";
  } else {
    elements.deviceNote.textContent = "当前没有可用的运行设备。";
  }
}

function renderExecutionRuntime(execution) {
  const cpu = execution?.cpu || {};
  const gpu = execution?.gpu || {};
  runtimeLoaded = true;
  elements.deviceCpu.disabled = !cpu.available;
  elements.deviceGpu.disabled = !gpu.available;
  elements.cpuDetail.textContent = execution?.logical_processors
    ? `${execution.logical_processors} 个逻辑处理器 · 兼容性优先`
    : "兼容性优先，推断速度较慢";
  elements.gpuDetail.textContent = gpu.available
    ? gpu.device_name || "CUDA 可用"
    : gpu.reason || "当前环境不可用";
  elements.gpuDetail.dataset.deviceName = gpu.available ? gpu.device_name || "CUDA" : "";
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
    const date = new Date(dataset.last_updated);
    $("#dataset-updated").textContent = `摘要更新于 ${date.toLocaleString("zh-CN", { hour12: false })}`;
  } else {
    $("#dataset-updated").textContent = "尚未生成数据集摘要";
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
  const datasetResult = $("#dataset-result");
  datasetResult.hidden = !job.dataset_result;
  if (job.dataset_result) {
    const result = job.dataset_result;
    datasetResult.textContent = `新发现 ${result.new_discovered_images} 张 · 下载 ${result.images_downloaded ?? "未执行"} 张 · 新通过质检 ${result.new_qa_ready_images} 张 · 新进入检索 ${result.new_indexed_images} 张 · 待标签 ${result.pending_label_images} 张 · 下载待补齐 ${result.missing_download_images} 张 · 训练准入尚未执行`;
  }
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
        renderResults(result, job.report_url);
      } else if (job.status === "succeeded" && job.kind === "dataset") {
        showToast(job.dataset_result?.message || "任务结束，请查看图片更新结果。");
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

function renderResults(result, reportUrl) {
  activeResult = result;
  activeSlot = result.slots?.[0]?.slot || null;
  elements.results.hidden = false;
  elements.rawReport.href = reportUrl;
  elements.resultSummary.textContent = `${deviceLabel(result.device)} 推断 · ${result.slots.length} 个部位 · 每个部位最多 ${result.top_k} 件候选 · 姿态质量 ${Math.round(result.pose_quality * 100)}%`;
  elements.visualGrid.innerHTML = "";

  const visuals = [
    ["输入截图", result.input.path],
    ["人物与区域质检", result.artifacts.overlay],
    ["装备部位裁剪", result.artifacts.contact_sheet],
  ];
  for (const [label, path] of visuals) {
    const figure = document.createElement("figure");
    figure.className = "visual-card";
    figure.innerHTML = `<div><img loading="lazy" alt="${escapeHtml(label)}"></div><figcaption>${escapeHtml(label)}</figcaption>`;
    $("img", figure).src = fileUrl(path);
    elements.visualGrid.append(figure);
  }

  const warnings = result.warnings || [];
  elements.notice.innerHTML = warnings.length
    ? `<ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
    : "候选分数是相对排序信号，并非准确率或概率。";

  elements.slotTabs.innerHTML = "";
  for (const slot of result.slots) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "tab";
    button.textContent = `${slotLabels[slot.slot] || slot.slot} · ${slot.candidates?.length || 0}`;
    button.setAttribute("aria-selected", String(slot.slot === activeSlot));
    button.addEventListener("click", () => {
      activeSlot = slot.slot;
      $$("button", elements.slotTabs).forEach((tab, index) => {
        tab.setAttribute("aria-selected", String(result.slots[index].slot === activeSlot));
      });
      renderSlot(slot);
    });
    elements.slotTabs.append(button);
  }
  if (result.slots[0]) renderSlot(result.slots[0]);
  elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSlot(slot) {
  const benchmark = slot.benchmark;
  const benchmarkText = benchmark
    ? `参考基准：Top-1 ${formatPercent(benchmark.top1)} / Top-10 ${formatPercent(benchmark.top10)}；${benchmark.scope}。该指标不是本图置信度。`
    : "当前部位没有可用的参考基准。";
  const quality = slot.ranking_quality?.level || "normal";
  const qualityText = quality === "low" ? "建议人工复核" : "排序正常";
  const candidates = slot.candidates || [];

  elements.slotResults.innerHTML = `
    <div class="slot-report-header">
      <div>
        <h3>${escapeHtml(slotLabels[slot.slot] || slot.slot)}</h3>
        <p>${escapeHtml(benchmarkText)}</p>
      </div>
      <span class="quality-badge ${escapeHtml(quality)}">${escapeHtml(qualityText)}</span>
    </div>
    <div class="candidate-grid"></div>
  `;
  const grid = $(".candidate-grid", elements.slotResults);
  if (!candidates.length) {
    grid.className = "empty-state";
    grid.textContent = "这个部位没有返回候选。";
    return;
  }
  for (const candidate of candidates) {
    const article = document.createElement("article");
    const isNoEquipment = Boolean(candidate.is_no_equipment);
    article.className = `candidate${isNoEquipment ? " no-equipment" : ""}`;
    const displayName = candidate.name_zh || candidate.name_en || "未知装备";
    const englishName = candidate.name_zh ? candidate.name_en : "";
    const sourceUrl = candidate.source_image_url || candidate.source_item_url || "";
    const alternatives = (candidate.appearance_alternatives || [])
      .map((item) => item.name_zh || item.name_en)
      .filter(Boolean)
      .join("、");
    article.innerHTML = `
      <span class="candidate-rank">#${escapeHtml(candidate.rank)}</span>
      ${isNoEquipment
        ? '<div class="candidate-image invisible-appearance"><strong>∅</strong><span>不显示外观</span></div>'
        : `<div class="candidate-image"><img loading="lazy" alt="${escapeHtml(displayName)}"></div>`}
      <div class="candidate-copy">
        <h4>${escapeHtml(displayName)}${englishName ? `<span>${escapeHtml(englishName)}</span>` : ""}</h4>
        ${isNoEquipment ? '<div class="no-equipment-note">未检测到可见装备</div>' : ""}
        ${alternatives ? `<div class="candidate-alternatives">同模型可选：${escapeHtml(alternatives)}</div>` : ""}
        <div class="candidate-meta">
          <span>${isNoEquipment ? "无装备信号" : "分数"} ${Number(candidate.score).toFixed(4)}</span>
          ${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener">来源 ↗</a>` : ""}
        </div>
      </div>
    `;
    if (!isNoEquipment) {
      const image = $("img", article);
      image.src = fileUrl(candidate.local_preview);
      image.addEventListener("error", () => {
        image.alt = `${displayName}（本地预览缺失）`;
        image.closest(".candidate-image").classList.add("missing");
      });
    }
    grid.append(article);
  }
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

for (const button of $$("[data-open-update]")) {
  button.addEventListener("click", () => elements.updateDialog.showModal());
}

elements.updateMode.addEventListener("change", () => {
  elements.modeNote.textContent = updateNotes[elements.updateMode.value];
});

elements.updateForm.addEventListener("submit", async (event) => {
  if (event.submitter?.value === "cancel") return;
  event.preventDefault();
  elements.startUpdate.disabled = true;
  try {
    const payload = await api("/api/dataset/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: elements.updateMode.value }),
    });
    elements.updateDialog.close();
    renderJob(payload.job);
    startPolling(payload.job.id);
    elements.jobPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.startUpdate.disabled = false;
  }
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

loadStatus();
