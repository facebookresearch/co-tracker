const state = {
  video: null,
  numFrames: 0,
  width: 0,
  height: 0,
  selectedColors: [],     // [{r,g,b}, ...]
  previewMaskId: null,
  previewNumFrames: 0,
};

const el = (id) => document.getElementById(id);

const videoSelect = el("videoSelect");
const loadBtn = el("loadBtn");
const videoInfo = el("videoInfo");

const sourceCanvas = el("sourceCanvas");
const sourceCtx = sourceCanvas.getContext("2d");
const frameSlider = el("frameSlider");
const frameLabel = el("frameLabel");
const toleranceSlider = el("toleranceSlider");
const toleranceLabel = el("toleranceLabel");
const cleanupCheck = el("cleanupCheck");
const invertCheck = el("invertCheck");
const pickStatus = el("pickStatus");
const selectedColorsEl = el("selectedColors");
const generateBtn = el("generateBtn");
const clearBtn = el("clearBtn");

const previewCanvas = el("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");
const previewSlider = el("previewSlider");
const previewFrameLabel = el("previewFrameLabel");
const maskMeta = el("maskMeta");
const maskList = el("maskList");

const npySelect = el("npySelect");
const refreshNpyBtn = el("refreshNpyBtn");
const loadNpyBtn = el("loadNpyBtn");
const loadNpyStatus = el("loadNpyStatus");

function drawImageUrlToCanvas(url, canvas, ctx) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = reject;
    img.src = url;
  });
}

async function fetchVideos() {
  const res = await fetch("/api/videos");
  const data = await res.json();
  videoSelect.innerHTML = "";
  for (const v of data.videos) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    videoSelect.appendChild(opt);
  }
}

async function loadVideo() {
  const name = videoSelect.value;
  if (!name) return;
  const res = await fetch("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video: name }),
  });
  if (!res.ok) {
    videoInfo.textContent = "failed to load video";
    return;
  }
  const data = await res.json();
  state.video = name;
  state.numFrames = data.num_frames;
  state.width = data.width;
  state.height = data.height;
  videoInfo.textContent = `${data.width}x${data.height}, ${data.num_frames} frames, ${data.fps.toFixed(1)} fps`;

  frameSlider.min = 0;
  frameSlider.max = data.num_frames - 1;
  frameSlider.value = 0;
  frameSlider.disabled = false;
  frameLabel.textContent = `0 / ${data.num_frames - 1}`;
  await showSourceFrame(0);

  clearSelection();
  previewSlider.disabled = true;
  maskMeta.textContent = "";
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

  await refreshMaskList();
  await refreshNpyList();
}

async function showSourceFrame(idx) {
  await drawImageUrlToCanvas(`/api/frame/${state.video}/${idx}`, sourceCanvas, sourceCtx);
}

async function showPreviewFrame(idx) {
  if (!state.previewMaskId) return;
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const url = `/api/mask_frame/${state.video}/${state.previewMaskId}/${idx}?mode=${mode}`;
  await drawImageUrlToCanvas(url, previewCanvas, previewCtx);
}

frameSlider.addEventListener("input", async () => {
  const idx = parseInt(frameSlider.value, 10);
  frameLabel.textContent = `${idx} / ${state.numFrames - 1}`;
  await showSourceFrame(idx);
});

toleranceSlider.addEventListener("input", () => {
  toleranceLabel.textContent = toleranceSlider.value;
});

previewSlider.addEventListener("input", async () => {
  const idx = parseInt(previewSlider.value, 10);
  previewFrameLabel.textContent = `${idx} / ${state.previewNumFrames - 1}`;
  await showPreviewFrame(idx);
});

for (const radio of document.querySelectorAll('input[name="mode"]')) {
  radio.addEventListener("change", async () => {
    const idx = parseInt(previewSlider.value, 10);
    await showPreviewFrame(idx);
  });
}

sourceCanvas.addEventListener("click", async (ev) => {
  if (!state.video) return;
  const rect = sourceCanvas.getBoundingClientRect();
  const scaleX = sourceCanvas.width / rect.width;
  const scaleY = sourceCanvas.height / rect.height;
  const x = Math.round((ev.clientX - rect.left) * scaleX);
  const y = Math.round((ev.clientY - rect.top) * scaleY);
  const frameIdx = parseInt(frameSlider.value, 10);

  try {
    const res = await fetch("/api/sample_color", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: state.video, frame_idx: frameIdx, x, y }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    addSelectedColor(data.rgb);
    pickStatus.textContent = `Added color rgb(${data.rgb.join(",")}). Click more blocks, or generate the mask.`;
    pickStatus.className = "status";
  } catch (err) {
    pickStatus.textContent = `Pick failed: ${err.message}`;
    pickStatus.className = "status err";
  }
});

function addSelectedColor(rgb) {
  state.selectedColors.push(rgb);
  renderSelectedColors();
}

function removeSelectedColor(index) {
  state.selectedColors.splice(index, 1);
  renderSelectedColors();
}

function clearSelection() {
  state.selectedColors = [];
  renderSelectedColors();
  pickStatus.textContent = "";
}

function renderSelectedColors() {
  selectedColorsEl.innerHTML = "";
  state.selectedColors.forEach((rgb, i) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `rgb(${rgb.join(",")})`;
    const label = document.createElement("span");
    label.textContent = `rgb(${rgb.join(",")})`;
    const remove = document.createElement("span");
    remove.className = "chip-remove";
    remove.textContent = "x";
    remove.addEventListener("click", () => removeSelectedColor(i));
    chip.appendChild(swatch);
    chip.appendChild(label);
    chip.appendChild(remove);
    selectedColorsEl.appendChild(chip);
  });
  generateBtn.disabled = state.selectedColors.length === 0;
}

generateBtn.addEventListener("click", async () => {
  if (!state.video || state.selectedColors.length === 0) return;
  const tolerance = parseFloat(toleranceSlider.value);
  const cleanup = cleanupCheck.checked;
  const invert = invertCheck.checked;

  pickStatus.textContent = "Generating mask...";
  pickStatus.className = "status";

  try {
    const res = await fetch("/api/make_mask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video: state.video, colors: state.selectedColors, tolerance, cleanup, invert,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const invertNote = invert ? " (inverted)" : "";
    pickStatus.textContent = `Mask ${data.mask_id}${invertNote}: present in ${data.found_frames}/${data.num_frames} frames.`;
    pickStatus.className = "status ok";
    await refreshMaskList();
    await selectMaskForPreview(data.mask_id);
  } catch (err) {
    pickStatus.textContent = `Generate failed: ${err.message}`;
    pickStatus.className = "status err";
  }
});

clearBtn.addEventListener("click", clearSelection);

async function refreshMaskList() {
  if (!state.video) return;
  const res = await fetch(`/api/masks/${state.video}`);
  const data = await res.json();
  maskList.innerHTML = "";
  for (const m of data.masks) {
    const li = document.createElement("li");
    li.className = "mask-item";

    const top = document.createElement("div");
    top.className = "mask-item-top";

    const swatches = document.createElement("div");
    swatches.className = "chip-row";
    if (m.source_npy) {
      const chip = document.createElement("div");
      chip.className = "chip";
      chip.textContent = `loaded: ${m.source_npy}`;
      swatches.appendChild(chip);
    } else {
      for (const rgb of m.colors_rgb) {
        const chip = document.createElement("div");
        chip.className = "chip";
        const sw = document.createElement("span");
        sw.className = "swatch";
        sw.style.background = `rgb(${rgb.join(",")})`;
        const lbl = document.createElement("span");
        lbl.textContent = `rgb(${rgb.join(",")})`;
        chip.appendChild(sw);
        chip.appendChild(lbl);
        swatches.appendChild(chip);
      }
    }

    const label = document.createElement("div");
    label.className = "mask-label";
    const invertNote = m.invert ? ", inverted" : "";
    const colorNote = m.source_npy ? "loaded from file" : `${m.colors_rgb.length} color(s)${invertNote}`;
    label.textContent = `${m.mask_id} — ${colorNote}, found ${m.found_frames}/${m.num_frames} frames${m.tolerance != null ? `, tol=${m.tolerance}` : ""}`;

    const actions = document.createElement("div");
    actions.className = "mask-actions";

    const viewBtn = document.createElement("button");
    viewBtn.textContent = "View";
    viewBtn.addEventListener("click", () => selectMaskForPreview(m.mask_id));

    const dlLink = document.createElement("a");
    dlLink.textContent = "Download .npy";
    dlLink.href = `/api/download/${state.video}/${m.mask_id}`;

    actions.appendChild(viewBtn);
    actions.appendChild(dlLink);

    top.appendChild(swatches);
    top.appendChild(actions);
    li.appendChild(top);
    li.appendChild(label);
    maskList.appendChild(li);
  }
}

async function refreshNpyList() {
  const res = await fetch("/api/npy_files");
  const data = await res.json();
  const prev = npySelect.value;
  npySelect.innerHTML = "";
  for (const f of data.files) {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    npySelect.appendChild(opt);
  }
  if (data.files.includes(prev)) npySelect.value = prev;
}

refreshNpyBtn.addEventListener("click", refreshNpyList);

loadNpyBtn.addEventListener("click", async () => {
  if (!state.video) {
    loadNpyStatus.textContent = "Load a video first.";
    loadNpyStatus.className = "status err";
    return;
  }
  const npyPath = npySelect.value;
  if (!npyPath) {
    loadNpyStatus.textContent = "No .npy file selected.";
    loadNpyStatus.className = "status err";
    return;
  }

  loadNpyStatus.textContent = "Loading...";
  loadNpyStatus.className = "status";

  try {
    const res = await fetch("/api/load_mask_npy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: state.video, npy_path: npyPath }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    loadNpyStatus.textContent = `Loaded ${data.source_npy}: present in ${data.found_frames}/${data.num_frames} frames.`;
    loadNpyStatus.className = "status ok";
    await refreshMaskList();
    await selectMaskForPreview(data.mask_id);
  } catch (err) {
    loadNpyStatus.textContent = `Load failed: ${err.message}`;
    loadNpyStatus.className = "status err";
  }
});

async function selectMaskForPreview(maskId) {
  state.previewMaskId = maskId;
  state.previewNumFrames = state.numFrames;
  previewSlider.min = 0;
  previewSlider.max = state.numFrames - 1;
  previewSlider.value = 0;
  previewSlider.disabled = false;
  previewFrameLabel.textContent = `0 / ${state.numFrames - 1}`;
  maskMeta.textContent = `Previewing mask ${maskId}`;
  await showPreviewFrame(0);
}

loadBtn.addEventListener("click", loadVideo);

fetchVideos();
refreshNpyList();
