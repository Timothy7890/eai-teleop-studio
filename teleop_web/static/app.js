const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
let appState = {device: null, tasks: [], process: {}, teleop: {}};
let activeTaskId = null;
let deviceFormDirty = false;
let cameraPreviewKey = '';
const DEFAULT_IMAGE_SERVER_IP = '192.168.123.5';
const DEFAULT_WEBRTC_SERVER_IP = '192.168.61.142';
const DEFAULT_H2_INIT_ARM_POSE_FILE = '';
const DEFAULT_DATA_DIR = '~/data';
const DEFAULT_IK_REPLAY_LIVE_URL = 'http://192.168.61.228:8000/api/live/state';
const DEFAULT_OSS_PACKAGE_ROOT = '/data03/data/datasets/lerobot/packages';
const DEFAULT_OSS_ROOT = 'oss://bwton-idc/openpi';
const DEFAULT_MODEL_DOWNLOAD_ROOT = '/data03/data/models/openpi_downloads';
const DEFAULT_OSS_TASK_NAME = 'h2_switch_close_to_remote_merged';
const DEFAULT_TRAINING_TASK_NAME = 'h2_switch_close_to_remote';
const TRAINING_COMMAND_DEFAULTS = {
  configName: 'pi05_h2_lerobot',
  actionDim: '32',
  realActionDim: '14',
  actionHorizon: '16',
  fsdpDevices: '2',
  batchSize: '32',
  numTrainSteps: '50000',
  saveInterval: '5000',
  keepPeriod: '25000',
};
const TASK_PAGE_SIZE = 10;
let dataPreviewState = {preview:null,index:0,taskId:null,episode:null};
let dataListState = {taskId:'',page:1,pageSize:50,total:0,episodes:[],tasks:[],loading:false};
let activePostprocessTaskId = null;
let taskPage = 1;
let userInteractingUntil = 0;
let refreshInFlight = false;
let activeView = localStorage.getItem('teleop.activeView') || 'devices';
let sidebarCollapsed = localStorage.getItem('teleop.sidebarCollapsed') === '1';
let editingTrainingSetId = null;
let expandedTrainingSetIds = new Set(JSON.parse(localStorage.getItem('teleop.expandedTrainingSets') || '[]'));
let ossTransferState = {remoteUri:'',remoteEntries:[],loading:false,localDir:'',taskName:''};
let runtimePending = null;
let fastRefreshTimer = null;
const pageJumpTimers = {};

function normalizeStaticLabels() {
  if (activeView === 'trainingDoc') {
    activeView = 'trainingTemplate';
    localStorage.setItem('teleop.activeView', activeView);
  }
  const ossButton = $('[data-view="oss"]');
  if (ossButton) {
    if (!$('.model-management-group')) {
      const group = document.createElement('div');
      group.className = 'nav-group model-management-group';
      group.innerHTML = '<span>▣</span>模型管理';
      ossButton.insertAdjacentElement('beforebegin', group);
    }
    ossButton.classList.add('sub', 'model-sub');
    ossButton.innerHTML = '<span>⇄</span>OSS传输';
    if (!$('[data-view="trainingTemplate"]')) {
      const templateButton = document.createElement('button');
      templateButton.className = 'nav-item sub model-sub';
      templateButton.dataset.view = 'trainingTemplate';
      templateButton.innerHTML = '<span>▤</span>训练模板';
      ossButton.insertAdjacentElement('afterend', templateButton);
    }
    if (!$('[data-view="trainingCommand"]')) {
      const commandButton = document.createElement('button');
      commandButton.className = 'nav-item sub model-sub';
      commandButton.dataset.view = 'trainingCommand';
      commandButton.innerHTML = '<span>⌁</span>训练命令';
      $('[data-view="trainingTemplate"]')?.insertAdjacentElement('afterend', commandButton);
    }
    $('[data-view="oss"]')?.classList.add('sub', 'model-sub');
    $('[data-view="trainingTemplate"]')?.classList.add('sub', 'model-sub');
    $('[data-view="trainingCommand"]')?.classList.add('sub', 'model-sub');
  }
  if (!$('#trainingTemplateView')) {
    const main = $('.content');
    const section = document.createElement('section');
    section.id = 'trainingTemplateView';
    section.className = 'view';
    section.innerHTML = `<div class="page-title">
      <div><h1>训练模板</h1><p>维护模型训练文档模板；这里改的是模板骨架，不是最终命令</p></div>
      <button id="refreshTrainingTemplate" class="primary" type="button">刷新</button>
    </div>
    <div class="training-panel oss-page-panel">
      <div class="training-panel-head">
        <div><h2>训练模板管理</h2><p>模板按当前数据目录保存，可使用变量占位符，保存后训练命令页会按新模板生成命令</p></div>
      </div>
      <div id="trainingTemplatePanel" class="oss-transfer-panel"></div>
    </div>`;
    main?.appendChild(section);
  }
  if (!$('#trainingCommandView')) {
    const main = $('.content');
    const section = document.createElement('section');
    section.id = 'trainingCommandView';
    section.className = 'view';
    section.innerHTML = `<div class="page-title">
      <div><h1>训练命令</h1><p>填写训练参数，将训练模板渲染成可复制的执行命令</p></div>
      <button id="refreshTrainingCommand" class="primary" type="button">刷新</button>
    </div>
    <div id="trainingCommandPanel" class="oss-transfer-panel training-command-flat"></div>`;
    main?.appendChild(section);
  }
}

function mountPackageButtonInPostprocessActions() {
  const packageButton = $('#packageButton');
  const packageAssetsButton = $('#packageAssetsButton');
  const normalizeButton = $('#normalizeButton');
  const actions = normalizeButton?.parentElement;
  if (packageAssetsButton && actions && packageAssetsButton.parentElement !== actions) {
    actions.insertBefore(packageAssetsButton, normalizeButton);
  }
  if (packageButton && actions && packageButton.parentElement !== actions) {
    actions.insertBefore(packageButton, normalizeButton);
  }
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
function showNotice(message, kind = 'error') {
  const node = $('#notice'); node.textContent = message; node.className = `notice visible ${kind}`;
  clearTimeout(showNotice.timer); showNotice.timer = setTimeout(() => node.className = 'notice', 4500);
}
function scheduleFastRefresh() {
  clearTimeout(fastRefreshTimer);
  if (!runtimePending) return;
  fastRefreshTimer = setTimeout(async () => {
    try {
      await refresh(true, {auto:true});
    } finally {
      if (runtimePending) scheduleFastRefresh();
    }
  }, 700);
}
function setRuntimePending(type, message, options = {}) {
  runtimePending = {
    type,
    message,
    taskId: options.taskId ?? activeTaskId,
    startedAt: Date.now(),
    timeoutMs: options.timeoutMs || 12000,
    mode: options.mode || '',
  };
  const status = $('#processStatus');
  if (status) status.textContent = message;
  showNotice(message, 'success');
  scheduleFastRefresh();
}
function clearRuntimePending() {
  runtimePending = null;
  clearTimeout(fastRefreshTimer);
  fastRefreshTimer = null;
}
function runtimePendingResolved(process, teleop) {
  if (!runtimePending) return true;
  const taskMatches = runtimePending.taskId == null || String(process.task?.id ?? '') === String(runtimePending.taskId);
  if (runtimePending.type === 'process_start') return Boolean(process.running && teleop.online && taskMatches);
  if (runtimePending.type === 'teleop_start') return Boolean(process.running && teleop.START && taskMatches);
  if (runtimePending.type === 'record_start') return Boolean(process.running && teleop.RECORD_RUNNING && taskMatches);
  if (runtimePending.type === 'record_stop') return Boolean(process.running && !teleop.RECORD_RUNNING && taskMatches);
  if (runtimePending.type === 'process_stop') return Boolean(!process.running);
  return true;
}
function runtimePendingTimedOut() {
  return runtimePending && Date.now() - runtimePending.startedAt > runtimePending.timeoutMs;
}
async function copyText(value) {
  const text = String(value || '');
  if (!text) return showNotice('没有可复制的内容');
  try {
    await navigator.clipboard.writeText(text);
    showNotice('已复制', 'success');
  } catch (_error) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
    showNotice('已复制', 'success');
  }
}
async function api(path, payload) {
  const response = await fetch(path, payload ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)} : {});
  const contentType = response.headers.get('content-type') || '';
  let result = null;
  if (contentType.includes('application/json')) {
    result = await response.json();
  } else {
    const text = await response.text();
    const fallback = response.status === 404
      ? '接口不存在，可能服务尚未重启或页面版本与后端不一致'
      : (text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() || `请求失败 (${response.status})`);
    result = {error: fallback};
  }
  if (!response.ok) throw new Error(result.error || `请求失败 (${response.status})`);
  return result;
}
function formatTime(value) {
  if (!value) return '—';
  const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', {hour12:false});
}
function episodeNumber(value) {
  const match = String(value || '').match(/^episode_(\d+)$/);
  return match ? Number(match[1]) : 0;
}
function clampPage(value, totalPages) {
  const maxPage = Math.max(1, Number(totalPages) || 1);
  const page = Math.trunc(Number(value) || 1);
  return Math.min(maxPage, Math.max(1, page));
}
function debouncePageJump(key, callback, delay = 500) {
  clearTimeout(pageJumpTimers[key]);
  pageJumpTimers[key] = setTimeout(callback, delay);
}
function formatBytes(value) {
  const size = Number(value) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  return `${(size / 1024 / 1024 / 1024).toFixed(1)} GB`;
}
function holdAutoRefresh(ms = 900) {
  userInteractingUntil = Math.max(userInteractingUntil, Date.now() + ms);
}
function ensureRuntimeCopyButtons() {
  const commandBlock = $('#commandText')?.closest('.command');
  if (commandBlock && !commandBlock.querySelector('[data-action="copy-target"]')) {
    commandBlock.insertAdjacentHTML('beforeend', '<button class="copy-icon-btn" title="复制" aria-label="复制" data-action="copy-target" data-target="commandText"><span></span></button>');
  }
  const logDetails = $('#logOutput')?.closest('details');
  if (logDetails && !logDetails.querySelector('[data-action="copy-target"]')) {
    logDetails.insertAdjacentHTML('beforeend', '<button class="copy-icon-btn log-copy-btn" title="复制" aria-label="复制" data-action="copy-target" data-target="logOutput"><span></span></button>');
  }
}
function renderCameraPreview() {
  const panel = $('#cameraPreviewGrid');
  const hint = $('#cameraPreviewHint');
  if (!panel || !hint) return;
  const preview = appState.camera_preview || {cameras: []};
  if (preview.error) {
    hint.textContent = preview.error;
    panel.innerHTML = '';
    cameraPreviewKey = '';
    return;
  }
  const cameras = (preview.cameras || []).filter(camera => camera.enable_webrtc || camera.url);
  if (!cameras.length) {
    hint.textContent = appState.process?.running ? '当前相机配置没有可预览的 WebRTC 流' : '开始采集后显示 WebRTC 相机流';
    panel.innerHTML = '';
    cameraPreviewKey = '';
    return;
  }
  const nextKey = cameras.map(camera => `${camera.name}:${camera.url}:${camera.webrtc_ready}:${camera.webrtc_error || ''}`).join('|');
  hint.textContent = preview.warning || `共 ${cameras.length} 路 WebRTC 相机流，已自动打开`;
  if (nextKey === cameraPreviewKey && panel.children.length) return;
  cameraPreviewKey = nextKey;
  panel.innerHTML = cameras.map(camera => {
    const recordColors = (camera.record_colors || []).map(name => `${name}.jpg`).join(' / ') || '不录制图片';
    const unavailable = camera.webrtc_ready === false;
    const body = unavailable
      ? `<div class="camera-offline"><strong>${escapeHtml(camera.webrtc_error || 'WebRTC 不可用')}</strong><span>${escapeHtml(camera.url || '未生成预览地址')}</span></div>`
      : `<iframe src="${escapeHtml(camera.embed_url || `${camera.url}?autostart=1&embed=1`)}" loading="eager" referrerpolicy="no-referrer" allow="autoplay; fullscreen; camera; microphone"></iframe>`;
    return `<article class="camera-card">
    <div class="camera-title"><strong>${escapeHtml(camera.name)}</strong><small>${escapeHtml(recordColors)}</small></div>
    ${body}
    <div class="camera-meta">WebRTC: ${escapeHtml(camera.webrtc_port)} · ZMQ: ${escapeHtml(camera.zmq_port ?? '—')}</div>
  </article>`;
  }).join('');
}
function openModal(id) { $(`#${id}`).classList.add('open'); $(`#${id}`).setAttribute('aria-hidden','false'); }
function closeModal(id) { $(`#${id}`).classList.remove('open'); $(`#${id}`).setAttribute('aria-hidden','true'); }

function renderDevice() {
  const device = appState.device;
  $('#currentDeviceId').textContent = device?.id || '保存配置后生成';
  $('#currentDeviceStatus').textContent = device?.status || '未配置';
  $('#currentDeviceStatus').className = `status ${device?.status === '使用中' ? 'active' : 'idle'}`;
  $('#deviceUpdated').textContent = device?.updated_at ? formatTime(device.updated_at) : '—';
  $('#configFileHint').textContent = appState.config_file || '本地 Python 配置';
  $('#taskDeviceName').textContent = device ? `${device.name} · ${device.config.arm}` : '尚未配置';
  if (!deviceFormDirty) fillDeviceForm(device);
}
function statusClass(status) { return ({'采集中':'active','已暂停':'paused','已完成':'done'}[status] || 'idle'); }
function postprocessBadge(ready, doneText = '已完成', pendingText = '未完成') {
  return `<span class="format-badge ${ready ? 'ok' : 'pending'}">${ready ? doneText : pendingText}</span>`;
}
function convertStatusText(status) {
  const cancelled = Boolean(status.convert_cancelled || isCancelledPostprocess(status.last_convert_record));
  if (status.convert_running) return '转换中';
  if (cancelled) return '已取消';
  if (status.convert_failed) return '失败';
  return status.lerobot_ready ? '已完成' : '未转换';
}
function normalizeStatusText(status) {
  const cancelled = Boolean(status.normalize_cancelled || isCancelledPostprocess(status.last_normalize_record));
  if (status.normalize_running) return '计算中';
  if (cancelled) return '已取消';
  if (status.normalize_failed) return '失败';
  if (status.norm_stats_ready) return '已完成';
  return '未计算';
}
function packageStatusText(status) {
  const cancelled = Boolean(status.package_cancelled || isCancelledPostprocess(status.last_package_record));
  if (status.package_running) return '压缩中';
  if (cancelled) return '已取消';
  if (status.package_failed) return '失败';
  if (status.last_package || status.packaged_at) return '已压缩';
  return '未压缩';
}
function dataPackageStatusText(status) {
  const fallbackRecord = status.last_package_record?.kind === 'package' ? status.last_package_record : null;
  const record = status.last_data_package_record || fallbackRecord;
  const fallbackPackage = String(status.last_package || '').includes('_lerobot_') ? status.last_package : '';
  const cancelled = Boolean(status.data_package_cancelled || isCancelledPostprocess(record));
  if (status.data_package_running) return '打包中';
  if (cancelled) return '已取消';
  if (status.data_package_failed) return '失败';
  if (status.last_data_package || status.data_packaged_at || fallbackPackage) return '已打包';
  return '未打包';
}
function assetsPackageStatusText(status) {
  const fallbackRecord = status.last_package_record?.kind === 'package_assets' ? status.last_package_record : null;
  const record = status.last_assets_package_record || fallbackRecord;
  const fallbackPackage = String(status.last_package || '').includes('_openpi_assets_') ? status.last_package : '';
  const cancelled = Boolean(status.assets_package_cancelled || isCancelledPostprocess(record));
  if (status.assets_package_running) return '打包中';
  if (cancelled) return '已取消';
  if (status.assets_package_failed) return '失败';
  if (status.last_assets_package || status.assets_packaged_at || fallbackPackage) return '已打包';
  return '未打包';
}
function postprocessStatusProgress(progress, {running = false, complete = false, failed = false, idleText = '等待中'} = {}) {
  const normalized = normalizePostprocessProgress(progress);
  const percent = normalized && typeof normalized.percent === 'number'
    ? Math.round(normalized.percent)
    : (complete ? 100 : null);
  const indeterminate = running && percent === null;
  const width = indeterminate ? 40 : Math.max(0, Math.min(100, percent ?? 0));
  const stage = normalized?.stage || (running ? '处理中' : (complete ? '已完成' : (failed ? '失败' : idleText)));
  const percentText = percent === null ? (running ? '进行中' : '—') : `${percent}%`;
  const ratioText = normalized && normalized.current !== null && normalized.total !== null
    ? `${normalized.current}/${normalized.total}`
    : '';
  const metaLeft = ratioText || normalized?.speed || (running ? '运行中' : (complete ? '完成' : idleText));
  const metaRight = ratioText && normalized?.speed ? normalized.speed : '';
  const title = [stage, percentText, ratioText, normalized?.speed].filter(Boolean).join(' ');
  return `<div class="postprocess-status-progress ${indeterminate ? 'indeterminate' : ''} ${failed ? 'failed' : ''} ${complete ? 'complete' : ''}" title="${escapeHtml(title)}">
    <div class="postprocess-status-head"><span>${escapeHtml(stage)}</span><strong>${escapeHtml(percentText)}</strong></div>
    <div class="postprocess-status-track"><i style="width:${width}%"></i></div>
    <div class="postprocess-status-meta"><span>${escapeHtml(metaLeft)}</span><span>${escapeHtml(metaRight)}</span></div>
  </div>`;
}
function renderConvertStatus(task) {
  const status = task.postprocess_status || {};
  const text = convertStatusText(status);
  return `<div class="postprocess-cell status-only">${postprocessBadge(text === '已完成' || text === '转换中', text, text)}</div>`;
}
function renderNormalizeStatus(task) {
  const status = task.postprocess_status || {};
  const text = normalizeStatusText(status);
  return `<div class="postprocess-cell status-only">${postprocessBadge(text === '已完成' || text === '计算中', text, text)}</div>`;
}
function renderPackageStatus(task) {
  const status = task.postprocess_status || {};
  const text = packageStatusText(status);
  return `<div class="postprocess-cell status-only">${postprocessBadge(text === '已压缩' || text === '压缩中', text, text)}</div>`;
}
function renderDataPackageStatus(task) {
  const status = task.postprocess_status || {};
  const text = dataPackageStatusText(status);
  return `<div class="postprocess-cell status-only">${postprocessBadge(text === '已打包' || text === '打包中', text, text)}</div>`;
}
function renderAssetsPackageStatus(task) {
  const status = task.postprocess_status || {};
  const text = assetsPackageStatusText(status);
  return `<div class="postprocess-cell status-only">${postprocessBadge(text === '已打包' || text === '打包中', text, text)}</div>`;
}
function archiveStatusText(task) {
  const status = task.archive_status || {};
  if (status.running) return '归档中';
  if (status.error || status.failed_at) return '归档失败';
  if (status.last_archive || status.finished_at) return '已归档';
  return '未归档';
}
function detailItem(label, value, extraClass = '') {
  return `<div class="${extraClass}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value || '—')}</strong></div>`;
}
function setControlCompactMode(compact) {
  $('#taskDetailPanel')?.classList.toggle('hidden', compact);
}
function renderTaskDetails(task) {
  const panel = $('#taskDetailPanel');
  if (!panel) return;
  if (!task) {
    panel.innerHTML = '';
    return;
  }
  const status = task.postprocess_status || {};
  const robotDir = status.robot_dir || `${appState.dataset_root}/${task.name}`;
  const lerobotDir = status.lerobot_dir || `local/${task.name}`;
  const normStatsDir = status.norm_stats_dir || '—';
  panel.innerHTML = [
    detailItem('任务 ID', task.id),
    detailItem('任务名称', task.name),
    detailItem('设备', task.device_name),
    detailItem('创建时间', formatTime(task.created_at)),
    detailItem('英文 instruction', task.instruction || task.instruction_en || '', 'wide'),
    detailItem('中文任务描述', task.description || task.description_zh || '', 'wide'),
    detailItem('数据目录', robotDir, 'wide path'),
    detailItem('LeRobot 目录', lerobotDir, 'wide path'),
    detailItem('归一化目录', normStatsDir, 'wide path'),
    detailItem('数据格式', 'robot'),
    detailItem('相机映射', status.camera_map || 'auto', 'wide path'),
    detailItem('image-size', status.image_size || 'original'),
    detailItem('原始图像尺寸', imageShapeText(status.source_image_shape)),
    detailItem('映射更新时间', formatTime(status.camera_map_updated_at)),
    detailItem('归档状态', archiveStatusText(task)),
    detailItem('归档开始', formatTime(task.archive_status?.started_at)),
    detailItem('归档完成', formatTime(task.archive_status?.finished_at)),
    detailItem('归档失败', formatTime(task.archive_status?.failed_at)),
    detailItem('归档错误', task.archive_status?.error || '—', 'wide'),
    detailItem('归档文件', task.archive_status?.last_archive || task.archive_status?.archive_path || '—', 'wide path'),
    detailItem('数据转换', convertStatusText(status)),
    detailItem('转换开始', formatTime(status.convert_started_at)),
    detailItem('转换完成', formatTime(status.converted_at)),
    detailItem('转换失败', formatTime(status.convert_failed_at)),
    detailItem('归一化', normalizeStatusText(status)),
    detailItem('归一化开始', formatTime(status.normalize_started_at)),
    detailItem('归一化完成', formatTime(status.normalized_at)),
    detailItem('归一化失败', formatTime(status.normalize_failed_at)),
  ].join('');
}
function renderTasks() {
  const status = $('#statusFilter').value, query = $('#taskSearch').value.trim().toLowerCase();
  const tasks = appState.tasks.filter(t => (!status || t.status === status) && (!query || t.name.toLowerCase().includes(query) || t.description.includes(query)));
  const totalPages = Math.max(1, Math.ceil(tasks.length / TASK_PAGE_SIZE));
  taskPage = Math.min(Math.max(taskPage, 1), totalPages);
  const pageTasks = tasks.slice((taskPage - 1) * TASK_PAGE_SIZE, taskPage * TASK_PAGE_SIZE);
  $('#taskRows').innerHTML = pageTasks.map(task => `<tr>
    ${(() => {
      const archiveText = task.archive_status?.running ? '归档中...' : (task.archive_status?.last_archive || task.archive_status?.failed_at) ? '重新归档' : '归档数据集';
      const runningArchiveJob = appState.archive_jobs?.find(job => job.running);
      const archiveBlockedByOther = Boolean(runningArchiveJob && runningArchiveJob.task_id !== task.id);
      const archiveDisabled = task.active || task.archive_status?.running;
      const archiveTitle = task.active
        ? '采集任务运行中，请先安全停止'
        : task.archive_status?.running
          ? '该任务正在归档'
          : archiveBlockedByOther
            ? `已有归档任务正在运行：${runningArchiveJob.task_name || '未知任务'}，请等待完成后再归档`
            : '';
      return `
    <td>${task.id}</td><td><strong>${escapeHtml(task.name)}</strong><span class="subline" title="${escapeHtml(task.description)}">${escapeHtml(task.description)}</span></td>
    <td>${escapeHtml(task.device_name)}</td><td><div class="progress-wrap"><div class="progress"><i style="width:${task.progress_percent}%"></i></div><span>${task.existing_episodes}/${task.target_episodes}</span></div></td>
    <td><span class="status ${statusClass(task.status)}">${task.status}</span></td><td>robot</td><td>${renderConvertStatus(task)}</td><td>${renderNormalizeStatus(task)}</td><td>${formatTime(task.created_at)}</td>
    <td><div class="actions"><button class="action" data-action="view-task" data-id="${task.id}">查看</button><button class="action" data-action="preview-data" data-id="${task.id}">数据预览</button><button class="action blue" data-action="start-task" data-id="${task.id}">${task.active ? '进入采集' : '开始采集'}</button><button class="action" data-action="archive-task" data-id="${task.id}" data-archive-blocked="${archiveBlockedByOther ? 'true' : 'false'}" title="${escapeHtml(archiveTitle)}" ${archiveDisabled ? 'disabled' : ''}>${archiveText}</button></div></td>`})()}</tr>`).join('');
  $('#taskEmpty').classList.toggle('visible', !tasks.length);
  const pagination = $('#taskPagination');
  if (pagination) {
    pagination.classList.toggle('visible', tasks.length > TASK_PAGE_SIZE);
    $('#taskPageInfo').textContent = `第 ${taskPage} / ${totalPages} 页，共 ${tasks.length} 个任务`;
    $('#prevTaskPage').disabled = taskPage <= 1;
    $('#nextTaskPage').disabled = taskPage >= totalPages;
    const pageJump = $('#taskPageJump');
    if (pageJump && document.activeElement !== pageJump) pageJump.value = String(taskPage);
    if (pageJump) pageJump.max = String(totalPages);
  }
}
function signatureText(profile, kind) {
  const metadata = profile.metadata || {};
  if (kind === 'control') {
    const stateDim = metadata.state_dim ?? '—';
    const actionDim = metadata.action_dim ?? '—';
    const vectorDims = (metadata.vector_dims || []).map(item => `${item.key}:${item.dim}`).join(', ');
    return `state ${stateDim} / action ${actionDim}${vectorDims ? ` · ${vectorDims}` : ''}`;
  }
  const keys = (metadata.image_keys || []).join(', ') || '—';
  const shape = Array.isArray(metadata.output_image_shape) && metadata.output_image_shape.length
    ? metadata.output_image_shape.join('x')
    : '—';
  return `${keys} · ${shape}`;
}
function trainingReadyText(profile) {
  if (profile.ready_for_training) return '已转换，可加入训练集';
  const warnings = profile.warnings || [];
  return warnings.length ? warnings.join('；') : '未准备好';
}
function processingProfiles() {
  const knownProfiles = new Map((appState.training?.profiles || []).map(profile => [Number(profile.task_id), profile]));
  return (appState.tasks || []).map(task => {
    const known = knownProfiles.get(Number(task.id)) || {};
    const status = task.postprocess_status || {};
    const converted = Boolean(status.lerobot_ready);
    return {
      ...known,
      task_id: task.id,
      task_name: known.task_name || task.name,
      repo_id: known.repo_id || status.lerobot_repo_id || `local/${task.name}`,
      instruction: known.instruction || task.instruction || task.description || '—',
      raw_episodes: known.raw_episodes ?? task.existing_episodes ?? 0,
      lerobot_episodes: status.lerobot_episodes ?? known.lerobot_episodes ?? 0,
      ready_for_training: converted,
      metadata: known.metadata || {},
      warnings: known.warnings || [],
      task,
    };
  });
}
function renderProcessingConvertProgress(profile, status) {
  const total = Number(status.lerobot_expected_episodes ?? profile.raw_episodes ?? 0);
  const done = Math.min(total || Number.MAX_SAFE_INTEGER, Math.max(0, Number(status.lerobot_episodes ?? profile.lerobot_episodes ?? 0)));
  const progress = total > 0
    ? {percent: Math.round((done / total) * 100), current: done, total, stage: convertStatusText(status)}
    : status.last_convert_record?.progress;
  return postprocessStatusProgress(progress, {
    running: Boolean(status.convert_running),
    complete: Boolean(status.lerobot_ready),
    failed: Boolean(status.convert_failed),
    idleText: convertStatusText(status),
  });
}
function renderProcessingNormalizeProgress(status) {
  const cancelled = Boolean(status.normalize_cancelled || isCancelledPostprocess(status.last_normalize_record));
  const running = Boolean(status.normalize_running);
  const progress = running ? (status.last_normalize_record?.progress || null) : null;
  return postprocessStatusProgress(progress, {
    running,
    complete: Boolean(status.norm_stats_ready),
    failed: Boolean(status.normalize_failed && !cancelled),
    idleText: normalizeStatusText(status),
  });
}
function renderTrainingSetNormalizeProgress(item, pendingTasks) {
  const normStats = item.norm_stats || {};
  const record = item.last_normalize_record || {};
  const running = Boolean(record.running || record.status === 'running');
  const failed = Boolean(record.status === 'failed' || (item.normalize_exit_code && item.normalize_exit_code !== 0));
  if (pendingTasks) {
    return postprocessStatusProgress(null, {
      running: false,
      complete: false,
      failed: false,
      idleText: '待加入任务',
    });
  }
  return postprocessStatusProgress(running ? (record.progress || null) : (record.progress || null), {
    running,
    complete: Boolean(normStats.ready),
    failed,
    idleText: failed ? '失败' : (normStats.stale ? '需重新归一化' : (normStats.ready ? '已完成' : '待归一化')),
  });
}
function trainingSetNormHint(item, pendingTasks) {
  if (pendingTasks) return '加入任务后再计算归一化';
  const record = item.last_normalize_record || {};
  const normStats = item.norm_stats || {};
  if (record.status === 'running' || record.running) return `归一化运行中：${formatTime(record.started_at)}`;
  if (record.status === 'failed' || (item.normalize_exit_code && item.normalize_exit_code !== 0)) {
    return `归一化失败：${record.error || `退出码 ${item.normalize_exit_code}`}`;
  }
  if (normStats.stale) return '训练集参数或成员已变化，需要重新计算归一化';
  if (normStats.ready) return `归一化已完成：${normStats.dir || ''}`;
  return '尚未计算训练集归一化';
}
function renderProcessingProfiles() {
  const box = $('#processingProfiles');
  if (!box) return;
  const profiles = processingProfiles();
  if (!profiles.length) {
    box.innerHTML = '<div class="data-preview-empty">暂无任务，请先在数据采集页创建并采集任务</div>';
    return;
  }
  const readyCount = profiles.filter(profile => profile.ready_for_training).length;
  const sets = appState.training?.training_sets || [];
  const targetOptions = sets.length
    ? sets.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join('')
    : '<option value="">先创建训练集</option>';
  box.innerHTML = `<div class="processing-profile-summary">
    <span>共 ${profiles.length} 个任务</span>
    <strong>${readyCount} 个已完成 LeRobot 转换</strong>
    <div class="training-add-bar">
      <select id="processingTrainingSetTarget" ${sets.length ? '' : 'disabled'}>${targetOptions}</select>
      <button class="action blue" data-action="add-selected-training" ${sets.length ? '' : 'disabled'}>加入训练集</button>
    </div>
  </div>
  <div class="processing-profile-table">
    <div class="processing-profile-row head">
      <span></span><span>任务</span><span>数据总量</span><span>LeRobot 转换</span><span>归一化计算</span><span>数据包</span><span>归一化包</span><span>动作/相机</span><span>instruction</span><span>操作</span>
    </div>
    ${profiles.map(profile => {
      const ready = Boolean(profile.ready_for_training);
      const task = profile.task || {};
      const status = task.postprocess_status || {};
      const controlText = signatureText(profile, 'control');
      const cameraText = signatureText(profile, 'camera');
      const instruction = profile.instruction || '—';
      return `<div class="processing-profile-row ${ready ? 'ready' : 'blocked'}">
        <span class="training-check"><input type="checkbox" name="processing_task" value="${escapeHtml(profile.task_id)}" ${ready ? '' : 'disabled'}></span>
        <span class="training-task-name"><strong>${escapeHtml(profile.task_name)}</strong><small>${escapeHtml(profile.repo_id)}</small></span>
        <span>${escapeHtml(profile.raw_episodes || 0)}</span>
        <span class="processing-progress-cell">${renderProcessingConvertProgress(profile, status)}</span>
        <span class="processing-progress-cell">${renderProcessingNormalizeProgress(status)}</span>
        <span>${renderDataPackageStatus(task)}</span>
        <span>${renderAssetsPackageStatus(task)}</span>
        <span title="${escapeHtml(`${controlText}；${cameraText}`)}">${escapeHtml(controlText)}<small>${escapeHtml(cameraText)}</small></span>
        <span title="${escapeHtml(instruction)}">${escapeHtml(instruction)}</span>
        <span class="processing-row-actions"><button class="action" data-action="preview-data" data-id="${escapeHtml(profile.task_id)}">预览</button><button class="action blue" data-action="postprocess-data" data-id="${escapeHtml(profile.task_id)}">数据转换</button></span>
      </div>`;
    }).join('')}
  </div>`;
}
function renderDataList() {
  const taskSelect = $('#dataListTaskSelect');
  const pageSizeSelect = $('#dataListPageSize');
  const rows = $('#dataListRows');
  const empty = $('#dataListEmpty');
  const subtitle = $('#dataListSubtitle');
  if (!taskSelect || !rows || !empty) return;
  const tasks = dataListState.tasks.length ? dataListState.tasks : (appState.tasks || []);
  const selected = String(dataListState.taskId || tasks[0]?.id || '');
  taskSelect.innerHTML = tasks.length
    ? tasks.map(task => `<option value="${escapeHtml(task.id)}" ${String(task.id) === selected ? 'selected' : ''}>${escapeHtml(task.name)}</option>`).join('')
    : '<option value="">暂无任务</option>';
  if (pageSizeSelect) pageSizeSelect.value = String(dataListState.pageSize || 50);
  const totalPages = Math.max(1, Math.ceil((dataListState.total || 0) / (dataListState.pageSize || 50)));
  $('#dataListPagination')?.classList.toggle('visible', totalPages > 1);
  $('#dataListPageInfo').textContent = `第 ${dataListState.page || 1} / ${totalPages} 页，共 ${dataListState.total || 0} 条`;
  $('#prevDataListPage').disabled = (dataListState.page || 1) <= 1;
  $('#nextDataListPage').disabled = (dataListState.page || 1) >= totalPages;
  const pageJump = $('#dataListPageJump');
  if (pageJump && document.activeElement !== pageJump) pageJump.value = String(dataListState.page || 1);
  if (pageJump) pageJump.max = String(totalPages);
  if (subtitle) {
    const task = tasks.find(item => String(item.id) === selected);
    subtitle.textContent = task ? `${task.name} · ${dataListState.dataset_dir || ''}` : '查看每条 episode 的帧数、文件数、大小并执行预览/删除';
  }
  const episodes = dataListState.episodes || [];
  empty.classList.toggle('visible', !episodes.length && !dataListState.loading);
  if (!episodes.length) {
    rows.innerHTML = dataListState.loading ? '<tr><td colspan="7">正在加载...</td></tr>' : '';
    return;
  }
  const task = tasks.find(item => String(item.id) === selected) || {};
  rows.innerHTML = episodes.map(item => {
    const statusText = item.error ? item.error : (item.completed ? '写入完成' : '写入中或异常');
    const statusKind = item.error || !item.completed ? 'idle' : 'done';
    const statusTitle = item.error || !item.completed
      ? '该 episode 的 data.json 未正常闭合，通常表示正在采集、保存中断或文件损坏'
      : '该 episode 的 data.json 已正常闭合，表示这条原始记录已完整写入';
    return `<tr>
      <td><strong>${escapeHtml(item.name)}</strong></td>
      <td>${escapeHtml(task.name || item.task_id)}</td>
      <td>${escapeHtml(item.frame_count ?? 0)}</td>
      <td>${escapeHtml(item.file_count ?? 0)}</td>
      <td>${escapeHtml(formatBytes(item.size_bytes))}</td>
      <td><span class="status ${statusKind}" title="${escapeHtml(statusTitle)}">${escapeHtml(statusText)}</span></td>
      <td><div class="actions"><button class="action" data-action="preview-episode" data-id="${escapeHtml(item.task_id)}" data-episode="${escapeHtml(item.name)}">预览</button><button class="action red" data-action="delete-episode" data-id="${escapeHtml(item.task_id)}" data-episode="${escapeHtml(item.name)}">删除</button></div></td>
    </tr>`;
  }).join('');
}
async function loadDataList({silent = true} = {}) {
  if (dataListState.loading) return;
  dataListState.loading = true;
  if (activeView === 'dataList') renderDataList();
  try {
    const result = await api('/api/data/list', {
      task_id: dataListState.taskId,
      page: dataListState.page,
      page_size: dataListState.pageSize,
    });
    dataListState = {
      ...dataListState,
      loading: false,
      taskId: String(result.task?.id || dataListState.taskId || ''),
      page: result.page || 1,
      pageSize: result.page_size || dataListState.pageSize || 50,
      total: result.total || 0,
      episodes: result.episodes || [],
      tasks: result.tasks || [],
      dataset_dir: result.dataset_dir || '',
    };
    renderDataList();
  } catch (error) {
    dataListState.loading = false;
    if (!silent) showNotice(error.message);
    renderDataList();
  }
}
function renderTrainingSets() {
  const rootHint = $('#trainingRootHint');
  if (rootHint) rootHint.textContent = appState.training?.root || '—';
  const box = $('#trainingSets');
  if (!box) return;
  if (!appState.training) {
    box.innerHTML = '<div class="data-preview-empty">等待服务重启后启用训练集和训练交接包管理</div>';
    return;
  }
  const sets = appState.training?.training_sets || [];
  const packages = appState.training?.packages || [];
  if (!sets.length) {
    box.innerHTML = '<div class="data-preview-empty">尚未创建训练集</div>';
    return;
  }
  box.innerHTML = `<div class="training-set-table">
    <div class="training-set-row head">
      <span>训练集</span><span>config-name</span><span>状态</span><span>任务数</span><span>归一化</span><span>交接包</span><span>操作</span>
    </div>
    ${sets.map(item => {
    const compatibility = item.compatibility || {};
    const normStats = item.norm_stats || {};
    const relatedPackages = packages.filter(pkg => pkg.training_set_id === item.id);
    const taskRows = item.tasks || [];
    const pendingTasks = taskRows.length === 0 || item.mode === 'empty' || ['empty', 'pending'].includes(compatibility.status);
    const compatible = !pendingTasks && compatibility.compatible === true;
    const issues = compatibility.issues || [];
    const typeText = item.mode === 'empty' ? '空训练集' : (item.mode === 'single_task' ? '单任务' : '多任务');
    const stateText = pendingTasks ? '待加入任务' : (compatible ? typeText : '不兼容');
    const stateClass = pendingTasks ? 'pending' : (compatible ? 'ok' : 'error');
    const stateTitle = pendingTasks
      ? '加入任务后校验动作维度和相机 schema'
      : (issues.join('；') || '动作维度和相机 schema 一致');
    const packageText = relatedPackages[0]?.path || '—';
    const taskTitle = taskRows.map(task => `${task.task_name || task.repo_id}: ${task.instruction || ''}`).join('；') || '尚未加入任务';
    const normalizeRecord = item.last_normalize_record || {};
    const normalizeRunning = Boolean(normalizeRecord.running || normalizeRecord.status === 'running');
    const normalizeDisabled = pendingTasks || !compatible || normalizeRunning;
    const packageReady = compatible && Boolean(normStats.ready);
    const normalizeTitle = pendingTasks
      ? '请先加入任务'
      : (!compatible ? stateTitle : (normalizeRunning ? '归一化正在运行' : '计算训练集归一化'));
    const packageTitle = pendingTasks
      ? '请先加入任务'
      : (!compatible ? stateTitle : (normStats.ready ? '生成训练交接包' : '请先计算训练集归一化'));
    const expanded = expandedTrainingSetIds.has(String(item.id));
    const memberRows = taskRows.length
      ? taskRows.map(task => `<div class="training-member-row">
          <span class="training-member-name"><strong>${escapeHtml(task.task_name || task.repo_id)}</strong><small>${escapeHtml(task.repo_id || '')}</small></span>
          <span title="${escapeHtml(task.instruction || '')}">${escapeHtml(task.instruction || '—')}</span>
          <span>${escapeHtml(task.raw_episodes ?? '—')}</span>
          <span><button class="action danger-text" data-action="remove-training-task" data-id="${escapeHtml(item.id)}" data-task-id="${escapeHtml(task.task_id)}">移除</button></span>
        </div>`).join('')
      : '<div class="data-preview-empty compact">尚未加入任务</div>';
    const normHint = trainingSetNormHint(item, pendingTasks);
    return `<div class="training-set-row">
      <span class="training-set-name" title="${escapeHtml(taskTitle)}"><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(taskRows.map(task => task.task_name || task.repo_id).join('；') || '尚未加入任务')}</small></span>
      <span title="${escapeHtml(item.config_name)}">${escapeHtml(item.config_name)}</span>
      <span><em class="${stateClass}" title="${escapeHtml(stateTitle)}">${escapeHtml(stateText)}</em></span>
      <span>${escapeHtml(taskRows.length)}</span>
      <span class="training-norm-cell" title="${escapeHtml(normHint)}">${renderTrainingSetNormalizeProgress(item, pendingTasks)}<small>${escapeHtml(normHint)}</small></span>
      <span title="${escapeHtml(packageText)}">${escapeHtml(packageText)}</span>
      <span class="training-row-actions"><button class="action" data-action="toggle-training-members" data-id="${escapeHtml(item.id)}">${expanded ? '收起成员' : '成员'}</button><button class="action" data-action="edit-training" data-id="${escapeHtml(item.id)}">编辑</button><button class="action" data-action="normalize-training" data-id="${escapeHtml(item.id)}" title="${escapeHtml(normalizeTitle)}" ${normalizeDisabled ? 'disabled' : ''}>计算归一化</button><button class="action blue" data-action="package-training" data-id="${escapeHtml(item.id)}" title="${escapeHtml(packageTitle)}" ${packageReady ? '' : 'disabled'}>生成包</button></span>
    </div>${expanded ? `<div class="training-set-detail">
      <div class="training-member-head"><span>任务成员</span><span>instruction</span><span>原始数据</span><span>操作</span></div>
      ${memberRows}
    </div>` : ''}`;
  }).join('')}</div>`;
}
function packageTaskName(name) {
  const text = String(name || '').replace(/\.(tar\.gz|tgz)$/i, '');
  return text.split('_lerobot_')[0].split('_openpi_assets_')[0] || text;
}
function cleanTaskName(value) {
  return String(value || '').trim().replace(/^\/+|\/+$/g, '');
}
function joinLocalPath(root, taskName = '') {
  const base = String(root || '').replace(/\/+$/g, '');
  const task = cleanTaskName(taskName);
  return task ? `${base}/${task}` : base;
}
function joinOssPath(root, taskName = '') {
  const base = String(root || DEFAULT_OSS_ROOT).replace(/\/+$/g, '');
  const task = cleanTaskName(taskName);
  return task ? `${base}/${task}` : base;
}
function ossTaskOptions(selectedTaskName = '') {
  const names = new Set();
  if (selectedTaskName) names.add(selectedTaskName);
  (appState.tasks || []).forEach(task => {
    if (task.name) names.add(task.name);
  });
  return [...names].filter(Boolean);
}
function taskByName(taskName) {
  const name = cleanTaskName(taskName);
  return (appState.tasks || []).find(task => task.name === name) || null;
}
function manualDerivedValues(taskName, values = {}) {
  const task = taskName || DEFAULT_TRAINING_TASK_NAME;
  const numTrainSteps = values.numTrainSteps || TRAINING_COMMAND_DEFAULTS.numTrainSteps;
  const expName = `pi05_${task}_${numTrainSteps}`;
  const packageTimestamp = values.packageTimestamp || currentTimestampText();
  const dataPackageSuffix = String(values.dataPackage || '').match(/(_lerobot_.+\.(?:tar\.gz|tgz))$/i)?.[1] || '_lerobot_YYYYMMDD_HHMMSS.tar.gz';
  return {
    taskName: task,
    dataPackage: `${task}${dataPackageSuffix}`,
    repoId: `local/${task}`,
    expName,
    modelFile: `${expName}_${packageTimestamp}.tar.gz`,
  };
}
function manualTaskOptions(selectedTaskName = '') {
  const names = (appState.tasks || []).map(task => task.name).filter(Boolean);
  const selected = cleanTaskName(selectedTaskName);
  if (selected && !names.includes(selected)) names.unshift(selected);
  return names.length
    ? names.map(name => `<option value="${escapeHtml(name)}" ${name === selected ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('')
    : `<option value="${escapeHtml(selected || DEFAULT_TRAINING_TASK_NAME)}">${escapeHtml(selected || DEFAULT_TRAINING_TASK_NAME)}</option>`;
}
function shellQuote(value) {
  const text = String(value || '');
  return `'${text.replace(/'/g, `'\\''`)}'`;
}
function currentTimestampText() {
  const date = new Date();
  const pad = value => String(value).padStart(2, '0');
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}
function renderManualCommandBlock(title, description, command) {
  return `<article class="manual-command">
    <div class="manual-command-copy"><button type="button" title="复制命令" aria-label="复制命令" data-action="copy-text" data-copy="${escapeHtml(command)}">复制</button></div>
    <div class="manual-command-head"><h4>${escapeHtml(title)}</h4><p>${escapeHtml(description || '')}</p></div>
    <pre><code>${escapeHtml(command)}</code></pre>
  </article>`;
}
function deliveryTemplates() {
  return Array.isArray(appState.delivery?.templates) ? appState.delivery.templates : [];
}
function deliveryTemplateSectionTitle(section) {
  return {
    data_upload: '一、数据上传',
    training: '二、模型训练',
    model_return: '三、模型回传',
    model_deploy: '四、模型部署',
  }[section] || '其他命令';
}
function deliveryTemplateSection(item) {
  const section = item?.section || 'training';
  if (section === 'model_deploy') return section;
  if (String(item?.title || '').includes('模型部署')) return 'model_deploy';
  return section;
}
function renderTemplateBody(body, values) {
  return String(body || '').replace(/\{\{([A-Z0-9_]+)\}\}/g, (_, key) => {
    return values[key] ?? '';
  });
}
function buildManualValues(input) {
  const task = input.taskName || DEFAULT_TRAINING_TASK_NAME;
  const derived = manualDerivedValues(task, input);
  const exp = input.expName || derived.expName;
  const numTrainSteps = input.numTrainSteps || TRAINING_COMMAND_DEFAULTS.numTrainSteps;
  const modelTrainDir = input.modelTrainDir || String(Math.max(0, (Number(numTrainSteps) || Number(TRAINING_COMMAND_DEFAULTS.numTrainSteps)) - 1));
  const packageTimestamp = input.packageTimestamp || currentTimestampText();
  const datasetFile = input.dataPackage || derived.dataPackage;
  const modelName = input.modelFile || `${exp}_${packageTimestamp}.tar.gz`;
  const root = (input.ossRoot || 'oss://bwton-idc/openpi').replace(/\/+$/, '');
  return {
    TASK_NAME: task,
    INSTRUCTION: input.instruction || taskByName(task)?.instruction || '',
    DATA_PACKAGE: datasetFile,
    MODEL_FILE: modelName,
    OSS_ROOT: root,
    DATA_OSS_URI: `${root}/${task}/${datasetFile}`,
    MODEL_OSS_URI: `${root}/${task}/${modelName}`,
    PROXY_PACKAGE_ROOT: input.proxyPackageRoot || '/opt/packages/openpi',
    TRAIN_HOST: input.trainHost || 'dgzs-docker-gpu11.prod-2227',
    TRAIN_PACKAGE_DIR: input.trainPackageDir || '/home/ubuntu/packages/openpi',
    TRAIN_LEROBOT_HOME: input.trainLerobotHome || '/home/ubuntu/datasets/lerobot',
    OPENPI_DIR: input.openpiDir || '/home/ubuntu/openpi',
    OPENPI_DATA_HOME: input.openpiDataHome || '/home/ubuntu/models/openpi',
    HF_HOME: input.hfHome || '/home/ubuntu/models/openpi/huggingface',
    ASSETS_DIR: input.assetsDir || '/home/ubuntu/assets',
    CHECKPOINT_DIR: input.checkpointDir || '/home/ubuntu/models/openpi/checkpoints',
    LOG_DIR: input.logDir || '/home/ubuntu/models/openpi/logs',
    REPO_ID: input.repoId || derived.repoId,
    CONFIG_NAME: input.configName || TRAINING_COMMAND_DEFAULTS.configName,
    EXP_NAME: exp,
    ACTION_DIM: input.actionDim || TRAINING_COMMAND_DEFAULTS.actionDim,
    REAL_ACTION_DIM: input.realActionDim || TRAINING_COMMAND_DEFAULTS.realActionDim,
    ACTION_HORIZON: input.actionHorizon || TRAINING_COMMAND_DEFAULTS.actionHorizon,
    FSDP_DEVICES: input.fsdpDevices || TRAINING_COMMAND_DEFAULTS.fsdpDevices,
    BATCH_SIZE: input.batchSize || TRAINING_COMMAND_DEFAULTS.batchSize,
    NUM_TRAIN_STEPS: numTrainSteps,
    MODEL_TRAIN_DIR: modelTrainDir,
    PACKAGE_TIMESTAMP: packageTimestamp,
    SAVE_INTERVAL: input.saveInterval || TRAINING_COMMAND_DEFAULTS.saveInterval,
    KEEP_PERIOD: input.keepPeriod || TRAINING_COMMAND_DEFAULTS.keepPeriod,
    TARGET_HOST: input.targetHost || 'robot@192.168.61.228',
    TARGET_MODEL_DIR: input.targetModelDir || '/data03/data/models/openpi_downloads',
  };
}
function deliveryTemplateRowHtml(item, index) {
  const section = deliveryTemplateSection(item);
  return `<article class="template-editor-row" data-template-index="${index}" data-template-id="${escapeHtml(item.id || `template_${index + 1}`)}">
    <div class="template-editor-meta">
      <label><span>标题</span><input class="template-title" value="${escapeHtml(item.title || '')}"></label>
      <label><span>分组</span><select class="template-section">
        <option value="data_upload" ${section === 'data_upload' ? 'selected' : ''}>数据上传</option>
        <option value="training" ${section === 'training' ? 'selected' : ''}>模型训练</option>
        <option value="model_return" ${section === 'model_return' ? 'selected' : ''}>模型回传</option>
        <option value="model_deploy" ${section === 'model_deploy' ? 'selected' : ''}>模型部署</option>
      </select></label>
      <label><span>说明</span><input class="template-description" value="${escapeHtml(item.description || '')}"></label>
      <button class="icon-button danger template-delete" type="button" data-action="delivery-delete-template" title="删除标题">×</button>
    </div>
    <textarea class="template-body" spellcheck="false">${escapeHtml(item.body || '')}</textarea>
  </article>`;
}
function blankDeliveryTemplate() {
  return {
    id: `custom_${Date.now()}`,
    section: 'training',
    title: '新模板标题',
    description: '填写这段命令的用途',
    body: [
      '# 在这里编写命令模板',
      '# 可使用 {{TASK_NAME}}、{{INSTRUCTION}}、{{REPO_ID}}、{{DATA_PACKAGE}} 等变量',
      '',
    ].join('\n'),
  };
}
function renderDeliveryTemplateEditors(templates) {
  const rows = templates.map((item, index) => deliveryTemplateRowHtml(item, index)).join('');
  return `<details class="manual-template-panel" open>
    <summary>命令模板</summary>
    <div class="template-toolbar">
      <div class="template-help">模板保存在 ${escapeHtml(appState.delivery?.path || 'config/delivery_templates.json')}；可使用 {{TASK_NAME}}、{{INSTRUCTION}}、{{DATA_PACKAGE}}、{{MODEL_FILE}}、{{REPO_ID}}、{{NUM_TRAIN_STEPS}}、{{MODEL_TRAIN_DIR}}、{{PACKAGE_TIMESTAMP}} 等变量。</div>
      <button class="action template-add" type="button" data-action="delivery-add-template">＋ 新增标题</button>
    </div>
    <div class="template-editor-list">${rows || '<div class="data-preview-empty compact">暂无模板</div>'}</div>
    <div class="manual-doc-actions"><button class="action" type="button" data-action="delivery-reset-templates">恢复默认模板</button><button class="action blue" type="button" data-action="delivery-save-templates">保存模板</button></div>
  </details>`;
}
function addDeliveryTemplateEditor() {
  const list = $('.template-editor-list');
  if (!list) return;
  list.querySelector('.data-preview-empty')?.remove();
  const index = $$('.template-editor-row').length;
  list.insertAdjacentHTML('beforeend', deliveryTemplateRowHtml(blankDeliveryTemplate(), index));
  const row = list.lastElementChild;
  row?.querySelector('.template-title')?.focus();
  showNotice('已新增模板标题，点击保存后生效', 'success');
}
function deleteDeliveryTemplateEditor(button) {
  const row = button.closest('.template-editor-row');
  const list = row?.parentElement;
  row?.remove();
  if (list && !list.querySelector('.template-editor-row')) {
    list.innerHTML = '<div class="data-preview-empty compact">暂无模板</div>';
  }
  showNotice('已删除模板标题，点击保存后生效', 'success');
}
function collectDeliveryTemplates() {
  return $$('.template-editor-row').map((row, index) => ({
    id: row.dataset.templateId || `template_${index + 1}`,
    section: row.querySelector('.template-section')?.value || 'training',
    title: row.querySelector('.template-title')?.value || '',
    description: row.querySelector('.template-description')?.value || '',
    body: row.querySelector('.template-body')?.value || '',
  }));
}
function currentManualValues() {
  const transfer = appState.oss_transfer || {};
  const packages = (transfer.local_entries || []).filter(item => item.is_package);
  const firstPackage = packages[0] || (transfer.packages || [])[0] || {};
  const defaultTaskName = (appState.tasks || [])[0]?.name || packageTaskName(firstPackage.name || '');
  const taskName = $('#manualTaskName')?.value || defaultTaskName || DEFAULT_TRAINING_TASK_NAME;
  const matchedPackage = packages.find(item => packageTaskName(item.name || '') === taskName)
    || (transfer.packages || []).find(item => packageTaskName(item.name || '') === taskName);
  const derived = manualDerivedValues(taskName, {dataPackage: matchedPackage?.name || firstPackage.name || '', numTrainSteps: $('#manualNumTrainSteps')?.value || TRAINING_COMMAND_DEFAULTS.numTrainSteps});
  const ossRoot = transfer.oss_root || 'oss://bwton-idc/openpi';
  return {
    taskName,
    instruction: $('#manualInstruction')?.value || taskByName(taskName)?.instruction || '',
    dataPackage: $('#manualDataPackage')?.value || matchedPackage?.name || derived.dataPackage,
    modelFile: $('#manualModelFile')?.value || '',
    ossRoot: $('#manualOssRoot')?.value || ossRoot,
    repoId: $('#manualRepoId')?.value || derived.repoId,
    configName: $('#manualConfigName')?.value || TRAINING_COMMAND_DEFAULTS.configName,
    actionDim: $('#manualActionDim')?.value || TRAINING_COMMAND_DEFAULTS.actionDim,
    realActionDim: $('#manualRealActionDim')?.value || TRAINING_COMMAND_DEFAULTS.realActionDim,
    actionHorizon: $('#manualActionHorizon')?.value || TRAINING_COMMAND_DEFAULTS.actionHorizon,
    expName: $('#manualExpName')?.value || derived.expName,
    numTrainSteps: $('#manualNumTrainSteps')?.value || TRAINING_COMMAND_DEFAULTS.numTrainSteps,
    modelTrainDir: $('#manualModelTrainDir')?.value || '',
    packageTimestamp: $('#manualPackageTimestamp')?.value || currentTimestampText(),
  };
}
function updateManualModelFileName() {
  const modelInput = $('#manualModelFile');
  if (!modelInput) return;
  const values = buildManualValues({...currentManualValues(), modelFile: ''});
  modelInput.value = values.MODEL_FILE;
}
function syncManualDerivedFieldsFromTask() {
  const taskInput = $('#manualTaskName');
  if (!taskInput) return;
  const current = currentManualValues();
  const derived = manualDerivedValues(taskInput.value, current);
  const dataInput = $('#manualDataPackage');
  const repoInput = $('#manualRepoId');
  const expInput = $('#manualExpName');
  const modelInput = $('#manualModelFile');
  const instructionInput = $('#manualInstruction');
  if (dataInput) dataInput.value = derived.dataPackage;
  if (repoInput) repoInput.value = derived.repoId;
  if (expInput) expInput.value = derived.expName;
  if (modelInput) modelInput.value = derived.modelFile;
  if (instructionInput) instructionInput.value = taskByName(taskInput.value)?.instruction || '';
}
function syncManualExpNameFromSteps() {
  const taskInput = $('#manualTaskName');
  const expInput = $('#manualExpName');
  if (!taskInput || !expInput) return;
  const current = currentManualValues();
  expInput.value = manualDerivedValues(taskInput.value, {...current, expName: ''}).expName;
  updateManualModelFileName();
}
function syncManualModelTrainDir() {
  const stepsInput = $('#manualNumTrainSteps');
  const dirInput = $('#manualModelTrainDir');
  if (!stepsInput || !dirInput) return;
  const steps = Number(stepsInput.value);
  if (!Number.isFinite(steps) || steps < 1) return;
  dirInput.value = String(Math.max(0, Math.floor(steps) - 1));
  updateManualModelFileName();
}
function renderManualCommands(input) {
  const values = buildManualValues(input);
  const templates = deliveryTemplates();
  const grouped = ['data_upload', 'training', 'model_return', 'model_deploy'].map(section => {
    const sectionTemplates = templates.filter(item => deliveryTemplateSection(item) === section);
    if (!sectionTemplates.length) return '';
    return `<div class="manual-command-section"><h3>${deliveryTemplateSectionTitle(section)}</h3>${sectionTemplates.map(item => renderManualCommandBlock(
      item.title || '未命名命令',
      item.description || '',
      renderTemplateBody(item.body || '', values).trim(),
    )).join('')}</div>`;
  }).join('');
  const modelName = values.MODEL_FILE;
  const datasetFile = values.DATA_PACKAGE;
  const taskOptions = manualTaskOptions(values.TASK_NAME);
  return `<section class="manual-doc-card training-command-card">
    <details class="manual-param-panel" open>
      <summary>交付参数</summary>
    <div class="manual-doc-form">
      <label><span>任务名</span><select id="manualTaskName">${taskOptions}</select></label>
      <label><span>instruction</span><input id="manualInstruction" value="${escapeHtml(values.INSTRUCTION)}"></label>
      <label><span>数据压缩包文件名</span><input id="manualDataPackage" value="${escapeHtml(datasetFile)}"></label>
      <label><span>模型文件名</span><input id="manualModelFile" value="${escapeHtml(modelName)}"></label>
      <label><span>OSS 根目录</span><input id="manualOssRoot" value="${escapeHtml(values.OSS_ROOT)}"></label>
      <label><span>LeRobot repo-id</span><input id="manualRepoId" value="${escapeHtml(values.REPO_ID)}"></label>
      <label><span>OpenPI config-name</span><input id="manualConfigName" value="${escapeHtml(values.CONFIG_NAME)}"></label>
      <label><span>ACTION_DIM</span><input id="manualActionDim" value="${escapeHtml(values.ACTION_DIM)}"></label>
      <label><span>REAL_ACTION_DIM</span><input id="manualRealActionDim" value="${escapeHtml(values.REAL_ACTION_DIM)}"></label>
      <label><span>ACTION_HORIZON</span><input id="manualActionHorizon" value="${escapeHtml(values.ACTION_HORIZON)}"></label>
      <label><span>exp-name</span><input id="manualExpName" value="${escapeHtml(values.EXP_NAME)}"></label>
      <label><span>num-train-steps</span><input id="manualNumTrainSteps" type="number" min="1" value="${escapeHtml(values.NUM_TRAIN_STEPS)}"></label>
      <label><span>模型训练目录</span><input id="manualModelTrainDir" value="${escapeHtml(values.MODEL_TRAIN_DIR)}"></label>
      <label><span>打包时间戳 <button class="action" type="button" data-action="timestamp-now">当前</button></span><input id="manualPackageTimestamp" pattern="\\d{8}_\\d{6}" value="${escapeHtml(values.PACKAGE_TIMESTAMP)}"></label>
    </div>
    <div class="manual-doc-actions"><button class="action blue" data-action="manual-refresh">生成命令</button></div>
    </details>
    ${grouped || '<div class="data-preview-empty compact">暂无命令模板，请先恢复默认模板</div>'}
  </section>`;
}
function renderTrainingTemplateView() {
  const panel = $('#trainingTemplatePanel');
  if (!panel) return;
  const templates = deliveryTemplates();
  panel.innerHTML = `<section class="oss-card wide manual-doc-card">
    <div class="oss-card-head"><div><h3>训练模板</h3><p>这里维护模板骨架；服务器路径、训练步数、batch 等固定项直接写在模板里，最终命令在“训练命令”页面生成。</p></div></div>
    ${renderDeliveryTemplateEditors(templates)}
  </section>`;
}
function renderTrainingCommandView() {
  const panel = $('#trainingCommandPanel');
  if (panel) panel.innerHTML = renderManualCommands(currentManualValues());
}
function renderOssTransfer() {
  const box = $('#ossTransferPanel');
  if (!box) return;
  const transfer = appState.oss_transfer || {};
  const taskCandidates = ossTaskOptions(ossTransferState.taskName);
  const inferredTaskName = ossTransferState.taskName || taskCandidates[0] || packageTaskName((transfer.packages || [])[0]?.name || (transfer.local_entries || []).find(item => item.is_package)?.name || '');
  const taskName = cleanTaskName(inferredTaskName);
  if (taskName && !ossTransferState.taskName) ossTransferState.taskName = taskName;
  const localBaseDir = DEFAULT_OSS_PACKAGE_ROOT;
  const localDir = ossTransferState.localDir || joinLocalPath(localBaseDir, taskName);
  const localParent = transfer.local_parent || localBaseDir;
  const entries = transfer.local_entries || [];
  const directories = entries.filter(item => item.is_dir);
  const packages = entries.filter(item => item.is_package);
  const ossRoot = transfer.oss_root || DEFAULT_OSS_ROOT;
  const taskOssUri = joinOssPath(ossRoot, taskName);
  const remoteUri = ossTransferState.remoteUri || taskOssUri;
  const downloadDir = joinLocalPath(DEFAULT_MODEL_DOWNLOAD_ROOT, taskName);
  const taskOptions = taskCandidates.length
    ? taskCandidates.map(name => `<option value="${escapeHtml(name)}" ${name === taskName ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('')
    : `<option value="${escapeHtml(taskName || DEFAULT_OSS_TASK_NAME)}">${escapeHtml(taskName || DEFAULT_OSS_TASK_NAME)}</option>`;
  const remoteFiles = (ossTransferState.remoteEntries || []).filter(item => !item.is_dir);
  const remoteOptions = remoteFiles.length
    ? remoteFiles.map(item => `<option value="${escapeHtml(item.uri)}">${escapeHtml(item.uri)}</option>`).join('')
    : '<option value="">请先刷新 OSS 列表</option>';
  const dirRows = directories.length
    ? directories.map(item => `<button class="oss-dir-item" type="button" data-action="oss-open-local-dir" data-path="${escapeHtml(item.path)}">${escapeHtml(item.name)}</button>`).join('')
    : '<span class="oss-empty-inline">无子目录</span>';
  const packageRows = packages.length
    ? packages.map(item => `<div class="oss-package-row">
        <span class="oss-package-name"><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.path)} · ${escapeHtml(formatBytes(item.size_bytes))}</small></span>
        <button class="action blue" data-action="oss-upload" data-local-path="${escapeHtml(item.path)}" data-package-name="${escapeHtml(item.name)}">上传</button>
      </div>`).join('')
    : '<div class="data-preview-empty compact">当前目录没有 .tar.gz/.tgz 压缩包</div>';
  const transferJobs = (appState.postprocess_jobs || []).filter(job => ['oss_upload', 'oss_download'].includes(job.kind));
  const jobRows = transferJobs.length
    ? transferJobs.slice(0, 30).map(job => {
      const failed = !job.running && job.exit_code !== 0 && !isCancelledPostprocess(job);
      const complete = !job.running && job.exit_code === 0;
      return `<div class="oss-job-row">
        <strong>${escapeHtml(job.kind === 'oss_upload' ? '上传' : '下载')}</strong>
        <span>${escapeHtml(postprocessRecordStatusText(job))}</span>
        <div class="oss-job-progress">${postprocessStatusProgress(job.progress, {
          running: Boolean(job.running),
          complete,
          failed,
          idleText: postprocessRecordStatusText(job),
        })}</div>
        <small>${escapeHtml(formatTime(job.started_at))}${job.finished_at ? ` → ${escapeHtml(formatTime(job.finished_at))}` : ''}</small>
        <div class="oss-job-paths">
          <div title="${escapeHtml(job.local_path || '')}"><b>本地</b><code>${escapeHtml(job.local_path || '—')}</code></div>
          <div title="${escapeHtml(job.oss_uri || '')}"><b>OSS</b><code>${escapeHtml(job.oss_uri || '—')}</code></div>
        </div>
      </div>`;
    }).join('')
    : '<div class="data-preview-empty compact">暂无 OSS 上传或下载任务</div>';
  box.innerHTML = `<div class="oss-workbench">
    <section class="oss-card wide">
      <div class="oss-card-head"><div><h3>任务目录</h3><p>选择任务名后，本地包目录、OSS 目录、模型下载目录会自动追加同名子目录</p></div></div>
      <div class="oss-transfer-grid oss-task-dir-grid">
        <label><span>任务名</span><select id="ossTaskNameInput">${taskOptions}</select></label>
        <label><span>本地包根目录</span><input id="ossLocalBaseInput" value="${escapeHtml(localBaseDir)}" title="${escapeHtml(localBaseDir)}" readonly></label>
        <label><span>OSS 根目录</span><input id="ossRootInput" value="${escapeHtml(ossRoot)}" title="${escapeHtml(ossRoot)}" placeholder="${escapeHtml(DEFAULT_OSS_ROOT)}"></label>
        <label><span>模型下载根目录</span><input id="ossDownloadBaseInput" value="${escapeHtml(DEFAULT_MODEL_DOWNLOAD_ROOT)}" title="${escapeHtml(DEFAULT_MODEL_DOWNLOAD_ROOT)}" readonly></label>
      </div>
    </section>
    <section class="oss-card">
      <div class="oss-card-head"><div><h3>数据上传</h3><p>选择本地压缩包，上传到 OSS 任务目录</p></div></div>
      <div class="oss-local-toolbar">
        <label><span>本地目录</span><input id="ossLocalDirInput" value="${escapeHtml(localDir)}" placeholder="${escapeHtml(joinLocalPath(DEFAULT_OSS_PACKAGE_ROOT, taskName))}"></label>
        <button class="action" data-action="oss-local-list">打开/刷新</button>
        <button class="action" data-action="oss-open-local-dir" data-path="${escapeHtml(localParent)}">上一级</button>
      </div>
      <div class="oss-dir-hint">点击下方子目录可进入；修改本地目录后点“打开/刷新”重新读取。</div>
      <div class="oss-dir-list">${dirRows}</div>
      <label class="oss-root-line"><span>OSS 上传目录</span><input id="ossUploadUriInput" value="${escapeHtml(taskOssUri)}" readonly><small>上传目标：OSS 根目录/任务名/</small></label>
      <div class="oss-package-list">${packageRows}</div>
    </section>
    <section class="oss-card">
      <div class="oss-card-head"><div><h3>模型回拉</h3><p>查看 OSS 目录，选择训练后的模型文件下载到本地</p></div></div>
      <div class="oss-transfer-grid model-download-grid">
        <label class="oss-list-uri"><span>OSS 查看目录</span><input id="ossListUriInput" value="${escapeHtml(remoteUri)}" placeholder="${escapeHtml(taskOssUri)}"></label>
        <div class="oss-actions"><button class="action" data-action="oss-list" ${ossTransferState.loading ? 'disabled' : ''}>${ossTransferState.loading ? '刷新中...' : '刷新 OSS 列表'}</button></div>
        <label><span>远端模型文件</span><select id="ossRemoteObject" ${remoteFiles.length ? '' : 'disabled'}>${remoteOptions}</select><small>${remoteFiles.length ? `共 ${remoteFiles.length} 个文件` : '使用 ossutil ls 查看目录后选择文件下载'}</small></label>
        <label><span>模型下载目录</span><input id="ossDownloadDir" value="${escapeHtml(downloadDir)}" placeholder="${escapeHtml(joinLocalPath(DEFAULT_MODEL_DOWNLOAD_ROOT, taskName))}"><small>模型文件会下载到该任务目录</small></label>
        <div class="oss-actions"><button class="action blue" data-action="oss-download" ${remoteFiles.length ? '' : 'disabled'}>下载模型</button></div>
      </div>
    </section>
    <section class="oss-card wide"><div class="oss-card-head"><div><h3>传输任务记录</h3><p>保留历史 OSS 上传和模型下载任务，运行中任务显示当前进度</p></div></div><div class="oss-job-list">${jobRows}</div></section>
  </div>`;
}
async function refreshOssLocalDir(directory, {notice = false} = {}) {
  const result = await api('/api/oss/local-packages', {directory});
  appState.oss_transfer = result;
  ossTransferState.localDir = result.local_dir || directory || '';
  if (notice) showNotice(`已打开目录：${ossTransferState.localDir}`, 'success');
  renderOssTransfer();
  return result;
}
async function ensureOssTaskLocalDir() {
  if (activeView !== 'oss') return;
  const taskName = cleanTaskName($('#ossTaskNameInput')?.value || ossTransferState.taskName || ossTaskOptions()[0] || DEFAULT_OSS_TASK_NAME);
  const expectedDir = joinLocalPath(DEFAULT_OSS_PACKAGE_ROOT, taskName);
  const currentDir = appState.oss_transfer?.local_dir || '';
  if (currentDir !== expectedDir) {
    ossTransferState.taskName = taskName;
    ossTransferState.localDir = expectedDir;
    await refreshOssLocalDir(expectedDir);
  }
}
function renderTraining() {
  renderTrainingSets();
}
function renderOssView() {
  const title = $('#ossView .page-title h1');
  const subtitle = $('#ossView .page-title p');
  const refreshButton = $('#refreshOssTransfer');
  const panelTitle = $('#ossView .training-panel-head h2');
  const panelSubtitle = $('#ossView .training-panel-head p');
  if (title) title.textContent = 'OSS传输';
  if (subtitle) subtitle.textContent = '上传本地压缩包到训练环境，或从 OSS 回拉训练后的模型文件';
  if (refreshButton) refreshButton.textContent = '刷新';
  if (panelTitle) panelTitle.textContent = '数据上传与模型回拉';
  if (panelSubtitle) panelSubtitle.textContent = '本地压缩包来自数据处理/训练交接包；OSS 列表只在手动刷新时请求';
  renderOssTransfer();
  ensureOssTaskLocalDir().catch(error => showNotice(error.message));
}
function renderDataPreview(preview, episodeName = '') {
  const episodes = preview?.episodes || [];
  const selectedIndex = episodeName ? episodes.findIndex(item => item.name === episodeName) : 0;
  dataPreviewState = {
    preview,
    index: selectedIndex >= 0 ? selectedIndex : 0,
    taskId: preview?.task?.id ?? dataPreviewState.taskId,
    episode: episodeName || dataPreviewState.episode,
  };
  renderDataPreviewEpisode();
}
async function loadSingleEpisodePreview(taskId, episode, framePage = 1) {
  const preview = await api('/api/tasks/preview', {
    task_id: taskId,
    limit: 1,
    episode,
    frame_page: framePage,
    frame_page_size: 3,
  });
  renderDataPreview(preview, episode);
}
function renderDataPreviewEpisode() {
  const preview = dataPreviewState.preview || {};
  const task = preview.task || {};
  const singleEpisode = Boolean(preview.single_episode);
  $('#dataPreviewModal .data-preview-dialog')?.classList.toggle('frame-mode', singleEpisode);
  $('#dataPreviewTitle').textContent = `${singleEpisode ? '数据记录预览' : '数据预览'} · ${task.name || '未命名任务'}`;
  $('#dataPreviewSubtitle').textContent = preview.exists ? preview.dataset_dir : (preview.error || '该任务尚未生成数据集目录');
  const progress = preview.progress || {};
  const episodes = preview.episodes || [];
  const total = preview.episode_total ?? episodes.length;
  const index = Math.min(Math.max(dataPreviewState.index || 0, 0), Math.max(episodes.length - 1, 0));
  dataPreviewState.index = index;
  const currentEpisode = episodes[index];
  $('#dataPreviewSummary').innerHTML = [
    [singleEpisode ? 'episode' : '当前轮数', currentEpisode ? (singleEpisode ? currentEpisode.name : `${total - index} / ${total}`) : `0 / ${total}`],
    ['当前帧数', `${currentEpisode?.frame_count ?? 0}`],
    ['当前大小', currentEpisode ? formatBytes(currentEpisode.size_bytes) : '0 B'],
    [singleEpisode ? '文件数' : '总轮数', singleEpisode ? `${currentEpisode?.file_count ?? 0}` : `${total}`],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join('');
  if (!episodes.length) {
    $('#dataPreviewList').innerHTML = `<div class="data-preview-empty">${escapeHtml(preview.error || '暂无已保存 episode')}</div>`;
    return;
  }
  if (singleEpisode && preview.frame_page) {
    const framePage = preview.frame_page;
    const frames = framePage.frames || [];
    const firstFrame = frames[0]?.number ?? 0;
    const lastFrame = frames[frames.length - 1]?.number ?? 0;
    $('#dataPreviewTitle').textContent = currentEpisode.name;
    $('#dataPreviewSubtitle').textContent = `${task.name || '未命名任务'} · 第 ${framePage.page} / ${framePage.total_pages} 页 · ${firstFrame}-${lastFrame} / ${framePage.total} 帧`;
    $('#dataPreviewSummary').innerHTML = '';
    const frameRows = frames.map(frame => {
      const images = (frame.images || []).map((image, imageIndex) => `<article class="episode-frame-image">
        ${imageIndex === 0 ? `<b class="episode-frame-badge">第 ${escapeHtml(frame.number)} 帧</b>` : ''}
        <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name)}" loading="lazy">
        <span>${escapeHtml(image.name)}.jpg · ${formatBytes(image.size_bytes)}</span>
      </article>`).join('');
      return `<section class="episode-frame-row">
        <div class="episode-frame-images">${images || '<div class="data-preview-empty">当前帧暂无图片</div>'}</div>
      </section>`;
    }).join('');
    $('#dataPreviewList').innerHTML = `<section class="data-frame-preview">
      <div class="episode-preview-body">
        <div class="data-preview-nav">
          <button class="plain" data-action="episode-frame-prev" ${framePage.page <= 1 ? 'disabled' : ''}>上一页</button>
          <strong>第 ${escapeHtml(framePage.page)} / ${escapeHtml(framePage.total_pages)} 页，共 ${escapeHtml(framePage.total)} 帧</strong>
          <label class="frame-jump page-jump">跳至 <input id="episodeFramePageJumpInput" type="number" min="1" max="${escapeHtml(framePage.total_pages)}" value="${escapeHtml(framePage.page || 1)}"> 页</label>
          <button class="plain" data-action="episode-frame-next" ${framePage.page >= framePage.total_pages ? 'disabled' : ''}>下一页</button>
        </div>
        <div class="episode-frame-list">${frameRows || '<div class="data-preview-empty">当前 episode 暂无帧数据</div>'}</div>
      </div>
    </section>`;
    return;
  }
  const currentImages = (currentEpisode.preview_images || []).slice(0, 4);
  const currentImageGrid = currentImages.map(image => `<article class="camera-card data-current-card">
    <div class="camera-title"><strong>${escapeHtml(image.name)}.jpg</strong><small>${formatBytes(image.size_bytes)}</small></div>
    <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name)}" loading="eager">
    <div class="camera-meta">${escapeHtml(currentEpisode.name)} 路 ${escapeHtml(image.name)}.jpg</div>
  </article>`).join('');
  $('#dataPreviewList').innerHTML = `<section class="episode-preview-card data-current-preview">
    <div class="episode-preview-head">
      <div><h3>当前预览 · ${escapeHtml(currentEpisode.name)}</h3><p>${currentEpisode.error ? escapeHtml(currentEpisode.error) : `idx ${currentEpisode.first_idx ?? '—'} → ${currentEpisode.last_idx ?? '—'}`}</p></div>
      <div class="actions">
        <span class="episode-preview-badge ${currentEpisode.completed ? 'ok' : ''}">${currentEpisode.completed ? '已完成' : '采集中/保存中'}</span>
        <button class="action red" data-action="delete-episode" data-id="${escapeHtml(task.id)}" data-episode="${escapeHtml(currentEpisode.name)}">删除当前条</button>
      </div>
    </div>
    <div class="episode-preview-body">
      ${singleEpisode ? '' : `<div class="data-preview-nav">
        <button class="plain" data-action="preview-prev" ${index >= episodes.length - 1 ? 'disabled' : ''}>上一条</button>
        <strong>${escapeHtml(currentEpisode.name)} · ${escapeHtml(total - index)} / ${escapeHtml(total)}</strong>
        <button class="plain" data-action="preview-next" ${index <= 0 ? 'disabled' : ''}>下一条</button>
      </div>`}
      ${currentImageGrid ? `<div class="camera-grid">${currentImageGrid}</div>` : '<div class="data-preview-empty">当前轮数暂无可预览图片</div>'}
    </div>
  </section>`;
}
function formatPostprocessLogs(rawLogs) {
  const logs = rawLogs.slice(-8);
  if (!logs.length) return '';
  const hasTorchcodecFallback = logs.some(line => line.includes("'torchcodec' is not available") && line.includes("falling back to 'pyav'"));
  const rendered = [...logs];
  if (hasTorchcodecFallback) {
    rendered.push('提示：torchcodec 未安装时 LeRobot 会自动回退到 pyav 解码；只要任务最终 exit_code=0，这不是转换失败。');
  }
  return rendered.join('\n');
}
function renderPostprocessLogBlock(detail) {
  if (!String(detail || '').trim()) return '';
  return `<div class="log-copy-wrap"><pre>${escapeHtml(detail)}</pre><button class="copy-icon-btn" title="复制" aria-label="复制" data-action="copy-text" data-copy="${escapeHtml(detail)}"><span></span></button></div>`;
}
function isCancelledPostprocess(record) {
  return Number(record?.exit_code) === -15 || Number(record?.exit_code) === -9 || record?.status === 'cancelled';
}
function postprocessRecordStatusText(record) {
  if (record?.running || record?.status === 'running') return '运行中';
  if (String(record?.progress?.stage || '').includes('跳过')) return '已跳过';
  if (record?.status === 'completed' || record?.exit_code === 0) return '已完成';
  if (isCancelledPostprocess(record)) return '已取消';
  return `失败${record?.exit_code === null || record?.exit_code === undefined ? '' : ` (${record.exit_code})`}`;
}
function cleanProgressLine(line) {
  return String(line || '').replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '').replace(/\r/g, '').trim();
}
function normalizePostprocessProgress(progress) {
  if (!progress || typeof progress !== 'object') return null;
  const percent = Number(progress.percent);
  const currentNumber = progress.current === null || progress.current === undefined ? null : Number(progress.current);
  const totalNumber = progress.total === null || progress.total === undefined ? null : Number(progress.total);
  const current = currentNumber !== null && Number.isFinite(currentNumber) ? currentNumber : (progress.current ?? null);
  const total = totalNumber !== null && Number.isFinite(totalNumber) ? totalNumber : (progress.total ?? null);
  return {
    percent: Number.isFinite(percent) ? Math.min(100, Math.max(0, percent)) : null,
    current,
    total,
    stage: friendlyPostprocessStage(progress.stage, progress.line),
    speed: progress.speed || '',
    line: progress.line || '',
    indeterminate: false,
  };
}
function friendlyPostprocessStage(stage, line = '') {
  const text = String(stage || '').trim();
  const rawLine = String(line || '');
  if (/^Downloading data$/i.test(text)) return '读取当前批次数据';
  if (/^Generating train split$/i.test(text)) return '生成训练索引';
  if (/^Creating parquet from Arrow format$/i.test(text)) return '写入 parquet 数据';
  if (/^Map$/i.test(text)) return '处理当前批次数据';
  if (/^OSS 上传/i.test(text)) return text;
  if (/^模型下载/i.test(text)) return text;
  if (/Downloading data:/i.test(rawLine)) return '读取当前批次数据';
  if (/Generating train split:/i.test(rawLine)) return '生成训练索引';
  if (/Creating parquet from Arrow format:/i.test(rawLine)) return '写入 parquet 数据';
  return text || '处理中';
}
function parsePostprocessProgress(rawLogs, running = false, exitCode = null, progressObject = null) {
  const direct = normalizePostprocessProgress(progressObject);
  if (direct) {
    if (exitCode === 0) direct.percent = 100;
    direct.indeterminate = running && (direct.percent === null || Number.isNaN(direct.percent));
    return direct;
  }
  const logs = (rawLogs || []).map(cleanProgressLine).filter(Boolean);
  let latest = null;
  for (const line of logs) {
    const percentMatch = line.match(/(\d{1,3})%\|/);
    const ratioMatch = line.match(/\|\s*(\d+)\s*\/\s*(\d+)\s*\[/);
    if (!percentMatch && !ratioMatch) continue;
    const percent = percentMatch ? Math.min(100, Math.max(0, Number(percentMatch[1]))) : null;
    const current = ratioMatch ? Number(ratioMatch[1]) : null;
    const total = ratioMatch ? Number(ratioMatch[2]) : null;
    const stageMatch = line.match(/^([^:]{2,48}):\s*/);
    const speedMatch = line.match(/,\s*([0-9.]+\s*[^,\]]+\/s)\]/);
    latest = {
      percent: percent ?? (total ? Math.round((current / total) * 100) : null),
      current,
      total,
      stage: friendlyPostprocessStage(stageMatch ? stageMatch[1].trim() : '处理中', line),
      speed: speedMatch ? speedMatch[1].trim() : '',
      line,
    };
  }
  if (!latest) {
    return {
      percent: exitCode === 0 ? 100 : null,
      current: null,
      total: null,
      stage: running ? '任务运行中' : (exitCode === 0 ? '已完成' : '等待日志'),
      speed: '',
      line: logs.at(-1) || '',
      indeterminate: running,
    };
  }
  if (exitCode === 0) latest.percent = 100;
  latest.indeterminate = running && (latest.percent === null || Number.isNaN(latest.percent));
  return latest;
}
function renderPostprocessProgress(rawLogs, {running = false, exitCode = null, compact = false, progress = null} = {}) {
  const parsed = parsePostprocessProgress(rawLogs, running, exitCode, progress);
  const hasPercent = typeof parsed.percent === 'number' && Number.isFinite(parsed.percent);
  const percent = hasPercent ? Math.min(100, Math.max(0, parsed.percent)) : 100;
  const ratio = parsed.current !== null && parsed.total !== null ? `${parsed.current}/${parsed.total}` : (running ? '运行中' : '—');
  const speed = parsed.speed || (running ? '读取进度中' : '—');
  const label = hasPercent ? `${percent}%` : (running ? '进行中' : '—');
  return `<div class="postprocess-progress ${compact ? 'compact' : ''} ${parsed.indeterminate ? 'indeterminate' : ''}">
    <div class="postprocess-progress-head"><strong>${escapeHtml(parsed.stage)}</strong><span>${escapeHtml(label)}</span></div>
    <div class="postprocess-progress-bar"><i style="width:${percent}%"></i></div>
    ${compact ? '' : `<div class="postprocess-progress-meta"><span>${escapeHtml(ratio)}</span><span>${escapeHtml(speed)}</span></div>`}
  </div>`;
}
function renderEpisodeProgressInline(status) {
  const total = Number(status?.lerobot_expected_episodes || 0);
  if (!total) return '';
  const done = Math.min(total, Math.max(0, Number(status?.lerobot_episodes || 0)));
  const percent = Math.round((done / total) * 100);
  const stateText = status.convert_running ? '转换中' : (status.lerobot_ready ? '已完成' : convertStatusText(status || {}));
  return `<div class="episode-inline-progress">
    <div class="episode-inline-progress-head"><span>转换 episode 进度</span><strong>${escapeHtml(String(percent))}%</strong></div>
    <div class="episode-inline-progress-bar"><i style="width:${percent}%"></i></div>
    <div class="episode-inline-progress-foot"><span>${escapeHtml(stateText)}</span><span>${escapeHtml(`${done}/${total}`)}</span></div>
  </div>`;
}
function renderPostprocessJobs() {
  const list = $('#postprocessJobList');
  const hint = $('#postprocessJobHint');
  if (!list || !hint) return;
  const datasetJobKinds = new Set(['convert', 'normalize', 'package', 'package_assets']);
  const activeTask = activePostprocessTaskId ? appState.tasks.find(t => t.id === activePostprocessTaskId) : null;
  const activeRepoId = activeTask?.postprocess_status?.repo_id || (activeTask ? `local/${activeTask.name}` : '');
  const activeRobotDir = activeTask ? `${appState.dataset_root}/${activeTask.name}` : '';
  let jobs = (appState.postprocess_jobs || []).filter(job => {
    if (!datasetJobKinds.has(job.kind)) return false;
    if (!activeTask) return true;
    return job.task_id === activeTask.id
      || job.task_name === activeTask.name
      || job.repo_id === activeRepoId
      || String(job.command || '').includes(activeRobotDir);
  });
  const activeStatus = activeTask?.postprocess_status || {};
  if (activeTask && !jobs.length && (activeStatus.convert_running || activeStatus.normalize_running || activeStatus.package_running)) {
    const pendingKind = activeStatus.convert_running ? 'convert' : (activeStatus.normalize_running ? 'normalize' : 'package');
    jobs = [{
      id: `pending-${activeTask.id}`,
      kind: pendingKind,
      task_id: activeTask.id,
      task_name: activeTask.name,
      repo_id: activeRepoId,
      pid: '—',
      running: true,
      started_at: activeStatus.convert_running ? activeStatus.convert_started_at : (activeStatus.normalize_running ? activeStatus.normalize_started_at : activeStatus.package_started_at),
      command: '任务已启动，正在同步处理记录...',
      logs: [],
    }];
  }
  if (activeTask) {
    const latestByKind = new Map();
    for (const job of jobs) {
      const key = job.kind || 'unknown';
      if (!latestByKind.has(key)) latestByKind.set(key, job);
    }
    jobs = Array.from(latestByKind.values());
  }
  const running = jobs.filter(job => job.running).length;
  const finished = jobs.length - running;
  hint.textContent = jobs.length ? `${running} 个运行中${finished ? `，${finished} 个已结束` : ''}` : '暂无运行任务';
  if (!jobs.length) {
    list.innerHTML = '<div class="data-preview-empty">暂无转换或归一化任务</div>';
    return;
  }
  list.innerHTML = jobs.slice(0, 5).map(job => {
    const status = postprocessRecordStatusText(job).replace('已完成', '完成');
    const logs = formatPostprocessLogs(job.logs || []);
    const ok = job.running || job.exit_code === 0 || isCancelledPostprocess(job);
    const jobTitle = job.kind === 'convert'
      ? 'LeRobot 转换'
      : (job.kind === 'package' || job.kind === 'package_assets'
        ? (job.kind === 'package_assets' ? '归一化压缩' : '数据压缩')
        : '归一化统计');
    return `<article class="postprocess-job">
      <div class="episode-preview-head">
        <div><h3>${jobTitle} · ${escapeHtml(job.task_name)}</h3><p>PID ${escapeHtml(job.pid || '—')} · ${escapeHtml(formatTime(job.started_at))}</p></div>
        <span class="episode-preview-badge ${ok ? 'ok' : ''}">${escapeHtml(status)}</span>
      </div>
      ${job.error ? `<div class="postprocess-error">失败原因：${escapeHtml(job.error)}</div>` : ''}
      <div class="command"><span>命令</span><code>${escapeHtml(job.command)}</code><button class="copy-icon-btn" title="复制" aria-label="复制" data-action="copy-text" data-copy="${escapeHtml(job.command)}"><span></span></button></div>
      ${renderPostprocessLogBlock(logs)}
    </article>`;
  }).join('');
}
function renderRuntime() {
  const process = appState.process || {}, teleop = appState.teleop || {};
  if (runtimePending && runtimePendingResolved(process, teleop)) {
    showNotice(`${runtimePending.message.replace('...', '')}已生效`, 'success');
    clearRuntimePending();
  } else if (runtimePending && runtimePendingTimedOut()) {
    showNotice(`${runtimePending.message} 暂未生效，请查看运行日志或设备状态`);
    clearRuntimePending();
  }
  ensureRuntimeCopyButtons();
  const current = process.task;
  $('#currentRuntime').textContent = process.running && current ? `运行中：${current.name} · PID ${process.pid}` : '当前无运行任务';
  if (!current) {
    const viewingTask = activeTaskId ? appState.tasks.find(t => t.id === activeTaskId) : null;
    if ($('#controlModal')?.classList.contains('open') && viewingTask) {
      renderTaskDetails(viewingTask);
    } else {
      renderTaskDetails(null);
    }
    $('#recordControl').hidden = false;
    $('#handEyeControl').hidden = true;
    renderCameraPreview();
    return;
  }
  activeTaskId = current.id;
  const latest = appState.tasks.find(t => t.id === current.id) || current;
  renderTaskDetails(latest);
  setControlCompactMode(true);
  $('#controlTitle').textContent = `数采控制 · ${current.name}`;
  $('#controlSubtitle').textContent = `${current.device_name} · ${current.description}`;
  const handEyeEnabled = Boolean(teleop.HAND_EYE_ENABLED);
  const handEyeState = teleop.HAND_EYE_STATE || 'FOLLOW';
  const replayEnabled = Boolean(teleop.TRAJECTORY_REPLAY_ENABLED);
  const replayState = teleop.TRAJECTORY_REPLAY_STATE || 'WAITING';
  const handEyeStatus = {
    FOLLOW: '手眼标定：遥操运行中',
    SETTLING: '手眼标定：双臂锁定，判稳中',
    CAPTURING: `手眼标定：采集中 ${teleop.HAND_EYE_CAPTURED_FRAMES || 0} / ${teleop.HAND_EYE_BURST_FRAMES || 5}`,
    SAVING: '手眼标定：数据保存中',
    HOLD: '手眼标定：已保存并保持，等待 Rebase',
  };
  let processStatus = !process.running ? `已退出 (${process.exit_code ?? '—'})` : (teleop.RECORD_RUNNING ? '正在录制' : teleop.START ? '遥操运行中' : teleop.online ? '等待开始遥操' : '初始化中');
  if (handEyeEnabled && teleop.START) processStatus = handEyeStatus[handEyeState] || processStatus;
  const replayStatus = {
    WAITING: '轨迹回放：等待开始',
    PLAYING: `轨迹回放：${teleop.TRAJECTORY_REPLAY_INDEX || 0} / ${teleop.TRAJECTORY_REPLAY_FRAMES || 0}`,
    EVENT_HOLD: '轨迹回放：关键帧判稳与拍摄中',
    COMPLETED: '轨迹回放完成，保持最终姿态',
    ERROR: `轨迹回放失败：${teleop.TRAJECTORY_REPLAY_ERROR || '未知错误'}`,
  };
  if (replayEnabled && teleop.START) processStatus = replayStatus[replayState] || processStatus;
  if (teleop.HAND_EYE_ERROR) processStatus += `；${teleop.HAND_EYE_ERROR}`;
  $('#processStatus').textContent = runtimePending?.message || processStatus;
  $('#episodeProgress').textContent = `${latest.existing_episodes ?? 0} / ${latest.target_episodes ?? 0}`;
  $('#pidText').textContent = process.pid || '—'; $('#commandText').textContent = process.command || '—';
  if ($('#controlModal')?.classList.contains('open')) {
    $('#logOutput').textContent = appState.logs?.length ? appState.logs.join('\n') : '暂无日志';
  }
  renderCameraPreview();
  const ready = process.running && teleop.online;
  $('#startControl').disabled = Boolean(runtimePending) || !ready || Boolean(teleop.START);
  $('#startControl').textContent = runtimePending?.type === 'teleop_start' ? '开始指令已发送...' : (replayEnabled ? '开始轨迹回放' : '开始遥操');
  $('#recordControl').hidden = handEyeEnabled;
  $('#handEyeControl').hidden = !handEyeEnabled || replayEnabled;
  const canStartRecording = ready && teleop.START && teleop.READY;
  const canStopRecording = process.running && Boolean(teleop.RECORD_RUNNING);
  $('#recordControl').disabled = Boolean(runtimePending) || !(canStartRecording || canStopRecording);
  const completedEpisodes = Number(latest.existing_episodes) || 0;
  const lastEpisode = episodeNumber(latest.last_episode);
  const nextEpisode = episodeNumber(latest.next_episode) || completedEpisodes + 1;
  const recordingEpisode = Math.max(lastEpisode, completedEpisodes + 1);
  const savingEpisode = lastEpisode > completedEpisodes ? lastEpisode : completedEpisodes + 1;
  if (runtimePending?.type === 'record_start') {
    $('#recordControl').textContent = '录制指令已发送...';
  } else if (runtimePending?.type === 'record_stop') {
    $('#recordControl').textContent = '结束录制并保存中...';
  } else if (teleop.RECORD_RUNNING) {
    $('#recordControl').textContent = `结束并保存第 ${recordingEpisode} 条`;
  } else if (process.running && teleop.START && !teleop.READY && lastEpisode > completedEpisodes) {
    $('#recordControl').textContent = `正在保存第 ${savingEpisode} 条...`;
  } else {
    $('#recordControl').textContent = `开始录制第 ${nextEpisode} 条`;
  }
  const canToggleHandEye = ready && teleop.START && ['FOLLOW', 'HOLD'].includes(handEyeState);
  $('#handEyeControl').disabled = !handEyeEnabled || !canToggleHandEye;
  if (handEyeState === 'HOLD') {
    $('#handEyeControl').textContent = 'Rebase 并恢复遥操';
  } else if (handEyeState === 'SETTLING') {
    $('#handEyeControl').textContent = '双臂判稳中...';
  } else if (handEyeState === 'CAPTURING') {
    $('#handEyeControl').textContent = `采集中 ${teleop.HAND_EYE_CAPTURED_FRAMES || 0} / ${teleop.HAND_EYE_BURST_FRAMES || 5}`;
  } else if (handEyeState === 'SAVING') {
    $('#handEyeControl').textContent = '数据保存中...';
  } else {
    $('#handEyeControl').textContent = '锁定并采样';
  }
  $('#stopControl').disabled = Boolean(runtimePending) || !ready;
  $('#stopControl').textContent = runtimePending?.type === 'process_stop' ? '结束采集指令已发送...' : '结束遥操';
}
function applyState(state) {
  appState = state;
  const taskFileHint = $('#taskFileHint');
  if (taskFileHint) taskFileHint.textContent = state.task_file || 'data/datasets/robot/tasks.json';
  renderActiveView();
  renderRuntime();
  if ($('#postprocessModal')?.classList.contains('open')) renderPostprocessJobs();
  refreshOpenPostprocessModal();
}
function renderActiveView() {
  if (activeView === 'devices') {
    renderDevice();
  } else if (activeView === 'tasks') {
    renderTasks();
  } else if (activeView === 'dataList') {
    renderDataList();
  } else if (activeView === 'processing') {
    renderProcessingProfiles();
  } else if (activeView === 'training') {
    renderTraining();
  } else if (activeView === 'oss') {
    renderOssView();
  } else if (activeView === 'trainingTemplate' || activeView === 'trainingDoc') {
    renderTrainingTemplateView();
  } else if (activeView === 'trainingCommand') {
    renderTrainingCommandView();
  } else {
    renderTasks();
  }
}
function isLogAutoRefreshNeeded() {
  return Boolean($('#controlModal')?.classList.contains('open') || $('#postprocessModal')?.classList.contains('open'));
}
async function refresh(silent = true, options = {}) {
  if (silent && document.activeElement?.closest('.template-editor-row,.manual-doc-form,.manual-advanced-panel')) return;
  if (silent && activeView === 'oss' && document.activeElement?.closest('.oss-local-toolbar,.oss-root-line,.oss-transfer-grid,.oss-package-row')) return;
  if (refreshInFlight || (silent && Date.now() < userInteractingUntil)) return;
  refreshInFlight = true;
  try {
    const nextState = await api('/api/state');
    if ((activeView === 'oss' || activeView === 'trainingTemplate' || activeView === 'trainingCommand') && ossTransferState.localDir) {
      nextState.oss_transfer = await api('/api/oss/local-packages', {directory: ossTransferState.localDir});
    }
    applyState(nextState);
    if (activeView === 'dataList') loadDataList({silent:true});
  } catch (error) {
    if (!silent) showNotice(error.message);
  } finally {
    refreshInFlight = false;
  }
}

function setActiveView(view) {
  const target = $(`#${view}View`) ? view : 'devices';
  activeView = target;
  localStorage.setItem('teleop.activeView', target);
  $$('[data-view]').forEach(n => n.classList.toggle('active', n.dataset.view === target));
  $$('.view').forEach(n => n.classList.remove('active'));
  $(`#${target}View`).classList.add('active');
  renderActiveView();
  if (target === 'dataList') loadDataList({silent:true});
}
function applySidebarState() {
  document.body.classList.toggle('sidebar-collapsed', sidebarCollapsed);
  const button = $('#sidebarToggle');
  if (!button) return;
  const label = sidebarCollapsed ? '展开菜单' : '收起菜单';
  button.setAttribute('aria-label', label);
  button.title = label;
}
function toggleSidebar() {
  sidebarCollapsed = !sidebarCollapsed;
  localStorage.setItem('teleop.sidebarCollapsed', sidebarCollapsed ? '1' : '0');
  applySidebarState();
}
function setupTrainingSetCreator() {
  const form = $('#trainingSetForm');
  const refreshButton = $('#refreshTraining');
  if (!form || !refreshButton || $('#trainingSetModal')) return;
  const titleBlock = $('#trainingView .page-title h1')?.parentElement;
  if (titleBlock) {
    titleBlock.querySelector('h1').textContent = '训练数据集';
    const subtitle = titleBlock.querySelector('p');
    if (subtitle) subtitle.textContent = '创建训练数据集，维护任务成员，并生成训练环境交接包';
  }
  const profilePanel = $('#trainingProfiles')?.closest('.training-panel');
  profilePanel?.classList.add('training-create-inline-hidden');
  const setPanel = $('#trainingSets')?.closest('.training-panel');
  setPanel?.querySelector('h2') && (setPanel.querySelector('h2').textContent = '已创建训练数据集');
  const inlinePanel = form.closest('.training-panel');
  inlinePanel?.classList.add('training-create-inline-hidden');
  const button = document.createElement('button');
  button.id = 'createTrainingSetButton';
  button.className = 'primary';
  button.type = 'button';
  button.textContent = '创建训练集';
  const actions = document.createElement('div');
  actions.className = 'page-actions';
  refreshButton.className = 'plain';
  refreshButton.textContent = '刷新';
  refreshButton.parentNode.insertBefore(actions, refreshButton);
  actions.appendChild(refreshButton);
  actions.appendChild(button);

  const modal = document.createElement('div');
  modal.id = 'trainingSetModal';
  modal.className = 'modal';
  modal.setAttribute('aria-hidden', 'true');
  modal.innerHTML = `<div class="dialog training-set-dialog">
    <div class="dialog-head"><div><h2>创建训练集</h2><p>先创建训练集容器，再从数据集画像中加入任务</p></div><button class="close" data-close="trainingSetModal">×</button></div>
  </div>`;
  modal.querySelector('.dialog').appendChild(form);
  document.body.appendChild(modal);
  button.addEventListener('click', () => openTrainingSetModal());
  modal.addEventListener('click', event => { if (event.target === modal) closeModal('trainingSetModal'); });
  modal.querySelector('[data-close]')?.addEventListener('click', () => closeModal('trainingSetModal'));
}
function fillTrainingSetForm(item = null) {
  const form = $('#trainingSetForm');
  if (!form) return;
  form.reset();
  const values = item || {
    name: '',
    config_name: '',
    action_dim: 32,
    real_action_dim: 14,
    action_horizon: 16,
    h200_dataset_root: '/home/ubuntu/datasets/lerobot',
    h200_assets_root: '/home/ubuntu/assets',
    description: '',
  };
  ['name', 'config_name', 'action_dim', 'real_action_dim', 'action_horizon', 'h200_dataset_root', 'h200_assets_root', 'description'].forEach(name => {
    if (form.elements[name]) form.elements[name].value = values[name] ?? '';
  });
}
function openTrainingSetModal(item = null) {
  editingTrainingSetId = item?.id || null;
  fillTrainingSetForm(item);
  const modal = $('#trainingSetModal');
  const title = modal?.querySelector('.dialog-head h2');
  const subtitle = modal?.querySelector('.dialog-head p');
  const submit = $('#trainingSetForm button[type="submit"]');
  if (title) title.textContent = editingTrainingSetId ? '编辑训练集' : '创建训练集';
  if (subtitle) subtitle.textContent = editingTrainingSetId ? '修改训练集参数；任务成员在数据处理页维护' : '先创建训练集容器，再从数据集画像中加入任务';
  if (submit) submit.textContent = editingTrainingSetId ? '保存训练集' : '创建训练集';
  openModal('trainingSetModal');
}
normalizeStaticLabels();
$$('[data-view]').forEach(button => button.addEventListener('click', () => setActiveView(button.dataset.view)));
setActiveView(activeView);
$('#sidebarToggle')?.addEventListener('click', toggleSidebar);
applySidebarState();
setupTrainingSetCreator();
$$('[data-close]').forEach(button => button.addEventListener('click', () => closeModal(button.dataset.close)));
$$('.modal').forEach(modal => modal.addEventListener('click', event => { if (event.target === modal) closeModal(modal.id); }));
document.addEventListener('pointerdown', event => {
  if (event.target.closest('button,input,select,textarea,a,[data-action]')) holdAutoRefresh();
}, true);

function fillDeviceForm(device) {
  const form = $('#deviceForm'); form.reset();
  const values = device ? {name:device.name,...device.config,xr_view:device.config.xr_view || 'head',arm_reference_mode:device.config.arm_reference_mode || 'head_position',img_server_ip:device.config.img_server_ip || DEFAULT_IMAGE_SERVER_IP,webrtc_server_ip:device.config.webrtc_server_ip || DEFAULT_WEBRTC_SERVER_IP,data_dir:device.config.data_dir || appState.data_dir || DEFAULT_DATA_DIR,init_arm_pose_file:device.config.init_arm_pose_file || '',init_arm_pose_duration:device.config.init_arm_pose_duration || 5,hand_eye_replay_trajectory:device.config.hand_eye_replay_trajectory || '',hand_eye_replay_time_scale:device.config.hand_eye_replay_time_scale || 1,hand_eye_record:Boolean(device.config.hand_eye_record),ik_replay_live_url:device.config.ik_replay_live_url || DEFAULT_IK_REPLAY_LIVE_URL,ik_replay_live_fps:device.config.ik_replay_live_fps || 10} : {arm:'H2',left_ee:'none',right_ee:'inspire_dfx',input_mode:'hand',display_mode:'pass-through',xr_view:'head',arm_reference_mode:'head_position',img_server_ip:DEFAULT_IMAGE_SERVER_IP,webrtc_server_ip:DEFAULT_WEBRTC_SERVER_IP,data_dir:appState.data_dir || DEFAULT_DATA_DIR,network_interface:'enp86s0',frequency:30,init_arm_pose_file:DEFAULT_H2_INIT_ARM_POSE_FILE,init_arm_pose_duration:5,headless:true,motion:true,hand_eye_replay_trajectory:'',hand_eye_replay_time_scale:1,hand_eye_record:false,ik_replay_live_enable:false,ik_replay_live_url:DEFAULT_IK_REPLAY_LIVE_URL,ik_replay_live_fps:10};
  if (values.ee && !values.left_ee && !values.right_ee) {
    values.left_ee = values.ee;
    values.right_ee = values.ee;
  }
  Object.entries(values).forEach(([key,value]) => { const input=form.elements[key]; if (!input) return; input.type==='checkbox' ? input.checked=Boolean(value) : input.value=value ?? ''; });
  syncEndEffector();
}
function syncEndEffector() {
  const form=$('#deviceForm');
  const passive = new Set(['none', 'rubber']);
  const isActive = value => !passive.has(value);
  const selects = {
    left_ee: form.elements.left_ee,
    right_ee: form.elements.right_ee,
  };
  const isAllowedWithOther = (value, otherValue) => {
    if (value === otherValue) return true;
    if (!isActive(value) || !isActive(otherValue)) return true;
    return false;
  };
  const normalizeSelect = (name, otherName) => {
    const select = selects[name];
    const otherValue = selects[otherName].value;
    [...select.options].forEach(option => {
      option.disabled = !isAllowedWithOther(option.value, otherValue);
    });
    if (select.selectedOptions.length && select.selectedOptions[0].disabled) {
      const fallback = [...select.options].find(option => !option.disabled);
      if (fallback) {
        select.value = fallback.value;
        deviceFormDirty = true;
      }
    }
  };
  normalizeSelect('left_ee', 'right_ee');
  normalizeSelect('right_ee', 'left_ee');
  const leftEe = form.elements.left_ee.value;
  const rightEe = form.elements.right_ee.value;
  const hasActiveEndEffector = [leftEe, rightEe].some(isActive);
  if (hasActiveEndEffector && form.elements.input_mode.value === 'controller') {
    form.elements.input_mode.value = 'hand';
    deviceFormDirty = true;
  }
  if (form.elements.input_mode.value === 'hand') {
    form.elements.display_mode.value = 'pass-through';
    form.elements.xr_view.value = 'head';
  }
  if (form.elements.arm.value === 'H2' && form.elements.arm_reference_mode && form.elements.arm_reference_mode.value === 'world') {
    form.elements.arm_reference_mode.value = 'head_position';
  }
  const sameEndEffector = leftEe === rightEe;
  $('#eeHint').textContent=hasActiveEndEffector
    ? '主动末端需要 hand；左右主动末端必须一致，或一侧主动、一侧被动'
    : '不控制/橡胶手不会发送末端控制命令';
}
function showTask(task) {
  activeTaskId=task.id;
  const isCurrent=appState.process?.task?.id===task.id;
  if(isCurrent) renderRuntime();
  else {
    setControlCompactMode(false);
    $('#controlTitle').textContent=`任务详情 · ${task.name}`; $('#controlSubtitle').textContent=`${task.device_name} · ${task.description}`;
    renderTaskDetails(task);
    $('#processStatus').textContent=task.status; $('#episodeProgress').textContent=`${task.existing_episodes} / ${task.target_episodes}`; $('#pidText').textContent='—';
    $('#commandText').textContent='点击“开始采集”后生成启动命令'; $('#logOutput').textContent='当前任务未运行';
    renderCameraPreview();
    ['startControl','recordControl','handEyeControl','stopControl'].forEach(id=>$(`#${id}`).disabled=true);
  }
  openModal('controlModal');
}
const CAMERA_SLOT_TARGETS = {camera_head: 'image', camera_left: 'left_wrist_image', camera_right: 'right_wrist_image'};
function cameraMapToSlots(cameraMap) {
  const slots = {camera_head: 'color_0', camera_left: 'color_1', camera_right: 'color_2'};
  if (!cameraMap || cameraMap === 'auto') return slots;
  Object.keys(slots).forEach(key => { slots[key] = 'none'; });
  String(cameraMap).split(',').forEach(item => {
    const [source, target] = item.split(':');
    const slot = Object.keys(CAMERA_SLOT_TARGETS).find(key => CAMERA_SLOT_TARGETS[key] === target);
    if (slot && /^color_[0-3]$/.test(source)) slots[slot] = source;
  });
  return slots;
}
function cameraSlotsToMap(form) {
  const parts = [];
  Object.entries(CAMERA_SLOT_TARGETS).forEach(([slot, target]) => {
    const source = form.elements[slot]?.value || 'none';
    if (source !== 'none') parts.push(`${source}:${target}`);
  });
  return parts.join(',');
}
function cameraSlotHintText(form) {
  const labels = [
    ['头部', form.elements.camera_head?.value, 'image'],
    ['左手', form.elements.camera_left?.value, 'left_wrist_image'],
    ['右手', form.elements.camera_right?.value, 'right_wrist_image'],
  ];
  const text = labels.map(([name, source, target]) => `${name}(${target})=${source === 'none' ? 'none 黑图' : source}`).join('，');
  return `${text}；提交映射：${cameraSlotsToMap(form) || '全部 none，不允许转换'}`;
}
function updateCameraMapHint() {
  const hint = $('#cameraMapHint');
  const form = $('#postprocessForm');
  if (!hint || !form) return;
  hint.textContent = cameraSlotHintText(form);
}
function imageShapeText(shape) {
  return Array.isArray(shape) && shape.length >= 2 ? `${shape[0]}x${shape[1]}` : '未知';
}
function updateImageSizeHint(task) {
  const hint = $('#imageSizeHint');
  const form = $('#postprocessForm');
  if (!hint || !form?.elements.image_size) return;
  const shape = task?.postprocess_status?.source_image_shape || [];
  const originalText = imageShapeText(shape);
  const smallOption = Array.from(form.elements.image_size.options).find(option => option.value === '240x320');
  const canUse240 = !(Array.isArray(shape) && shape.length >= 2 && (Number(shape[0]) < 240 || Number(shape[1]) < 320));
  if (smallOption) {
    smallOption.disabled = !canUse240;
    smallOption.title = canUse240 ? '' : `原始尺寸 ${originalText} 小于 240x320，不能放大`;
  }
  if (!canUse240 && form.elements.image_size.value === '240x320') {
    form.elements.image_size.value = 'original';
  }
  const selected = form.elements.image_size.value === '240x320' ? '输出 240x320' : '保持原始尺寸';
  hint.textContent = `原始尺寸：${originalText}；${selected}`;
}
function updateImageEncodingHint() {
  const form = $('#postprocessForm');
  const hint = $('#imageEncodingHint');
  if (!form?.elements.image_encoding || !form?.elements.jpeg_quality) return;
  const encoding = form.elements.image_encoding.value || 'auto';
  form.elements.jpeg_quality.disabled = encoding === 'png' || encoding === 'video';
  if (form.elements.video_backend) form.elements.video_backend.disabled = encoding !== 'video';
  if (encoding === 'video') {
    if (form.elements.image_writer_processes && !form.elements.image_writer_processes.value) {
      form.elements.image_writer_processes.value = '2';
    }
    if (form.elements.image_writer_threads && !form.elements.image_writer_threads.value) {
      form.elements.image_writer_threads.value = '4';
    }
  }
  if (hint) {
    hint.textContent = encoding === 'auto'
      ? '自动：原始 JPG 保存 JPG，原始 PNG 保存 PNG'
      : (encoding === 'jpg'
        ? '强制保存为 JPG/JPEG，使用 JPEG 质量参数'
        : (encoding === 'video' ? '保存为 LeRobot Video MP4；会启用 writer 并发，仍建议先小批量验证' : '强制保存为 PNG，JPEG 质量不生效'));
  }
}
function showPostprocess(task) {
  mountPackageButtonInPostprocessActions();
  activePostprocessTaskId = task.id;
  const defaults = appState.postprocess_defaults || {};
  const status = task.postprocess_status || {};
  const form = $('#postprocessForm');
  form.reset();
  form.elements.task_id.value = task.id;
  form.elements.repo_id.value = status.repo_id || `local/${task.name}`;
  form.elements.repo_id.readOnly = true;
  form.elements.robot_type.value = status.robot_type || defaults.robot_type || 'robot';
  const cameraSlots = cameraMapToSlots(status.camera_map || defaults.camera_map || 'auto');
  Object.entries(cameraSlots).forEach(([key, value]) => {
    if (form.elements[key]) form.elements[key].value = value;
  });
  updateCameraMapHint();
  form.elements.image_size.value = status.image_size || defaults.image_size || 'original';
  form.elements.image_encoding.value = status.image_encoding || defaults.image_encoding || 'auto';
  form.elements.jpeg_quality.value = status.jpeg_quality || defaults.jpeg_quality || 95;
  if (form.elements.video_backend) form.elements.video_backend.value = status.video_backend || defaults.video_backend || '';
  if (form.elements.image_writer_processes) form.elements.image_writer_processes.value = status.image_writer_processes ?? defaults.image_writer_processes ?? '';
  if (form.elements.image_writer_threads) form.elements.image_writer_threads.value = status.image_writer_threads ?? defaults.image_writer_threads ?? '';
  updateImageSizeHint(task);
  updateImageEncodingHint();
  form.elements.config_name.value = status.config_name || defaults.openpi_config_name || `${defaults.config_prefix || 'pi05'}_${task.name}`;
  form.elements.openpi_dir.value = status.openpi_dir || defaults.openpi_dir || '~/openpi';
  form.elements.max_frames.value = status.max_frames || '';
  form.elements.start_episode.value = status.start_episode || '';
  form.elements.batch_size.value = status.batch_size || '';
  form.elements.resume.checked = Boolean(status.resume);
  form.elements.overwrite.checked = status.overwrite !== false;
  form.elements.overwrite.disabled = form.elements.resume.checked;
  const assetsDir = defaults.openpi_assets_dir || `${appState.dataset_root}/../openpi/assets`;
  const lerobotDir = status.lerobot_dir || `${defaults.lerobot_home || `${appState.dataset_root}/../lerobot`}/local/${task.name}`;
  const normStatsDir = status.norm_stats_dir || `${assetsDir}/${form.elements.config_name.value}/local/${task.name}`;
  $('#postprocessTitle').textContent = `转换/归一化 · ${task.name}`;
  updatePostprocessModalStatus(task, {lerobotDir, normStatsDir});
  renderPostprocessJobs();
  openModal('postprocessModal');
}
function updatePostprocessModalStatus(task, paths = {}) {
  if (!task) return;
  updateCameraMapHint();
  updateImageSizeHint(task);
  const status = task.postprocess_status || {};
  const form = $('#postprocessForm');
  const defaults = appState.postprocess_defaults || {};
  const assetsDir = defaults.openpi_assets_dir || `${appState.dataset_root}/../openpi/assets`;
  const lerobotDir = paths.lerobotDir || status.lerobot_dir || `${defaults.lerobot_home || `${appState.dataset_root}/../lerobot`}/${form.elements.repo_id.value || `local/${task.name}`}`;
  const normStatsDir = paths.normStatsDir || status.norm_stats_dir || `${assetsDir}/${form.elements.config_name.value || defaults.openpi_config_name || `pi05_${task.name}`}/${form.elements.repo_id.value || `local/${task.name}`}`;
  const lerobotEpisodes = status.lerobot_expected_episodes
    ? `，episode：${status.lerobot_episodes || 0}/${status.lerobot_expected_episodes}`
    : '';
  const subtitleLines = [
    `数据目录：${escapeHtml(`${appState.dataset_root}/${task.name}`)}`,
    `相机映射：${escapeHtml(cameraSlotsToMap(form) || '全部 none')}（更新：${escapeHtml(formatTime(status.camera_map_updated_at))}）`,
    `转换选项：${form.elements.resume?.checked ? '断点续转，回滚最近 2 条' : '覆盖/从起始位置转换'}；image-size：${escapeHtml(form.elements.image_size?.value || 'original')}；image-encoding：${escapeHtml(form.elements.image_encoding?.value || 'auto')}；jpeg-quality：${escapeHtml(form.elements.jpeg_quality?.value || '95')}；writer：${escapeHtml(form.elements.image_writer_processes?.value || '默认')}/${escapeHtml(form.elements.image_writer_threads?.value || '默认')}；起始 episode：${escapeHtml(form.elements.start_episode?.value || '自动')}；batch-size：${escapeHtml(form.elements.batch_size?.value || '单批全量')}；填写后会自动逐批转完`,
    `LeRobot：${escapeHtml(lerobotDir)}（${convertStatusText(status)}${escapeHtml(lerobotEpisodes)}，开始：${escapeHtml(formatTime(status.convert_started_at))}，完成：${escapeHtml(formatTime(status.converted_at))}，失败：${escapeHtml(formatTime(status.convert_failed_at))}）`,
    `归一化：${escapeHtml(normStatsDir)}（${normalizeStatusText(status)}，开始：${escapeHtml(formatTime(status.normalize_started_at))}，完成：${escapeHtml(formatTime(status.normalized_at))}，失败：${escapeHtml(formatTime(status.normalize_failed_at))}）`,
    `数据包：${escapeHtml(status.last_data_package || '—')}（${dataPackageStatusText(status)}，开始：${escapeHtml(formatTime(status.data_package_started_at))}，完成：${escapeHtml(formatTime(status.data_packaged_at))}，失败：${escapeHtml(formatTime(status.data_package_failed_at))}）`,
    `归一化包：${escapeHtml(status.last_assets_package || '—')}（${assetsPackageStatusText(status)}，开始：${escapeHtml(formatTime(status.assets_package_started_at))}，完成：${escapeHtml(formatTime(status.assets_packaged_at))}，失败：${escapeHtml(formatTime(status.assets_package_failed_at))}）`,
  ];
  $('#postprocessSubtitle').innerHTML = subtitleLines.join('<br>');
  const convertButton = $('#postprocessForm button[type="submit"]');
  if (convertButton) {
    const sameTaskBusy = Boolean(status.convert_running || status.normalize_running || status.package_running);
    convertButton.disabled = Boolean(status.postprocess_busy && !sameTaskBusy);
    convertButton.title = status.postprocess_busy && !sameTaskBusy ? '其他转换或归一化任务正在运行' : sameTaskBusy ? '重新转换会停止当前任务并启动新的转换' : '';
  }
  const sameTaskBusy = Boolean(status.convert_running || status.normalize_running || status.package_running);
  $('#normalizeButton').disabled = !status.lerobot_ready || Boolean(status.postprocess_busy && !sameTaskBusy);
  $('#normalizeButton').title = status.postprocess_busy && !sameTaskBusy ? '其他转换或归一化任务正在运行' : sameTaskBusy ? '重新计算会停止当前任务并启动新的归一化' : status.lerobot_ready ? '' : '请先转换为 LeRobot';
  const packageButton = $('#packageButton');
  if (packageButton) {
    packageButton.disabled = !status.lerobot_ready || Boolean(status.postprocess_busy && !sameTaskBusy);
    packageButton.title = status.postprocess_busy && !sameTaskBusy ? '其他转换或归一化任务正在运行' : sameTaskBusy ? '请等待当前处理任务完成' : status.lerobot_ready ? '打包压缩 LeRobot 数据集为 tar.gz' : '请先转换为 LeRobot';
  }
  const packageAssetsButton = $('#packageAssetsButton');
  if (packageAssetsButton) {
    packageAssetsButton.disabled = !status.norm_stats_ready || Boolean(status.postprocess_busy && !sameTaskBusy);
    packageAssetsButton.title = status.postprocess_busy && !sameTaskBusy ? '其他转换或归一化任务正在运行' : sameTaskBusy ? '请等待当前处理任务完成' : status.norm_stats_ready ? '打包压缩 OpenPI 归一化值为 tar.gz' : '请先计算归一化';
  }
}
function refreshOpenPostprocessModal() {
  if (!activePostprocessTaskId || !$('#postprocessModal')?.classList.contains('open')) return;
  const task = appState.tasks.find(t => t.id === activePostprocessTaskId);
  updatePostprocessModalStatus(task);
}
$('#addTaskButton').addEventListener('click', () => { if (!appState.device) return showNotice('请先保存当前设备信息'); $('#taskForm').reset(); $('#taskDeviceName').textContent=`${appState.device.name} · ${appState.device.config.arm}`; openModal('taskModal'); });
$('#refreshTasks').addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '刷新中...';
  try {
    await refresh(false);
    showNotice('任务列表已刷新', 'success');
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});
$('#refreshTraining')?.addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '刷新中...';
  try {
    await refresh(false);
    showNotice('训练画像已刷新', 'success');
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});
$('#refreshOssTransfer')?.addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '刷新中...';
  try {
    await refresh(false);
    showNotice('OSS传输信息已刷新', 'success');
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});
$('#refreshProcessing')?.addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '刷新中...';
  try {
    await refresh(false);
    showNotice('数据处理画像已刷新', 'success');
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});
$('#refreshDataList')?.addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '刷新中...';
  try {
    await refresh(false);
    await loadDataList({silent:false});
    showNotice('数据列表已刷新', 'success');
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});
$('#dataListTaskSelect')?.addEventListener('change', event => {
  dataListState.taskId = event.currentTarget.value;
  dataListState.page = 1;
  loadDataList({silent:false});
});
$('#dataListPageSize')?.addEventListener('change', event => {
  dataListState.pageSize = Number(event.currentTarget.value || 50);
  dataListState.page = 1;
  loadDataList({silent:false});
});
$('#prevDataListPage')?.addEventListener('click', () => {
  dataListState.page = Math.max(1, (dataListState.page || 1) - 1);
  loadDataList({silent:false});
});
$('#nextDataListPage')?.addEventListener('click', () => {
  const totalPages = Math.max(1, Math.ceil((dataListState.total || 0) / (dataListState.pageSize || 50)));
  dataListState.page = Math.min(totalPages, (dataListState.page || 1) + 1);
  loadDataList({silent:false});
});
function jumpDataListPage(input) {
  const totalPages = Math.max(1, Math.ceil((dataListState.total || 0) / (dataListState.pageSize || 50)));
  const nextPage = clampPage(input?.value, totalPages);
  if (input) input.value = String(nextPage);
  if (nextPage === dataListState.page) return;
  dataListState.page = nextPage;
  loadDataList({silent:false});
}
$('#dataListPageJump')?.addEventListener('keydown', event => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  clearTimeout(pageJumpTimers.dataList);
  jumpDataListPage(event.currentTarget);
});
$('#dataListPageJump')?.addEventListener('input', event => {
  debouncePageJump('dataList', () => jumpDataListPage(event.currentTarget));
});
$('#dataListPageJump')?.addEventListener('change', event => jumpDataListPage(event.currentTarget));
$('#dataListPageJump')?.addEventListener('blur', event => jumpDataListPage(event.currentTarget));
document.addEventListener('change', event => {
  if (event.target?.id === 'ossTaskNameInput') {
    const taskName = cleanTaskName(event.target.value);
    ossTransferState.taskName = taskName;
    ossTransferState.localDir = joinLocalPath(DEFAULT_OSS_PACKAGE_ROOT, taskName);
    ossTransferState.remoteUri = joinOssPath($('#ossRootInput')?.value || DEFAULT_OSS_ROOT, taskName);
    ossTransferState.remoteEntries = [];
    renderOssTransfer();
    refreshOssLocalDir(ossTransferState.localDir).catch(error => showNotice(error.message));
  }
  if (event.target?.id === 'ossRootInput') {
    const taskName = cleanTaskName($('#ossTaskNameInput')?.value || ossTransferState.taskName);
    ossTransferState.remoteUri = joinOssPath(event.target.value || DEFAULT_OSS_ROOT, taskName);
    ossTransferState.remoteEntries = [];
    renderOssTransfer();
  }
});
document.addEventListener('input', event => {
  if (event.target?.id === 'ossTaskNameInput') {
    ossTransferState.taskName = cleanTaskName(event.target.value);
  }
});
async function jumpEpisodeFramePage(input) {
  const framePage = dataPreviewState.preview?.frame_page || {};
  const nextPage = clampPage(input?.value, framePage.total_pages || 1);
  if (input) input.value = String(nextPage);
  if (!dataPreviewState.taskId || !dataPreviewState.episode || nextPage === framePage.page) return;
  try {
    await loadSingleEpisodePreview(dataPreviewState.taskId, dataPreviewState.episode, nextPage);
  } catch(e) {
    showNotice(e.message);
  }
}
document.addEventListener('keydown', async event => {
  if (event.key !== 'Enter' || event.target?.id !== 'episodeFramePageJumpInput') return;
  event.preventDefault();
  clearTimeout(pageJumpTimers.episodeFrame);
  await jumpEpisodeFramePage(event.target);
});
document.addEventListener('input', event => {
  if (event.target?.id === 'episodeFramePageJumpInput') {
    debouncePageJump('episodeFrame', () => jumpEpisodeFramePage(event.target));
  }
});
document.addEventListener('change', event => {
  if (event.target?.id === 'episodeFramePageJumpInput') jumpEpisodeFramePage(event.target);
});
document.addEventListener('blur', event => {
  if (event.target?.id === 'episodeFramePageJumpInput') jumpEpisodeFramePage(event.target);
}, true);
['statusFilter','taskSearch'].forEach(id => $(`#${id}`).addEventListener(id==='taskSearch'?'input':'change', () => { taskPage = 1; renderTasks(); }));
$('#prevTaskPage')?.addEventListener('click', () => { taskPage = Math.max(1, taskPage - 1); renderTasks(); });
$('#nextTaskPage')?.addEventListener('click', () => { taskPage += 1; renderTasks(); });
function jumpTaskPage(input) {
  const status = $('#statusFilter').value;
  const query = $('#taskSearch').value.trim().toLowerCase();
  const tasks = appState.tasks.filter(t => (!status || t.status === status) && (!query || t.name.toLowerCase().includes(query) || t.description.includes(query)));
  const totalPages = Math.max(1, Math.ceil(tasks.length / TASK_PAGE_SIZE));
  taskPage = clampPage(input?.value, totalPages);
  if (input) input.value = String(taskPage);
  renderTasks();
}
$('#taskPageJump')?.addEventListener('keydown', event => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  clearTimeout(pageJumpTimers.task);
  jumpTaskPage(event.currentTarget);
});
$('#taskPageJump')?.addEventListener('input', event => {
  debouncePageJump('task', () => jumpTaskPage(event.currentTarget));
});
$('#taskPageJump')?.addEventListener('change', event => jumpTaskPage(event.currentTarget));
$('#taskPageJump')?.addEventListener('blur', event => jumpTaskPage(event.currentTarget));

$('#deviceForm').addEventListener('submit', async event => {
  event.preventDefault(); const f=new FormData(event.currentTarget);
  const leftEe=f.get('left_ee'), rightEe=f.get('right_ee');
  const device={name:f.get('name'),arm:f.get('arm'),ee:leftEe===rightEe?leftEe:'none',left_ee:leftEe,right_ee:rightEe,input_mode:f.get('input_mode'),display_mode:f.get('display_mode'),xr_view:f.get('xr_view') || 'head',arm_reference_mode:f.get('arm_reference_mode') || 'head_position',img_server_ip:f.get('img_server_ip') || DEFAULT_IMAGE_SERVER_IP,webrtc_server_ip:f.get('webrtc_server_ip') || DEFAULT_WEBRTC_SERVER_IP,data_dir:f.get('data_dir') || DEFAULT_DATA_DIR,network_interface:f.get('network_interface'),frequency:Number(f.get('frequency')),init_arm_pose_file:f.get('init_arm_pose_file') || '',init_arm_pose_duration:Number(f.get('init_arm_pose_duration') || 5),headless:f.has('headless'),motion:f.has('motion'),hand_eye_replay_trajectory:f.get('hand_eye_replay_trajectory') || '',hand_eye_replay_time_scale:Number(f.get('hand_eye_replay_time_scale') || 1),hand_eye_record:f.has('hand_eye_record'),ik_replay_live_enable:f.has('ik_replay_live_enable'),ik_replay_live_url:f.get('ik_replay_live_url') || DEFAULT_IK_REPLAY_LIVE_URL,ik_replay_live_fps:Number(f.get('ik_replay_live_fps') || 10)};
  try { deviceFormDirty=false; applyState(await api('/api/device/save',{device})); showNotice('设备配置文件已更新','success'); } catch(error){deviceFormDirty=true;showNotice(error.message);}
});
$('#deviceForm').addEventListener('input',()=>deviceFormDirty=true);
$('#deviceForm [name=input_mode]').addEventListener('change',syncEndEffector);
$('#deviceForm [name=left_ee]').addEventListener('change',syncEndEffector);
$('#deviceForm [name=right_ee]').addEventListener('change',syncEndEffector);
$('#taskForm').addEventListener('submit', async event => {
  event.preventDefault(); const f=new FormData(event.currentTarget);
  const task={
    name:f.get('name'),
    instruction:f.get('instruction'),
    description:f.get('description'),
    target_episodes:Number(f.get('target_episodes')),
  };
  try { applyState(await api('/api/tasks/create',{task})); closeModal('taskModal'); showNotice('数采任务已创建','success'); } catch(error){showNotice(error.message);}
});
$('#trainingSetForm')?.addEventListener('submit', async event => {
  event.preventDefault();
  const f = new FormData(event.currentTarget);
  const payload = {
    training_set_id: editingTrainingSetId,
    name: f.get('name'),
    config_name: f.get('config_name'),
    h200_dataset_root: f.get('h200_dataset_root'),
    h200_assets_root: f.get('h200_assets_root'),
    action_dim: Number(f.get('action_dim') || 32),
    real_action_dim: Number(f.get('real_action_dim') || 14),
    action_horizon: Number(f.get('action_horizon') || 16),
    description: f.get('description'),
    task_ids: [],
  };
  try {
    const editing = Boolean(editingTrainingSetId);
    const result = await api(editingTrainingSetId ? '/api/training/update-set' : '/api/training/create-set', payload);
    applyState(result.state || result);
    closeModal('trainingSetModal');
    editingTrainingSetId = null;
    showNotice(editing ? '训练集已保存' : '训练集容器已创建，可从左侧选择任务加入', 'success');
  } catch (error) {
    showNotice(error.message);
  }
});
$('#postprocessForm').addEventListener('submit', async event => {
  event.preventDefault();
  const submitButton = event.submitter || $('#postprocessForm button[type="submit"]');
  const originalText = submitButton?.textContent || '';
  const f = new FormData(event.currentTarget);
  const task = appState.tasks.find(t => t.id === activePostprocessTaskId);
  if (task?.postprocess_status?.postprocess_busy && !task?.postprocess_status?.convert_running && !task?.postprocess_status?.normalize_running && !task?.postprocess_status?.package_running) {
    showNotice('其他处理任务正在运行，请等待完成');
    return;
  }
  const payload = {
    task_id: f.get('task_id'),
    repo_id: f.get('repo_id'),
    robot_type: f.get('robot_type'),
    camera_map: cameraSlotsToMap(event.currentTarget),
    image_size: f.get('image_size'),
    image_encoding: f.get('image_encoding'),
    jpeg_quality: f.get('jpeg_quality'),
    video_backend: f.get('video_backend'),
    image_writer_processes: f.get('image_writer_processes'),
    image_writer_threads: f.get('image_writer_threads'),
    resume: f.has('resume'),
    start_episode: f.get('start_episode'),
    batch_size: f.get('batch_size'),
    overwrite: f.has('overwrite'),
  };
  if (!payload.camera_map) {
    showNotice('OpenPI 三路相机不能全部选择 none，至少保留一路真实相机');
    return;
  }
  try {
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = '启动中...';
    }
    applyState(await api('/api/tasks/convert', payload));
    showNotice('LeRobot 转换任务已启动', 'success');
  } catch(error) {
    showNotice(error.message);
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.textContent = originalText;
    }
  }
});
$('#postprocessForm').addEventListener('input', () => {
  const task = appState.tasks.find(t => t.id === activePostprocessTaskId);
  updatePostprocessModalStatus(task);
});
$('#postprocessForm').addEventListener('change', () => {
  const form = $('#postprocessForm');
  if (form?.elements.resume && form?.elements.overwrite) {
    if (form.elements.resume.checked) {
      form.elements.overwrite.checked = false;
      form.elements.overwrite.disabled = true;
    } else {
      form.elements.overwrite.disabled = false;
    }
  }
  updateImageEncodingHint();
  const task = appState.tasks.find(t => t.id === activePostprocessTaskId);
  updatePostprocessModalStatus(task);
});
['camera_head', 'camera_left', 'camera_right'].forEach(name => {
  $(`#postprocessForm [name=${name}]`)?.addEventListener('change', updateCameraMapHint);
});
document.addEventListener('change', event => {
  if (event.target?.id === 'ossLocalPackage') {
    const option = event.target.selectedOptions?.[0];
    const input = $('#ossTaskNameInput');
    if (input && option) input.value = packageTaskName(option.textContent || '');
  }
  if (event.target?.id === 'manualTaskName') {
    syncManualDerivedFieldsFromTask();
  }
});
document.addEventListener('input', event => {
  const id = event.target?.id;
  if (id === 'manualTaskName') {
    syncManualDerivedFieldsFromTask();
    return;
  }
  if (id === 'manualNumTrainSteps') {
    syncManualModelTrainDir();
    syncManualExpNameFromSteps();
    return;
  }
  if (id === 'manualModelTrainDir' || id === 'manualPackageTimestamp' || id === 'manualExpName') {
    updateManualModelFileName();
  }
});
$('#normalizeButton').addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  const task = appState.tasks.find(t => t.id === activePostprocessTaskId);
  if (task?.postprocess_status?.postprocess_busy && !task?.postprocess_status?.convert_running && !task?.postprocess_status?.normalize_running && !task?.postprocess_status?.package_running) {
    showNotice('其他处理任务正在运行，请等待完成');
    return;
  }
  if (task && !task.postprocess_status?.lerobot_ready) {
    showNotice('请先转换为 LeRobot，再计算归一化');
    return;
  }
  const f = new FormData($('#postprocessForm'));
  const payload = {
    task_id: f.get('task_id'),
    repo_id: f.get('repo_id'),
    config_name: f.get('config_name'),
    openpi_dir: f.get('openpi_dir'),
    max_frames: f.get('max_frames'),
  };
  try {
    button.disabled = true;
    button.textContent = '启动中...';
    applyState(await api('/api/tasks/normalize', payload));
    showNotice('归一化统计任务已启动', 'success');
  } catch(error) {
    showNotice(error.message);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});

$('#packageButton')?.addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  const task = appState.tasks.find(t => t.id === activePostprocessTaskId);
  if (task?.postprocess_status?.postprocess_busy && !task?.postprocess_status?.convert_running && !task?.postprocess_status?.normalize_running && !task?.postprocess_status?.package_running) {
    showNotice('其他处理任务正在运行，请等待完成');
    return;
  }
  if (task && !task.postprocess_status?.lerobot_ready) {
    showNotice('请先转换为 LeRobot，再打包压缩数据');
    return;
  }
  const f = new FormData($('#postprocessForm'));
  const payload = {
    task_id: f.get('task_id'),
    repo_id: f.get('repo_id'),
  };
  try {
    button.disabled = true;
    button.textContent = '启动中...';
    applyState(await api('/api/tasks/package-lerobot', payload));
    showNotice('数据打包压缩任务已启动', 'success');
  } catch(error) {
    showNotice(error.message);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});

$('#packageAssetsButton')?.addEventListener('click', async event => {
  const button = event.currentTarget;
  const originalText = button.textContent;
  const task = appState.tasks.find(t => t.id === activePostprocessTaskId);
  if (task?.postprocess_status?.postprocess_busy && !task?.postprocess_status?.convert_running && !task?.postprocess_status?.normalize_running && !task?.postprocess_status?.package_running) {
    showNotice('其他处理任务正在运行，请等待完成');
    return;
  }
  if (task && !task.postprocess_status?.norm_stats_ready) {
    showNotice('请先计算归一化，再打包压缩归一化值');
    return;
  }
  const f = new FormData($('#postprocessForm'));
  const payload = {
    task_id: f.get('task_id'),
    repo_id: f.get('repo_id'),
    config_name: f.get('config_name'),
  };
  try {
    button.disabled = true;
    button.textContent = '启动中...';
    applyState(await api('/api/tasks/package-assets', payload));
    showNotice('归一化打包压缩任务已启动', 'success');
  } catch(error) {
    showNotice(error.message);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});

document.addEventListener('click', async event => {
  if (event.target?.id === 'refreshTrainingTemplate' || event.target?.id === 'refreshTrainingCommand') {
    const button = event.target;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '刷新中...';
    try {
      await refresh(false);
      showNotice(event.target.id === 'refreshTrainingTemplate' ? '训练模板已刷新' : '训练命令已刷新', 'success');
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
    return;
  }
  const button=event.target.closest('[data-action]'); if(!button)return;
  event.preventDefault();
  holdAutoRefresh(1600);
  const id=button.dataset.id, action=button.dataset.action;
  const task=appState.tasks.find(t=>t.id===Number(id));
  if(action==='copy-text'){
    await copyText(button.dataset.copy || '');
    return;
  }
  if(action==='copy-target'){
    const target = document.getElementById(button.dataset.target);
    await copyText(target?.textContent || '');
    return;
  }
  if(action==='timestamp-now'){
    const input = $('#manualPackageTimestamp');
    if (input) {
      input.value = currentTimestampText();
      updateManualModelFileName();
      showNotice('已生成当前时间戳', 'success');
    }
    return;
  }
  if(action==='view-task') showTask(task);
  if(action==='preview-prev'){
    const episodes = dataPreviewState.preview?.episodes || [];
    if(dataPreviewState.index < episodes.length - 1){dataPreviewState.index += 1; renderDataPreviewEpisode();}
    return;
  }
  if(action==='preview-next'){
    if(dataPreviewState.index > 0){dataPreviewState.index -= 1; renderDataPreviewEpisode();}
    return;
  }
  if(action==='episode-frame-prev' || action==='episode-frame-next'){
    const framePage = dataPreviewState.preview?.frame_page || {};
    const nextPage = action === 'episode-frame-prev'
      ? Math.max(1, (framePage.page || 1) - 1)
      : Math.min(framePage.total_pages || 1, (framePage.page || 1) + 1);
    if (!dataPreviewState.taskId || !dataPreviewState.episode || nextPage === framePage.page) return;
    button.disabled = true;
    try {
      await loadSingleEpisodePreview(dataPreviewState.taskId, dataPreviewState.episode, nextPage);
    } catch(e) {
      button.disabled = false;
      showNotice(e.message);
    }
    return;
  }
  if(action==='episode-frame-jump'){
    const framePage = dataPreviewState.preview?.frame_page || {};
    const input = $('#episodeFramePageJumpInput');
    const nextPage = clampPage(input?.value, framePage.total_pages || 1);
    if (!dataPreviewState.taskId || !dataPreviewState.episode) return;
    button.disabled = true;
    try {
      await loadSingleEpisodePreview(dataPreviewState.taskId, dataPreviewState.episode, nextPage);
    } catch(e) {
      button.disabled = false;
      showNotice(e.message);
    }
    return;
  }
  if(action==='delete-episode'){
    const episode = button.dataset.episode;
    if(!episode || !confirm(`确定删除 ${episode} 吗？该操作不可恢复。`)) return;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '删除中...';
    try{
      const result = await api('/api/tasks/delete-episode',{task_id:id,episode});
      applyState(result);
      if (activeView === 'dataList') await loadDataList({silent:true});
      renderDataPreview(result.preview);
      showNotice(`已删除 ${episode}`,'success');
    }catch(e){
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
    return;
  }
  if(action==='preview-episode'){
    $('#dataPreviewTitle').textContent=`数据记录预览 · ${task?.name || ''}`;
    $('#dataPreviewSubtitle').textContent=`正在读取 ${button.dataset.episode || ''} ...`;
    $('#dataPreviewSummary').innerHTML='';
    $('#dataPreviewList').innerHTML='<div class="data-preview-empty">正在加载...</div>';
    openModal('dataPreviewModal');
    try{await loadSingleEpisodePreview(id, button.dataset.episode, 1);}catch(e){$('#dataPreviewList').innerHTML=`<div class="data-preview-empty">${escapeHtml(e.message)}</div>`;}
    return;
  }
  if(action==='preview-data'){
    $('#dataPreviewTitle').textContent=`数据预览 · ${task?.name || ''}`;
    $('#dataPreviewSubtitle').textContent='正在读取数据目录...';
    $('#dataPreviewSummary').innerHTML='';
    $('#dataPreviewList').innerHTML='<div class="data-preview-empty">正在加载...</div>';
    openModal('dataPreviewModal');
    try{renderDataPreview(await api('/api/tasks/preview',{task_id:id,limit:20}));}catch(e){$('#dataPreviewList').innerHTML=`<div class="data-preview-empty">${escapeHtml(e.message)}</div>`;}
  }
  if(action==='postprocess-data') showPostprocess(task);
  if(action==='add-selected-training'){
    const target = $('#processingTrainingSetTarget') || $('#trainingSetTarget');
    const taskIds = $$('input[name="processing_task"]:checked, input[name="training_task"]:checked').map(input => Number(input.value));
    if (!target?.value) {
      showNotice('请先创建训练集');
      return;
    }
    if (!taskIds.length) {
      showNotice('请先勾选要加入训练集的任务');
      return;
    }
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '加入中...';
    try {
      const result = await api('/api/training/add-tasks', {training_set_id: target.value, task_ids: taskIds});
      applyState(result.state || result);
      showNotice('已加入训练集', 'success');
    } catch(e) {
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
    return;
  }
  if(action==='toggle-training-members'){
    const key = String(id);
    if (expandedTrainingSetIds.has(key)) expandedTrainingSetIds.delete(key);
    else expandedTrainingSetIds.add(key);
    localStorage.setItem('teleop.expandedTrainingSets', JSON.stringify([...expandedTrainingSetIds]));
    renderTrainingSets();
    return;
  }
  if(action==='remove-training-task'){
    if (!confirm('确定从训练集中移除该任务吗？移除后需要重新计算训练集归一化。')) return;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '移除中...';
    try {
      const result = await api('/api/training/remove-task', {training_set_id:id, task_id:button.dataset.taskId});
      applyState(result.state || result);
      showNotice('已从训练集移除', 'success');
    } catch(e) {
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
    return;
  }
  if(action==='edit-training'){
    const item = (appState.training?.training_sets || []).find(set => String(set.id) === String(id));
    if (!item) {
      showNotice('训练集不存在，请刷新后重试');
      return;
    }
    openTrainingSetModal(item);
    return;
  }
  if(action==='normalize-training'){
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '启动中...';
    try {
      const result = await api('/api/training/normalize', {training_set_id:id});
      applyState(result.state || result);
      showNotice('训练集归一化任务已启动', 'success');
    } catch(e) {
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
    return;
  }
  if(action==='package-training'){
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '生成中...';
    try {
      const result = await api('/api/training/package', {training_set_id:id});
      applyState(result.state || result);
      showNotice(`训练交接包已生成：${result.package?.path || ''}`, 'success');
    } catch(e) {
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
    return;
  }
  if(action==='oss-list'){
    const taskName = cleanTaskName($('#ossTaskNameInput')?.value || ossTransferState.taskName);
    const uri = $('#ossListUriInput')?.value || joinOssPath($('#ossRootInput')?.value || appState.oss_transfer?.oss_root || DEFAULT_OSS_ROOT, taskName);
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '刷新中...';
    ossTransferState.loading = true;
    renderOssTransfer();
    try {
      const result = await api('/api/oss/list', {uri});
      ossTransferState.remoteUri = result.uri || uri;
      ossTransferState.remoteEntries = result.entries || [];
      showNotice(`OSS 列表已刷新：${ossTransferState.remoteEntries.length} 项`, 'success');
    } catch(e) {
      showNotice(e.message);
    } finally {
      ossTransferState.loading = false;
      button.disabled = false;
      button.textContent = originalText;
      renderOssTransfer();
    }
    return;
  }
  if(action==='oss-local-list' || action==='oss-open-local-dir'){
    const taskName = cleanTaskName($('#ossTaskNameInput')?.value || ossTransferState.taskName);
    const directory = action === 'oss-open-local-dir' ? button.dataset.path : ($('#ossLocalDirInput')?.value || joinLocalPath(DEFAULT_OSS_PACKAGE_ROOT, taskName));
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '打开中...';
    try {
      await refreshOssLocalDir(directory, {notice: true});
    } catch(e) {
      showNotice(e.message);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
    return;
  }
  if(action==='oss-upload'){
    const localPath = button.dataset.localPath || $('#ossLocalPackage')?.value || '';
    const ossRoot = $('#ossRootInput')?.value || appState.oss_transfer?.oss_root || DEFAULT_OSS_ROOT;
    const taskName = cleanTaskName($('#ossTaskNameInput')?.value || ossTransferState.taskName || packageTaskName(button.dataset.packageName || ''));
    if (!localPath) return showNotice('请先选择本地压缩包');
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '启动中...';
    try {
      applyState(await api('/api/oss/upload', {local_path: localPath, oss_root: ossRoot, task_name: taskName}));
      showNotice('OSS 上传任务已启动', 'success');
    } catch(e) {
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
    return;
  }
  if(action==='oss-download'){
    const ossUri = $('#ossRemoteObject')?.value || '';
    const taskName = cleanTaskName($('#ossTaskNameInput')?.value || ossTransferState.taskName);
    const targetDir = $('#ossDownloadDir')?.value || joinLocalPath(DEFAULT_MODEL_DOWNLOAD_ROOT, taskName);
    if (!ossUri) return showNotice('请先选择远端模型文件');
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '启动中...';
    try {
      applyState(await api('/api/oss/download', {oss_uri: ossUri, target_dir: targetDir}));
      showNotice('模型下载任务已启动', 'success');
    } catch(e) {
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
    return;
  }
  if(action==='manual-refresh'){
    if (activeView === 'trainingCommand') renderTrainingCommandView();
    else renderActiveView();
    showNotice('训练命令已生成', 'success');
    return;
  }
  if(action==='delivery-add-template'){
    addDeliveryTemplateEditor();
    return;
  }
  if(action==='delivery-delete-template'){
    deleteDeliveryTemplateEditor(button);
    return;
  }
  if(action==='delivery-save-templates'){
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '保存中...';
    try {
      const templates = collectDeliveryTemplates();
      if (!templates.length) throw new Error('没有可保存的模板');
      const result = await api('/api/delivery/templates/save', {templates});
      if (result.state) applyState(result.state);
      else if (result.delivery) appState.delivery = result.delivery;
      if (activeView === 'trainingTemplate') renderTrainingTemplateView();
      else renderActiveView();
      showNotice('命令模板已保存', 'success');
    } catch(e) {
      showNotice(e.message === 'Failed to fetch' ? '保存请求没有到达服务端，请刷新页面后重试' : e.message);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
    return;
  }
  if(action==='delivery-reset-templates'){
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '恢复中...';
    try {
      const result = await api('/api/delivery/templates/reset', {});
      if (result.state) applyState(result.state);
      else if (result.delivery) appState.delivery = result.delivery;
      if (activeView === 'trainingTemplate') renderTrainingTemplateView();
      else renderActiveView();
      showNotice('已恢复默认命令模板', 'success');
    } catch(e) {
      showNotice(e.message);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
    return;
  }
  if(action==='start-task'){
    if(task.active){activeTaskId=task.id;renderRuntime();openModal('controlModal');return;}
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '启动采集中...';
    activeTaskId = task.id;
    showTask(task);
    setRuntimePending('process_start', '采集进程启动中...', {taskId: task.id, timeoutMs: 20000});
    try{
      applyState(await api('/api/start',{task_id:task.id}));
      openModal('controlModal');
    }catch(e){
      clearRuntimePending();
      button.disabled = false;
      button.textContent = originalText;
      showNotice(e.message);
    }
  }
  if(action==='archive-task'){
    if (button.dataset.archiveBlocked === 'true') {
      const runningArchiveJob = appState.archive_jobs?.find(job => job.running);
      const taskName = runningArchiveJob?.task_name || '其他任务';
      showNotice(`已有归档任务正在运行：${taskName}，请等待完成后再归档`);
      return;
    }
    const originalText=button.textContent;
    button.disabled=true;
    button.textContent='归档中...';
    try{
      const result=await api('/api/tasks/archive',{task_id:task.id});
      applyState(result);
      showNotice(`归档任务已启动：${result.archive_job?.task_name || task.name}`,'success');
    }catch(e){
      button.disabled=false;
      button.textContent=originalText;
      showNotice(e.message);
    }
  }
});
async function control(action){
  const process = appState.process || {};
  const teleop = appState.teleop || {};
  const taskId = process.task?.id ?? activeTaskId;
  const pendingByAction = {
    start: ['teleop_start', '开始遥操指令已发送...'],
    stop: ['process_stop', '结束采集指令已发送，等待进程退出...'],
  };
  let pending = pendingByAction[action];
  if (action === 'record') {
    pending = teleop.RECORD_RUNNING
      ? ['record_stop', '结束录制指令已发送，正在保存...']
      : ['record_start', '开始录制指令已发送...'];
  }
  if (pending) {
    setRuntimePending(pending[0], pending[1], {taskId, timeoutMs: action === 'stop' ? 20000 : 12000});
    renderRuntime();
  }
  try{
    applyState(await api('/api/control',{action}));
  }catch(e){
    clearRuntimePending();
    renderRuntime();
    showNotice(e.message);
  }
}
$('#startControl').addEventListener('click',()=>control('start')); $('#recordControl').addEventListener('click',()=>control('record')); $('#handEyeControl').addEventListener('click',()=>control('hand_eye')); $('#stopControl').addEventListener('click',()=>control('stop'));

refresh(false); setInterval(()=>refresh(true, {auto:true}),5000);
