document.addEventListener('DOMContentLoaded', () => {
    initEval();
});

let evalState = {
    localModels: [],
    mergedModels: [],
    selectedModel: null 
};

let currentTaskId = null;
let pollTimer = null;
let deleteTargetId = null; // 用于删除功能

function initEval() {
    loadAllModels();
    setupEvalUI();
    setupDatasetSource();
    setupSidebar(); 
    setupDeleteModal();
    initSegmentedControls();
}

function initSegmentedControls() {
    document.querySelectorAll('.segmented-control').forEach(control => {
        const options = control.querySelectorAll('.segmented-option');
        const hiddenInput = control.parentElement.querySelector('input[type="hidden"]') ||
            document.getElementById(control.id && control.id.replace('-control', '') ? control.id.replace('-control', '') : null);
        
        options.forEach(opt => {
            opt.addEventListener('click', () => {
                options.forEach(o => o.classList.remove('active', 'selected'));
                opt.classList.add('active', 'selected');
                const inp = control.id === 'eval-dataset-source-control' ? document.getElementById('eval-dataset-source') : hiddenInput;
                if (inp) inp.value = opt.dataset.value;
                if (control.id === 'eval-dataset-source-control') onDatasetSourceChange(opt.dataset.value);
            });
        });
    });
}

function onDatasetSourceChange(value) {
    document.getElementById('eval-source-builtin').style.display = value === 'builtin' ? 'block' : 'none';
    document.getElementById('eval-source-repo').style.display = value === 'repo' ? 'block' : 'none';
    document.getElementById('eval-source-custom').style.display = value === 'custom' ? 'block' : 'none';
    if (value === 'repo') loadTestsetRepoOptions();
}

let testsetsList = [];
async function loadTestsetRepoOptions() {
    try {
        const res = await fetch('/api/testset/list');
        const data = await res.json();
        testsetsList = (data.testsets || []) || [];
        const ul = document.getElementById('eval-repo-options');
        ul.innerHTML = '';
        if (!testsetsList.length) {
            ul.innerHTML = '<li class="ios-option" style="color:#999;">暂无测试集</li>';
            return;
        }
        testsetsList.forEach((t, i) => {
            const li = document.createElement('li');
            li.className = 'ios-option' + (i === 0 ? ' selected' : '');
            li.dataset.testsetId = t.testset_id || '';
            li.dataset.hfDataset = t.hf_dataset || '';
            li.dataset.hfSubset = t.hf_subset || '';
            li.dataset.hfSplit = t.hf_split || 'train';
            li.innerHTML = '<span class="opt-name">' + (t.name || t.hf_dataset || t.testset_id) + '</span>';
            li.addEventListener('click', () => selectRepoTestset(t));
            ul.appendChild(li);
        });
        document.getElementById('eval-repo-trigger').querySelector('.selected-text').textContent = testsetsList[0] ? (testsetsList[0].name || testsetsList[0].hf_dataset) : '请选择测试集';
        document.getElementById('eval-testset-id').value = testsetsList[0] ? (testsetsList[0].testset_id || '') : '';
        document.getElementById('eval-hf-dataset').value = testsetsList[0] ? (testsetsList[0].hf_dataset || '') : '';
        if (testsetsList[0]) fillSubsetSplitFromTestset(testsetsList[0]);
    } catch (e) {
        console.error(e);
        document.getElementById('eval-repo-options').innerHTML = '<li class="ios-option" style="color:#999;">加载失败</li>';
    }
}

function selectRepoTestset(t) {
    document.getElementById('eval-repo-trigger').querySelector('.selected-text').textContent = t.name || t.hf_dataset || t.testset_id;
    document.getElementById('eval-testset-id').value = t.testset_id || '';
    document.getElementById('eval-hf-dataset').value = t.hf_dataset || '';
    fillSubsetSplitFromTestset(t);
    document.querySelectorAll('#eval-repo-options .ios-option').forEach((o, i) => o.classList.toggle('selected', o.dataset.testsetId === (t.testset_id || '')));
    document.getElementById('eval-repo-options').classList.remove('open');
}

function fillSubsetSplitFromTestset(t) {
    const subsetVal = t.hf_subset || '';
    const splitVal = t.hf_split || 'train';
    const subsetUl = document.getElementById('eval-subset-options');
    const splitUl = document.getElementById('eval-split-options');
    subsetUl.innerHTML = '<li class="ios-option selected" data-value="">— 无</li>';
    splitUl.innerHTML = '<li class="ios-option selected" data-value="train">train</li><li class="ios-option" data-value="validation">validation</li><li class="ios-option" data-value="test">test</li>';
    document.getElementById('eval-subset-trigger').querySelector('.selected-text').textContent = subsetVal || '— 无';
    document.getElementById('eval-split-trigger').querySelector('.selected-text').textContent = splitVal;
    document.getElementById('eval-subset').value = subsetVal;
    document.getElementById('eval-split').value = splitVal;
    if (t.hf_dataset) {
        fetch('/api/dataset/hf_info', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hf_dataset: t.hf_dataset }) })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'success' && (d.configs && d.configs.length || d.splits && d.splits.length)) {
                    if (d.configs && d.configs.length) {
                        subsetUl.innerHTML = '<li class="ios-option" data-value="">— 无</li>' + d.configs.map(c => '<li class="ios-option' + (c === subsetVal ? ' selected' : '') + '" data-value="' + c + '">' + c + '</li>').join('');
                    }
                    if (d.splits && d.splits.length) {
                        splitUl.innerHTML = d.splits.map(s => '<li class="ios-option' + (s === splitVal ? ' selected' : '') + '" data-value="' + s + '">' + s + '</li>').join('');
                    }
                }
            })
            .catch(() => {});
    }
}

function setupDatasetSource() {
    onDatasetSourceChange(document.getElementById('eval-dataset-source').value);
    document.getElementById('eval-hf-search-btn').addEventListener('click', doHfSearch);
    document.getElementById('eval-hf-download-btn').addEventListener('click', doHfDownloadAndAdd);
}

let customHfDataset = null;
let customConfigs = [];
let customSplits = [];

async function doHfSearch() {
    const q = document.getElementById('eval-hf-search-input').value.trim();
    if (!q) { alert('请输入数据集名称或关键词'); return; }
    const btn = document.getElementById('eval-hf-search-btn');
    btn.disabled = true;
    btn.textContent = '搜索中...';
    document.getElementById('eval-hf-results').innerHTML = '<span style="color:#999;">搜索中...</span>';
    try {
        const res = await fetch('/api/hf/datasets/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q, limit: 20 }) });
        const data = await res.json();
        if (data.status !== 'success' || !data.results || !data.results.length) {
            document.getElementById('eval-hf-results').innerHTML = '<span style="color:#999;">未找到结果</span>';
            return;
        }
        document.getElementById('eval-hf-results').innerHTML = data.results.map(r => 
            '<div class="eval-hf-result-item" data-id="' + (r.id || '') + '" style="padding:6px 8px; border-radius:6px; cursor:pointer; margin-bottom:4px; background:var(--apple-bg-secondary, #f5f5f7);">' + (r.id || '') + '</div>'
        ).join('');
        document.querySelectorAll('.eval-hf-result-item').forEach(el => {
            el.addEventListener('click', () => selectCustomHfDataset(el.dataset.id));
        });
    } catch (e) {
        document.getElementById('eval-hf-results').innerHTML = '<span style="color:#ff3b30;">搜索失败</span>';
    }
    btn.disabled = false;
    btn.textContent = '搜索';
}

async function selectCustomHfDataset(hfId) {
    customHfDataset = hfId;
    document.querySelectorAll('.eval-hf-result-item').forEach(el => el.style.background = el.dataset.id === hfId ? 'var(--apple-blue)' : 'var(--apple-bg-secondary, #f5f5f7)');
    document.getElementById('eval-custom-subset-split').style.display = 'block';
    try {
        const res = await fetch('/api/dataset/hf_info', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hf_dataset: hfId }) });
        const data = await res.json();
        if (data.status === 'success') {
            customConfigs = data.configs || [];
            customSplits = data.splits || ['train', 'validation', 'test'];
            const subUl = document.getElementById('eval-custom-subset-options');
            const splitUl = document.getElementById('eval-custom-split-options');
            subUl.innerHTML = '<li class="ios-option selected" data-value="">— 无</li>' + (customConfigs.map(c => '<li class="ios-option" data-value="' + c + '">' + c + '</li>').join(''));
            splitUl.innerHTML = customSplits.map(s => '<li class="ios-option' + (s === 'test' ? ' selected' : '') + '" data-value="' + s + '">' + s + '</li>').join('');
            document.getElementById('eval-custom-subset-trigger').querySelector('.selected-text').textContent = '— 无';
            document.getElementById('eval-custom-split-trigger').querySelector('.selected-text').textContent = 'test';
            document.getElementById('eval-custom-subset').value = '';
            document.getElementById('eval-custom-split').value = 'test';
        }
    } catch (e) { console.error(e); }
}

async function doHfDownloadAndAdd() {
    if (!customHfDataset) { alert('请先搜索并选择数据集'); return; }
    const subset = document.getElementById('eval-custom-subset').value.trim();
    const split = document.getElementById('eval-custom-split').value.trim() || 'train';
    const btn = document.getElementById('eval-hf-download-btn');
    btn.disabled = true;
    btn.textContent = '下载中...';
    try {
        const res = await fetch('/api/testset/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: customHfDataset, hf_dataset: customHfDataset, hf_subset: subset || undefined, hf_split: split })
        });
        const data = await res.json();
        if (data.status === 'success') {
            alert('已添加到测试集仓库，可在「测试集仓库」页面查看。');
            loadTestsetRepoOptions();
            document.getElementById('eval-dataset-source').value = 'repo';
            onDatasetSourceChange('repo');
        } else {
            alert('添加失败: ' + (data.message || ''));
        }
    } catch (e) {
        alert('请求失败');
    }
    btn.disabled = false;
    btn.textContent = '下载并添加到测试集仓库';
}

async function loadHistoryList() {
    const listEl = document.getElementById('history-list');
    listEl.innerHTML = '';
    
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        
        if (!data.history || data.history.length === 0) {
            listEl.innerHTML = '<div style="padding:20px; text-align:center; color:#999; font-size:0.8rem;">暂无融合记录</div>';
            return;
        }

        data.history.forEach(item => {
            const div = document.createElement('div');
            // 高亮当前选中的任务
            div.className = `history-item ${currentTaskId === item.id ? 'active' : ''}`;
            
            // 1. 时间格式化
            const date = new Date(item.created_at * 1000);
            const timeStr = `${(date.getMonth()+1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
            
            // 2. 名称
            const displayName = item.custom_name || (item.models || []).join(' + ');

            // 3. 构建 HTML
            div.innerHTML = `
                <div class="h-content">
                    <div class="h-title">
                        <i class="ri-git-merge-line" style="font-size:0.8rem; margin-right:4px; opacity:0.7;"></i>
                        ${displayName}
                    </div>
                    <div class="h-meta">
                        <span>${timeStr}</span>
                        <span>•</span>
                        <span>${item.method || 'Merge'}</span>
                    </div>
                </div>
                <button class="btn-delete-history" onclick="confirmDelete(event, '${item.id}')">
                    <i class="ri-delete-bin-line"></i>
                </button>
            `;

            // 5. 点击跳转 (回显结果)
            div.onclick = (e) => {
                if (e.target.closest('.btn-delete-history')) return; 
                loadHistoryDetail(item);
                
                document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
                div.classList.add('active');
            };

            listEl.appendChild(div);
        });
    } catch (e) {
        console.error(e);
        listEl.innerHTML = '<div style="padding:15px; color:#999; text-align:center">加载失败</div>';
    }
}

// 加载历史详情：主要是为了显示结果
function loadHistoryDetail(item) {
    // 1. 如果有 metrics 结果，直接渲染
    if (item.metrics) {
        renderResults(item.metrics);
        setStatusUI("历史记录已加载", 100, false, '#86868b');
    } else {
        // 如果 metadata 里没有 metrics (比如旧任务)，提示无法显示
        if (item.status === 'success') {
             alert("该历史记录缺少详细测试数据，无法渲染图表。");
        } else {
             alert(`该任务状态为 ${item.status}，没有测试结果。`);
        }
        document.getElementById('results-section').style.display = 'none';
    }

    // 2. 回显模型名称到放置区 (视觉上的还原)
    const dropzoneContent = document.querySelector('.dropzone-content');
    const selectedContainer = document.getElementById('selected-eval-model');
    const startBtn = document.getElementById('start-eval');
    const clearBtn = document.getElementById('clear-eval');

    dropzoneContent.style.display = 'none';
    selectedContainer.innerHTML = '';
    
    // 模拟一个选中项用于展示
    let nameToShow = item.custom_name || item.model_name || "Historical Task";
    if (item.type === 'merge') nameToShow += " (Fused)";

    const div = document.createElement('div');
    div.className = 'selected-model-item';
    div.innerHTML = `
        <div style="display:flex; align-items:center; gap:10px;">
             <i class="ri-history-line" style="color:var(--apple-gray);"></i>
             <span>${nameToShow}</span>
        </div>
        <div style="font-size:0.8rem; color:#999;">(只读)</div>
    `;
    selectedContainer.appendChild(div);

    // 禁用开始按钮，允许重置
    startBtn.disabled = true;
    startBtn.innerText = "历史模式";
    clearBtn.disabled = false;
    clearBtn.innerText = "重置";
    
    // 移动视图
    if (window.innerWidth < 768) {
        document.getElementById('close-sidebar').click(); // 移动端自动关侧栏
    }
}

// ================== 删除功能 (与 Index 统一) ==================

function setupDeleteModal() {
    const modal = document.getElementById('delete-modal-overlay');
    const cancelBtn = document.getElementById('cancel-delete');
    const confirmBtn = document.getElementById('confirm-delete');

    // 取消
    cancelBtn.addEventListener('click', () => {
        modal.classList.remove('show');
        setTimeout(() => modal.style.display = 'none', 300);
        deleteTargetId = null;
    });

    // 确认删除
    confirmBtn.addEventListener('click', async () => {
        if (!deleteTargetId) return;
        
        const originalText = confirmBtn.innerText;
        confirmBtn.innerText = "删除中...";
        confirmBtn.disabled = true;

        try {
            await fetch(`/api/history/${deleteTargetId}`, { method: 'DELETE' });
            
            // 刷新列表
            await loadHistoryList();
            
            // 如果删除的是当前显示的，重置界面
            const currentDisplayedName = document.querySelector('#selected-eval-model span')?.innerText;
            if (currentDisplayedName) { 
                 // 这里只是简单判断，实际上可以做得更细，但重置总没错
                 document.getElementById('clear-eval').click();
            }

            // 关闭弹窗
            modal.classList.remove('show');
            setTimeout(() => modal.style.display = 'none', 300);
        } catch (e) {
            console.error(e);
            alert("删除失败");
        } finally {
            confirmBtn.innerText = originalText;
            confirmBtn.disabled = false;
            deleteTargetId = null;
        }
    });
}

// 暴露给 HTML onclick 调用
window.confirmDelete = function(e, id) {
    if (e) e.stopPropagation();
    deleteTargetId = id;
    const modal = document.getElementById('delete-modal-overlay');
    modal.style.display = 'flex';
    modal.offsetHeight; // 触发重绘
    modal.classList.add('show');
}

// ================== 原有逻辑保持不变 (模型加载、测试执行等) ==================

// 1. 加载所有模型
async function loadAllModels() {
    const localRes = await fetch('/api/models');
    const localData = await localRes.json();
    evalState.localModels = localData.models || [];
    renderModelGrid(evalState.localModels, 'local-models-container', 'local');

    const mergedRes = await fetch('/api/merged_models');
    const mergedData = await mergedRes.json();
    evalState.mergedModels = mergedData.models || [];
    renderModelGrid(evalState.mergedModels, 'merged-models-container', 'merged');
}

// 2. 渲染模型网格
function renderModelGrid(models, containerId, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (models.length === 0) {
        container.innerHTML = '<div class="empty" style="color:#999; font-size:0.9rem;">无可用模型</div>';
        return;
    }

    models.forEach(model => {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.style.padding = '15px'; 
        card.dataset.modelType = type; 
        card.dataset.modelValue = model.path; 
        card.dataset.modelName = model.name;

        card.innerHTML = `
            <div style="display:flex; align-items:center; gap:10px;">
                <i class="${type === 'local' ? 'ri-hard-drive-2-line' : 'ri-git-merge-line'}" style="color:var(--apple-blue); font-size:1.2rem;"></i>
                <div style="text-align:left;">
                    <h4 style="font-size:0.8rem; margin:0;">${model.name}</h4>
                    <p style="font-size:0.75rem; margin:0; opacity:0.6;">${type === 'local' ? 'Base Model' : 'Fused Model'}</p>
                </div>
            </div>
        `;

        card.addEventListener('click', () => selectModel(model, type));
        container.appendChild(card);
    });
}

// 3. 选择模型
function selectModel(model, type) {
    if (document.getElementById('start-eval').disabled === true && document.getElementById('clear-eval').innerText === "停止测试") {
         alert("任务运行中，请勿切换模型");
         return;
    }

    evalState.selectedModel = { ...model, type };
    renderSelectedModel();
}

function renderSelectedModel() {
    const container = document.getElementById('selected-eval-model');
    const dropzoneContent = document.querySelector('.dropzone-content');
    const startBtn = document.getElementById('start-eval');
    const clearBtn = document.getElementById('clear-eval');

    container.innerHTML = '';

    if (!evalState.selectedModel) {
        dropzoneContent.style.display = 'block';
        startBtn.disabled = true;
        clearBtn.disabled = true;
        return;
    }

    dropzoneContent.style.display = 'none';
    startBtn.disabled = false;
    clearBtn.disabled = false;

    const item = document.createElement('div');
    item.className = 'selected-model-item';
    item.innerHTML = `
        <div style="display:flex; align-items:center; gap:10px;">
             <i class="${evalState.selectedModel.type === 'local' ? 'ri-hard-drive-2-line' : 'ri-git-merge-line'}" style="color:var(--apple-blue);"></i>
             <span>${evalState.selectedModel.name}</span>
        </div>
        <button class="remove-btn" onclick="clearSelection()">×</button>
    `;
    container.appendChild(item);
}

window.clearSelection = function() {
    if (document.getElementById('clear-eval').innerText === "停止测试") return;
    evalState.selectedModel = null;
    renderSelectedModel();
}

// 4. UI 事件
function setupEvalUI() {
    initDropdowns(); 

    document.getElementById('start-eval').addEventListener('click', startEvaluation);
    
    const clearBtn = document.getElementById('clear-eval');
    clearBtn.addEventListener('click', () => {
        if (clearBtn.innerText === "停止测试") {
            stopEvalTask();
        } else {
            clearSelection();
            document.getElementById('results-section').style.display = 'none';
            // 重置按钮文字
            document.getElementById('start-eval').innerText = "开始测试";
            document.getElementById('start-eval').disabled = true; // 因为 clearSelection 会清空模型
            
            // 如果是从历史记录重置回来的，需要刷新一下列表状态
            loadHistoryList(); 
        }
    });
}

async function startEvaluation() {
    if (!evalState.selectedModel) return;
    
    const limit = document.getElementById('eval-limit').value;
    const startBtn = document.getElementById('start-eval');
    const clearBtn = document.getElementById('clear-eval');
    const statusDiv = document.getElementById('task-status');
    const progressContainer = document.querySelector('.progress-container');

    startBtn.disabled = true;
    startBtn.innerText = "正在初始化...";
    clearBtn.innerText = "停止测试";
    statusDiv.style.opacity = '1';
    progressContainer.style.display = 'block';
    
    setStatusUI("正在提交评估任务...", 0, true);

    const source = document.getElementById('eval-dataset-source').value;
    const payload = {
        model_path: evalState.selectedModel.path,
        model_name: evalState.selectedModel.name,
        task_type: 'eval_only',
        limit: limit
    };
    if (source === 'builtin') {
        payload.dataset = document.getElementById('eval-dataset').value;
    } else if (source === 'repo') {
        payload.testset_id = document.getElementById('eval-testset-id').value;
        payload.hf_dataset = document.getElementById('eval-hf-dataset').value || null;
        payload.hf_subset = document.getElementById('eval-subset').value || null;
        payload.hf_split = document.getElementById('eval-split').value || null;
    } else {
        payload.hf_dataset = customHfDataset;
        payload.hf_subset = document.getElementById('eval-custom-subset').value;
        payload.hf_split = document.getElementById('eval-custom-split').value;
    }
    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            currentTaskId = data.task_id;
            pollStatus(currentTaskId);
            // 刷新历史列表，显示新任务
            loadHistoryList(); 
        } else {
            alert("启动失败: " + data.message);
            resetUI();
        }
    } catch (e) {
        console.error(e);
        alert("网络错误");
        resetUI();
    }
}

async function stopEvalTask() {
    if (!currentTaskId) return;
    if(!confirm("确定停止测试？")) return;
    
    await fetch(`/api/stop/${currentTaskId}`, { method: 'POST' });
    resetUI();
    setStatusUI("测试已停止", 0, false, '#86868b');
}

function pollStatus(taskId) {
    if (pollTimer) clearInterval(pollTimer);
    
    pollTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${taskId}`);
            const data = await res.json();
            
            if (data.status === 'running' || data.status === 'queued') {
                const msg = data.status === 'queued' ? `排队中 (${data.queue_position})` : (data.message || "正在评估...");
                const prog = data.status === 'queued' ? 100 : (data.progress || 0);
                const stripe = data.status === 'queued';
                
                setStatusUI(msg, prog, stripe);
                document.getElementById('start-eval').innerText = "正在运行...";
            } 
            else if (['completed', 'error', 'stopped'].includes(data.status)) {
                clearInterval(pollTimer);
                currentTaskId = null;
                resetUI();
                
                if (data.status === 'completed') {
                    setStatusUI("评估完成", 100, false, '#34c759');
                    // 刷新历史列表状态
                    loadHistoryList();
                    
                    if (data.result && data.result.metrics) {
                        renderResults(data.result.metrics);
                    } else {
                        console.error("Result format error:", data.result);
                    }
                    
                } else {
                    setStatusUI(data.message || "任务失败", 0, false, '#ff3b30');
                }
            }
        } catch(e) { console.error(e); }
    }, 1000);
}

function resetUI() {
    const startBtn = document.getElementById('start-eval');
    const clearBtn = document.getElementById('clear-eval');
    startBtn.disabled = false;
    startBtn.innerText = "开始测试";
    clearBtn.innerText = "重置";
}

function setStatusUI(msg, progress, stripe, color) {
    document.querySelector('.status-message').innerText = msg;
    const bar = document.querySelector('.progress-fill');
    bar.style.width = `${progress}%`;
    if(color) bar.style.backgroundColor = color;
    else bar.style.backgroundColor = '#0071e3'; 
    
    if(stripe) bar.classList.add('stripes-animation');
    else bar.classList.remove('stripes-animation');
}

function renderResults(metrics) {
    document.getElementById('results-section').style.display = 'block';
    document.getElementById('results-section').scrollIntoView({behavior:'smooth'});
    
    document.getElementById('accuracy').innerText = (metrics.accuracy != null ? metrics.accuracy : '--') + (String(metrics.accuracy).indexOf('%') >= 0 ? '' : '%');
    document.getElementById('f1-score').innerText = metrics.f1_score != null ? metrics.f1_score : '--';
    document.getElementById('test-cases').innerText = metrics.test_cases != null ? metrics.test_cases : '--';
    document.getElementById('context-len').innerText = metrics.context || "N/A";

    const hintEl = document.getElementById('results-placeholder-hint');
    if (hintEl) {
        const isPlaceholder = (String(metrics.test_cases) === '0' && (metrics.accuracy === '0.0' || metrics.accuracy === 0));
        hintEl.style.display = isPlaceholder ? 'block' : 'none';
        if (isPlaceholder && metrics.eval_note) {
            hintEl.textContent = metrics.eval_note;
        } else if (isPlaceholder) {
            hintEl.textContent = '当前为占位结果，未运行 lm_eval 或评测未返回数据。图中三角形仅为占位显示。';
        }
    }

    if (metrics.comparison && metrics.comparison.labels && (metrics.comparison.merged_data || metrics.comparison.base_data)) {
        drawChart(metrics.comparison, metrics.base_name || "Baseline");
    } else {
        if (window.myChart) { window.myChart.destroy(); window.myChart = null; }
    }
}

let myChart = null;
function drawChart(compData, baseName) {
    const ctx = document.getElementById('evaluation-chart');
    if (!ctx) return;
    const chartCtx = ctx.getContext('2d');
    if (myChart) myChart.destroy();

    let rawLabels = Array.isArray(compData.labels) ? compData.labels : ['Task'];
    let rawBase = (Array.isArray(compData.base_data) ? compData.base_data : []).map(Number);
    let rawMerged = (Array.isArray(compData.merged_data) ? compData.merged_data : []).map(Number);
    // 雷达图至少需要 3 个轴；单任务或双任务时用有意义标签与数据补齐，避免出现“—”和 0 的占位轴
    const minAxes = 3;
    const n = Math.max(rawLabels.length, minAxes);
    if (rawLabels.length >= 1 && rawLabels.length < minAxes) {
        const padLabels = rawLabels.length === 1 ? ['准确率', '得分'] : ['综合'];
        const padBase = rawLabels.length === 1 ? [rawBase[0] || 0, rawBase[0] || 0] : [(rawBase[0] + rawBase[1]) / 2];
        const padMerged = rawLabels.length === 1 ? [rawMerged[0] || 0, rawMerged[0] || 0] : [(rawMerged[0] + rawMerged[1]) / 2];
        rawLabels = [...rawLabels, ...padLabels];
        rawBase = [...rawBase, ...padBase];
        rawMerged = [...rawMerged, ...padMerged];
    }
    const labels = rawLabels.length >= n ? rawLabels.slice(0, n) : [...rawLabels];
    while (labels.length < n) labels.push('—');
    const baseData = rawBase.length >= n ? rawBase.slice(0, n) : [...rawBase];
    const mergedData = rawMerged.length >= n ? rawMerged.slice(0, n) : [...rawMerged];
    while (baseData.length < n) baseData.push(0);
    while (mergedData.length < n) mergedData.push(0);

    const allZero = (arr) => arr.length && arr.every((v) => v === 0);
    const baseAllZero = allZero(baseData);
    const mergedAllZero = allZero(mergedData);
    // 全为 0 时多边形会塌缩到中心不可见，用最小显示值画出小三角形
    const displayBase = baseAllZero ? baseData.map(() => 2) : baseData;
    const displayMerged = mergedAllZero ? mergedData.map(() => 2) : mergedData;

    myChart = new Chart(chartCtx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: baseName,
                    data: displayBase,
                    backgroundColor: 'rgba(108, 117, 125, 0.25)',
                    borderColor: 'rgba(108, 117, 125, 1)',
                    pointBackgroundColor: 'rgba(108, 117, 125, 1)',
                    pointBorderColor: '#fff',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    borderWidth: 2
                },
                {
                    label: 'Target Model',
                    data: displayMerged,
                    backgroundColor: 'rgba(67, 97, 238, 0.25)',
                    borderColor: '#4361ee',
                    pointBackgroundColor: '#4361ee',
                    pointBorderColor: '#fff',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800 },
            scales: {
                r: {
                    angleLines: { display: true },
                    suggestedMin: 0,
                    suggestedMax: 100,
                    pointLabels: { font: { size: 12 } }
                }
            },
            plugins: {
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const idx = context.dataIndex;
                            if (context.datasetIndex === 0) {
                                if (baseAllZero) return baseName + ': 0（无数据占位）';
                                return baseName + ': ' + displayBase[idx];
                            }
                            if (mergedAllZero) return 'Target Model: 0（无数据占位）';
                            return 'Target Model: ' + displayMerged[idx];
                        }
                    }
                }
            }
        }
    });
}

function setupSidebar() {
    const menuBtn = document.getElementById('menu-btn');
    const sidebar = document.getElementById('history-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const closeBtn = document.getElementById('close-sidebar');

    const toggle = () => {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('show');
    };

    menuBtn.addEventListener('click', toggle);
    closeBtn.addEventListener('click', toggle);
    overlay.addEventListener('click', toggle);
    
    loadHistoryList();
}

function initDropdowns() {
    const triggers = document.querySelectorAll('.ios-select-trigger');
    triggers.forEach(t => {
        t.addEventListener('click', (e) => {
            e.stopPropagation();
            const wrapper = t.closest('.ios-select-wrapper');
            const menu = wrapper.querySelector('.ios-select-options');
            
            document.querySelectorAll('.ios-select-options.open').forEach(el => {
                if(el !== menu) el.classList.remove('open');
            });
            menu.classList.toggle('open');
        });
    });
    
    document.querySelectorAll('.ios-select-options').forEach(menu => {
        menu.addEventListener('click', function(e) {
            const opt = e.target.closest('.ios-option');
            if (!opt) return;
            e.stopPropagation();
            const val = opt.dataset.value !== undefined ? opt.dataset.value : opt.textContent.trim();
            const nameEl = opt.querySelector('.opt-name');
            const name = nameEl ? nameEl.innerText : opt.textContent.trim();
            const wrapper = opt.closest('.ios-select-wrapper');
            if (!wrapper) return;
            const sel = wrapper.querySelector('.selected-text');
            const input = wrapper.querySelector('input[type="hidden"]');
            if (sel) sel.innerText = name;
            if (input) input.value = val;
            wrapper.querySelector('.ios-select-options').classList.remove('open');
            wrapper.querySelectorAll('.ios-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
        });
    });

    document.addEventListener('click', () => {
        document.querySelectorAll('.ios-select-options').forEach(el => {
            el.classList.remove('open');
        });
    });
}
