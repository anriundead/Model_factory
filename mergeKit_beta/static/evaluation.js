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
    setupSidebar(); 
    setupDeleteModal(); // 初始化删除弹窗监听
    initSegmentedControls();
}

function initSegmentedControls() {
    document.querySelectorAll('.segmented-control').forEach(control => {
        const options = control.querySelectorAll('.segmented-option');
        const hiddenInput = control.parentElement.querySelector('input[type="hidden"]');
        
        options.forEach(opt => {
            opt.addEventListener('click', () => {
                options.forEach(o => o.classList.remove('active', 'selected'));
                opt.classList.add('active', 'selected');
                if (hiddenInput) hiddenInput.value = opt.dataset.value;
            });
        });
    });
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
    
    const dataset = document.getElementById('eval-dataset').value;
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

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: evalState.selectedModel.path, 
                model_name: evalState.selectedModel.name,
                dataset: dataset,
                task_type: 'eval_only',
                limit: limit
            })
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
    
    document.getElementById('accuracy').innerText = `${metrics.accuracy}%`;
    document.getElementById('f1-score').innerText = metrics.f1_score;
    document.getElementById('test-cases').innerText = metrics.test_cases;
    document.getElementById('context-len').innerText = metrics.context || "N/A";

    drawChart(metrics.comparison, metrics.base_name || "Baseline");
}

let myChart = null;
function drawChart(compData, baseName) {
    const ctx = document.getElementById('evaluation-chart').getContext('2d');
    if (myChart) myChart.destroy();

    myChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: compData.labels,
            datasets: [
                {
                    label: baseName,
                    data: compData.base_data,
                    backgroundColor: 'rgba(108, 117, 125, 0.2)',
                    borderColor: 'rgba(108, 117, 125, 1)',
                    pointBackgroundColor: 'rgba(108, 117, 125, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Target Model',
                    data: compData.merged_data,
                    backgroundColor: 'rgba(67, 97, 238, 0.2)',
                    borderColor: '#4361ee',
                    pointBackgroundColor: '#4361ee',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { r: { angleLines: {display:true}, suggestedMin:0, suggestedMax:100 } }
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
    
    document.querySelectorAll('.ios-option').forEach(opt => {
        opt.addEventListener('click', function(e) {
            e.stopPropagation();
            const val = this.dataset.value;
            const name = this.querySelector('.opt-name').innerText;
            const wrapper = this.closest('.ios-select-wrapper');
            
            wrapper.querySelector('.selected-text').innerText = name;
            wrapper.querySelector('input').value = val;
            
            wrapper.querySelector('.ios-select-options').classList.remove('open');
            wrapper.querySelectorAll('.ios-option').forEach(o => o.classList.remove('selected'));
            this.classList.add('selected');
        });
    });

    document.addEventListener('click', () => {
        document.querySelectorAll('.ios-select-options').forEach(el => {
            el.classList.remove('open');
        });
    });
}
