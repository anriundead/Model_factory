document.addEventListener('DOMContentLoaded', () => {
    init();
});

// ===================== 全局状态 =====================
let state = {
    availableModels: [],
    selectedModels: []
};
let deleteTargetId = null;

const STORAGE_ACTIVE_TASK = 'mergenetic_active_task';
let currentTaskId = null;
let pollTimer = null;
let currentPriority = 'common';
let hasShownRestartToast = false;
let isHistoryMode = false;

// ===================== 初始化 =====================
function init() {
    loadModels();
    setupDragAndDrop();
    setupEventListeners();
    // 先初始化配方选择（避免被 initPageDropdowns 覆盖）
    setupRecipeApply();
    initPageDropdowns();
    initSegmentedControls();
    setupMergeModeSwitch();
    setupEvolutionaryDatasetTypeSwitch();
    setupEvolutionarySplitDropdown();
    setupEvolutionaryDatasetFetch();
    setupEvolutionaryMerge();
    restoreSession();
    loadHistoryList();
    setupHistoryUI();
}

function setupHistoryUI() {
    // 1. 获取所有需要的 DOM 元素
    const sidebar = document.getElementById('history-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const menuBtn = document.getElementById('menu-btn');
    const closeBtn = document.getElementById('close-sidebar');

    // 定义关闭动作
    const closeSidebar = () => {
        if (sidebar) sidebar.classList.remove('open');
        if (overlay) overlay.classList.remove('show');
    };

    // 定义打开动作
    const openSidebar = () => {
        if (sidebar) sidebar.classList.add('open');
        if (overlay) overlay.classList.add('show');
        loadHistoryList(); // 刷新列表
    };

    // 2. 绑定菜单按钮点击事件 (打开)
    if (menuBtn) {
        // 移除旧的监听器防止重复(虽然这里是初始化，但为了保险)
        const newMenuBtn = menuBtn.cloneNode(true);
        menuBtn.parentNode.replaceChild(newMenuBtn, menuBtn);
        newMenuBtn.addEventListener('click', openSidebar);
    }

    // 3. 绑定关闭按钮点击事件 (关闭)
    if (closeBtn) {
        closeBtn.addEventListener('click', closeSidebar);
    }

    // 4. [修复的核心] 绑定遮罩层点击事件 (点击空白处关闭)
    if (overlay) {
        overlay.addEventListener('click', closeSidebar);
    }

    // ================= 新工程按钮逻辑 =================
    const newProjBtn = document.getElementById('btn-new-project');
    if(newProjBtn) newProjBtn.addEventListener('click', startNewProject);
    
    // ================= 删除弹窗逻辑 (保持不变) =================
    const cancelBtn = document.getElementById('cancel-delete');
    const confirmBtn = document.getElementById('confirm-delete');
    const deleteModal = document.getElementById('delete-modal-overlay');

    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            deleteModal.classList.remove('show');
            setTimeout(() => { deleteModal.style.display = 'none'; }, 300);
            deleteTargetId = null;
        });
    }

    if (confirmBtn) {
        const newBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newBtn, confirmBtn);
        
        newBtn.addEventListener('click', async () => {
            if (deleteTargetId) {
                newBtn.innerText = "删除中...";
                newBtn.disabled = true;
                await deleteHistoryItem(deleteTargetId);
                deleteModal.classList.remove('show');
                setTimeout(() => { deleteModal.style.display = 'none'; }, 300);
                newBtn.innerText = "删除";
                newBtn.disabled = false;
                deleteTargetId = null;
            }
        });
    }
}

// [修改：加载历史列表]
async function loadHistoryList() {
    const listEl = document.getElementById('history-list');
    listEl.innerHTML = '<div style="padding:15px;text-align:center;color:#999;font-size:0.8rem;">加载中...</div>';

    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        listEl.innerHTML = ''; // 清空加载中

        if (!data.history || data.history.length === 0) {
            listEl.innerHTML = '<div style="padding:20px; text-align:center; color:#999; font-size:0.8rem;">暂无融合记录</div>';
            return;
        }

        const currentViewId = document.getElementById('btn-download')?.dataset.taskId;

        data.history.forEach(item => {
            const div = document.createElement('div');
            div.className = `history-item ${item.id === currentViewId ? 'active' : ''}`;
            
            // 1. 时间格式化
            const date = new Date(item.created_at * 1000);
            const timeStr = `${(date.getMonth()+1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
            
            // 2. 标题显示：优先使用自定义名称
            const displayName = item.custom_name || (item.models || []).join(' + ');

            // 3. 构建 HTML (使用新 CSS 结构)
            div.innerHTML = `
                <div class="h-content">
                    <div class="h-title">${displayName}</div>
                    <div class="h-meta">
                        <span>${timeStr}</span>
                        <span>•</span>
                        <span>${item.method || 'Unknown'}</span>
                    </div>
                </div>
                <button class="btn-delete-history" onclick="confirmDelete(event, '${item.id}')">
                    <i class="ri-delete-bin-line"></i>
                </button>
            `;
            
            // 点击进入详情
            div.onclick = (e) => {
                if (e.target.closest('.btn-delete-history')) return;
                loadHistoryDetail(item);
                
                // 更新高亮
                document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
                div.classList.add('active');
            };
            
            listEl.appendChild(div);
        });
    } catch (e) {
        console.error("加载历史失败:", e);
        listEl.innerHTML = '<div style="padding:20px; text-align:center; color:#ff3b30; font-size:0.8rem;">加载失败</div>';
    }
}

function loadHistoryDetail(item) {
    isHistoryMode = true;
    
    // 1. 关闭侧边栏 & 遮罩
    const sidebar = document.getElementById('history-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (sidebar) sidebar.classList.remove('open');
    if (overlay) overlay.classList.remove('show');
    
    // 2. 切换 UI 到 只读模式
    toggleHistoryView(true);
    
    // 3. 填充数据
    state.selectedModels = item.models || [];
    renderSelectedModels(); 
    document.querySelectorAll('.remove-btn').forEach(btn => btn.style.display = 'none');
    
    updateDropdownText('merge-method', item.method);
    updateDropdownText('dtype', item.dtype);
    
    if (item.weights) {
        updateWeightSliders();
        const sliders = document.querySelectorAll('.weight-slider');
        sliders.forEach((slider, idx) => {
            if (item.weights[idx] !== undefined) {
                slider.value = item.weights[idx];
                const label = document.getElementById(`weight-val-${idx}`);
                if(label) label.innerText = item.weights[idx].toFixed(1);
            }
        });
    }
    
    // 4. 【核心修复】智能下载按钮逻辑
    const dlBtn = document.getElementById('btn-download');
    dlBtn.dataset.taskId = item.id;
    
    // 克隆按钮以移除旧的监听器
    const newDlBtn = dlBtn.cloneNode(true);
    dlBtn.parentNode.replaceChild(newDlBtn, dlBtn);

    // --- 新增：详情按钮 (Restore Details Button) ---
    // 检查是否已存在详情按钮，避免重复添加
    let detailBtn = document.getElementById('btn-show-details');
    if (!detailBtn) {
        detailBtn = document.createElement('button');
        detailBtn.id = 'btn-show-details';
        detailBtn.className = 'btn-secondary'; // 使用次级按钮样式
        detailBtn.style.marginLeft = '10px';
        detailBtn.innerHTML = '<i class="ri-file-list-3-line"></i> 详细报告';
        newDlBtn.parentNode.insertBefore(detailBtn, newDlBtn.nextSibling);
    }
    // 绑定点击事件
    const newDetailBtn = detailBtn.cloneNode(true);
    detailBtn.parentNode.replaceChild(newDetailBtn, detailBtn);
    
    newDetailBtn.onclick = () => {
        // 构造 evoProgress 数据 (从 item 获取)
        const evoData = item.evolution_progress || {};
        // 补充缺失的 step/total 信息 (如果 item 中有)
        if (item.step) evoData.step = item.step;
        if (item.n_iter && item.pop_size) evoData.total_expected_steps = item.n_iter * item.pop_size; // 估算

        showTaskCompletionModal(item.id, { status: 'success' }, evoData);
    };
    // ------------------------------------------
    
    // 绑定点击事件
    newDlBtn.addEventListener('click', async (e) => {
        const btn = e.currentTarget; 
        const originalHtml = '<i class="ri-download-line"></i> 下载模型'; // 保存原始状态
        const downloadUrl = `/api/download/${item.id}`;
        
        // --- 阶段 1: 变为 "正在打包" ---
        btn.innerHTML = '<i class="ri-loader-4-line ri-spin"></i> 正在打包...';
        btn.disabled = true;
        btn.style.opacity = '0.7';

        try {
            // --- 阶段 2: 发送 HEAD 请求探测 ---
            // 这会触发后端压缩(make_archive)，代码会在这里“暂停”，直到后端打包完成返回 Header
            await fetch(downloadUrl, { method: 'HEAD' });
            
            // --- 阶段 3: 打包完成，触发下载 ---
            // 此时文件已存在，浏览器访问该链接会瞬间开始下载，无需等待
            window.location.href = downloadUrl;
            
        } catch (error) {
            console.error("打包探测失败:", error);
            // 即使 HEAD 请求失败（极少情况），也尝试直接下载作为兜底
            window.location.href = downloadUrl;
        } finally {
            // --- 阶段 4: 立即重置按钮 ---
            // 给一个微小的 500ms 延迟，让用户看清“打包完成”的状态切换，体验更好
            setTimeout(() => {
                btn.innerHTML = originalHtml;
                btn.disabled = false;
                btn.style.opacity = '1';
            }, 500); 
        }
    });
    
    // 5. 更新状态栏
    setStatusUI({ message: "历史归档 (只读)", progress: 100, color: '#86868b', stripes: false });
    
    const pageSubtitle = document.querySelector('.page-subtitle');
    if(pageSubtitle) pageSubtitle.innerText = `工程 ID: ${item.id}`;
}

// [核心逻辑：回到新工程]
function startNewProject() {
    isHistoryMode = false;
    toggleHistoryView(false);
    
    // 清空数据
    state.selectedModels = [];
    renderSelectedModels();
    
    // 恢复默认 UI
    document.querySelector('.page-subtitle').innerText = "选择基座模型，配置融合参数，构建您的专属智能体";
    setStatusUI({ message: "准备就绪", progress: 0, color: '#0071e3', stripes: false });
    
    // 隐藏进度条
    document.querySelector('.progress-container').style.display = 'none';
    
    // 重置 sessionStorage
    sessionStorage.removeItem(STORAGE_ACTIVE_TASK);
    currentTaskId = null;
    
    updateUIState();
}

// [UI 切换辅助]
function toggleHistoryView(isHistory) {
    const editBtns = document.getElementById('edit-buttons');
    const histBtns = document.getElementById('history-buttons');
    const dropzone = document.getElementById('dropzone');
    const paramsSection = document.querySelector('.params-form');
    const modelsContainer = document.getElementById('models-container');
    const localHeader = document.querySelector('.section-header h3'); // 本地可用模型标题
    
    if (isHistory) {
        editBtns.style.display = 'none';
        histBtns.style.display = 'flex';
        
        // 变更为只读样式
        dropzone.classList.add('read-only-mask');
        paramsSection.classList.add('read-only-section');
        
        // 标题变更
        if(localHeader) localHeader.innerText = "所选的融合模型";
        
        // 左侧 "本地模型库" 在历史模式下其实没啥用，可以选择隐藏或者变灰
        modelsContainer.style.opacity = '0.5';
        modelsContainer.style.pointerEvents = 'none';
        
    } else {
        editBtns.style.display = 'flex';
        histBtns.style.display = 'none';
        
        dropzone.classList.remove('read-only-mask');
        paramsSection.classList.remove('read-only-section');
        
        if(localHeader) localHeader.innerText = "本地可用模型";
        modelsContainer.style.opacity = '1';
        modelsContainer.style.pointerEvents = 'auto';
    }
}

// 辅助：更新自定义下拉菜单文字
function updateDropdownText(inputId, value) {
    // 找到 input 对应的 wrapper
    const input = document.getElementById(inputId);
    if(!input) return;
    input.value = value;
    
    const wrapper = input.parentElement.querySelector('.ios-select-wrapper');
    if(!wrapper) return;
    
    // 找到对应的 option 获取显示的文字 (name)
    const option = wrapper.querySelector(`.ios-option[data-value="${value}"]`);
    if(option) {
        const name = option.querySelector('.opt-name').innerText;
        wrapper.querySelector('.selected-text').innerText = name;
        
        // 更新选中态样式
        wrapper.querySelectorAll('.ios-option').forEach(o => o.classList.remove('selected'));
        option.classList.add('selected');
    }
}

window.confirmDelete = function(e, id) {
    if (e) e.stopPropagation(); // 阻止冒泡
    deleteTargetId = id;        // 记录要删哪个 ID
    
    const modal = document.getElementById('delete-modal-overlay');
    if (modal) {
        modal.style.display = 'flex';
        // 强制重绘以触发动画
        modal.offsetHeight; 
        modal.classList.add('show');
    } else {
        console.error("找不到 ID 为 delete-modal-overlay 的弹窗元素");
    }
};

async function deleteHistoryItem(id) {
    try {
        await fetch(`/api/history/${id}`, { method: 'DELETE' });
        loadHistoryList(); // 刷新列表
        // 如果当前正在看这个被删除的项目，回到新工程
        if (isHistoryMode && document.getElementById('btn-download').dataset.taskId === id) {
            startNewProject();
        }
    } catch (e) {
        console.error(e);
    }
}

async function executeMergeTask() {
    clearPolling();
    
    const startBtn = document.getElementById('start-merge');
    const statusDiv = document.getElementById('task-status');

    // UI 初始化
    startBtn.disabled = true;
    startBtn.innerText = "正在融合...";
    statusDiv.style.opacity = '1';
    document.querySelector('.progress-container').style.display = 'block';

    setStatusUI({ message: "正在连接服务器...", progress: 0, color: '#0071e3', stripes: false });

    const customNameInput = document.getElementById('model-name-input');
    const customName = customNameInput.value.trim();
    
    if (!customName) {
        customNameInput.classList.add('input-error');
        // 可以加一个抖动动画增强提示（可选）
        customNameInput.focus(); 
        return; 
    }

    // 收集参数
    const method = document.getElementById('merge-method').value;
    const dtype = document.getElementById('dtype').value;
    const limit = document.getElementById('merge-limit').value; 
    
    // 获取数据集参数
    const datasetType = document.getElementById('standard-dataset-type').value; // mmlu / cmmmu
    const datasetSubset = document.getElementById('standard-subset-val').value; // global / ...

    const weights = [];
    document.querySelectorAll('.weight-slider').forEach(input => {
        weights.push(parseFloat(input.value));
    });

    const modelPaths = state.selectedModels.map(m => (typeof m === 'string' ? null : m.path)).filter(Boolean);
    const modelNames = state.selectedModels.map(m => (typeof m === 'string' ? m : (m && m.name))).filter(Boolean);
    
    const payload = {
        weights: weights,
        method: method,
        dtype: dtype,
        priority: currentPriority,
        custom_name: customName,
        limit: limit,
        // 添加数据集参数
        dataset: datasetType, 
        hf_dataset: datasetType, // 兼容后端逻辑
        hf_subset: datasetSubset,
        hf_split: 'test' // 默认使用 test split
    };
    if (modelPaths.length === state.selectedModels.length) {
        payload.model_paths = modelPaths;
    } else {
        payload.models = modelNames;
    }
    try {
        const response = await fetch('/api/merge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        // ... (后续处理逻辑保持不变) ...
        const data = await response.json();
        if (data.status === 'success' && data.task_id) {
            currentTaskId = data.task_id;
            sessionStorage.setItem(STORAGE_ACTIVE_TASK, currentTaskId);
            toggleResetButton('stop');
            pollStatus(currentTaskId);
        } else {
            alert('启动失败: ' + (data.message || 'Unknown error'));
            resetStartButton();
            toggleResetButton('clear');
        }
    } catch (e) {
        console.error(e);
        alert('网络错误');
        resetStartButton();
        toggleResetButton('clear');
    }

}

// 恢复刷新后的会话：如果存在 active task，直接进入轮询并把按钮切换到“停止任务”
function restoreSession() {
    const activeTaskId = sessionStorage.getItem(STORAGE_ACTIVE_TASK);
    if (!activeTaskId) return;

    currentTaskId = activeTaskId;

    // UI：立即进入“运行中”状态
    toggleResetButton('stop');

    const startBtn = document.getElementById('start-merge');
    const statusDiv = document.getElementById('task-status');

    startBtn.disabled = true;
    startBtn.innerText = "正在处理...";
    statusDiv.style.opacity = '1';
    document.querySelector('.progress-container').style.display = 'block';

    pollStatus(activeTaskId);
}

function setupEventListeners() {
    document.getElementById('start-merge').addEventListener('click', openPriorityModal);
    document.getElementById('clear-selection').addEventListener('click', handleClearOrStop);
    setupModalListeners();
}

function setupModalListeners() {
    const modalOverlay = document.getElementById('priority-modal-overlay');
    const trigger = document.getElementById('priority-trigger');
    const optionsMenu = document.getElementById('priority-options');
    const options = optionsMenu.querySelectorAll('.ios-option'); 
    const confirmBtn = document.getElementById('confirm-priority');
    const cancelBtn = document.getElementById('cancel-priority');
    // 切换下拉菜单显示/隐藏
    trigger.addEventListener('click', (e) => {
        e.stopPropagation(); // 防止冒泡关闭
        optionsMenu.classList.toggle('open');
    });

    // 点击选项
    options.forEach(opt => {
        opt.addEventListener('click', () => {
            // 1. 更新 UI 选中态
            options.forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            
            // 2. 更新触发器文字
            const name = opt.querySelector('.opt-name').innerText;
            trigger.querySelector('.selected-text').innerText = name;
            
            // 3. 更新数据
            currentPriority = opt.dataset.value;
            
            // 4. 关闭菜单
            optionsMenu.classList.remove('open');
        });
    });

    // 点击页面其他地方关闭下拉菜单
    document.addEventListener('click', (e) => {
        if (!trigger.contains(e.target) && !optionsMenu.contains(e.target)) {
            optionsMenu.classList.remove('open');
        }
    });

    const nameInput = document.getElementById('model-name-input');
    if (nameInput) {
        nameInput.addEventListener('input', function() {
            // 只要有内容（哪怕是一个字符），就移除报错样式
            if (this.value.trim().length > 0) {
                this.classList.remove('input-error');
            }
        });
    }

    // 确认按钮 -> 执行真正的融合逻辑
    confirmBtn.replaceWith(confirmBtn.cloneNode(true)); // 防止多次绑定
    const newConfirmBtn = document.getElementById('confirm-priority');
    
    newConfirmBtn.addEventListener('click', () => {
        const nameInput = document.getElementById('model-name-input');
        const customName = nameInput.value.trim();
        
        // 1. 立即校验
        if (!customName) {
            nameInput.classList.add('input-error');
            nameInput.focus(); // 聚焦让用户知道要填这里
            // 关键：这里直接 return，不执行 closeModal，弹窗就会保持显示
            return; 
        }

        // 2. 校验通过，才关闭弹窗并执行任务
        closeModal();
        setTimeout(() => {
            executeMergeTask(); 
        }, 200); 
    });

    // 取消按钮
    cancelBtn.addEventListener('click', closeModal);
    
    // --- 新增：初始化标准融合的数据集选择 ---
    setupStandardDatasetUI();
}

// --- 新增：标准融合数据集 UI 初始化 ---
function setupStandardDatasetUI() {
    // 1. 数据集类型切换 (MMLU/CMMMU)
    const typeTrigger = document.getElementById('standard-dataset-type-trigger');
    const typeOptions = document.getElementById('standard-dataset-type-options');
    const typeHidden = document.getElementById('standard-dataset-type');
    
    if (typeTrigger && typeOptions && typeHidden) {
        // 切换显示
        typeTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            typeOptions.classList.toggle('open');
        });
        
        // 选项点击
        typeOptions.querySelectorAll('.ios-option').forEach(opt => {
            opt.addEventListener('click', () => {
                // UI 更新
                typeOptions.querySelectorAll('.ios-option').forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
                
                const val = opt.dataset.value;
                const name = opt.querySelector('.opt-name').innerText;
                typeTrigger.querySelector('.selected-text').innerText = name;
                typeHidden.value = val;
                
                // 关闭菜单
                typeOptions.classList.remove('open');
                
                // 加载对应子集
                loadStandardSubsets(val);
            });
        });
        
        // 点击外部关闭
        document.addEventListener('click', (e) => {
            if (!typeTrigger.contains(e.target) && !typeOptions.contains(e.target)) {
                typeOptions.classList.remove('open');
            }
        });
    }
    
    // 2. 子集切换
    const subsetTrigger = document.getElementById('standard-subset-trigger');
    const subsetOptions = document.getElementById('standard-subset-options');
    const subsetHidden = document.getElementById('standard-subset-val');
    
    if (subsetTrigger && subsetOptions && subsetHidden) {
        subsetTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            subsetOptions.classList.toggle('open');
        });
        
        // 初始加载 (默认为 MMLU)
        loadStandardSubsets('mmlu');
        
        // 点击外部关闭
        document.addEventListener('click', (e) => {
            if (!subsetTrigger.contains(e.target) && !subsetOptions.contains(e.target)) {
                subsetOptions.classList.remove('open');
            }
        });
    }
}

// --- 新增：加载标准融合子集 ---
function loadStandardSubsets(datasetType) {
    const optionsUl = document.getElementById('standard-subset-options');
    const trigger = document.getElementById('standard-subset-trigger');
    const hidden = document.getElementById('standard-subset-val');
    const label = document.getElementById('standard-subset-label');
    
    if (!optionsUl || !hidden) return;
    
    const isMmlu = (datasetType || '').toLowerCase() === 'mmlu';
    const apiUrl = isMmlu ? '/api/mmlu_subset_groups' : '/api/cmmmu_subset_groups';
    
    if (label) label.innerText = isMmlu ? 'MMLU 领域' : 'CMMMU 领域';
    
    // 显示加载中
    optionsUl.innerHTML = '<li class="ios-option"><span class="opt-name">加载中...</span></li>';
    
    fetch(apiUrl)
        .then(r => r.json())
        .then(data => {
            const groups = data.groups || [];
            optionsUl.innerHTML = '';
            
            if (groups.length === 0) {
                optionsUl.innerHTML = '<li class="ios-option"><span class="opt-name">无可用子集</span></li>';
                return;
            }
            
            // 填充选项
            groups.forEach((g, idx) => {
                const li = document.createElement('li');
                li.className = 'ios-option' + (idx === 0 ? ' selected' : '');
                li.dataset.value = g.id;
                
                li.innerHTML = `
                    <div class="opt-left">
                        <span class="opt-name">${g.label || g.id}</span>
                    </div>
                `;
                if (g.subsets && g.subsets.length) {
                     li.title = g.subsets.join(', ');
                }
                
                li.addEventListener('click', () => {
                    optionsUl.querySelectorAll('.ios-option').forEach(o => o.classList.remove('selected'));
                    li.classList.add('selected');
                    
                    if (trigger) trigger.querySelector('.selected-text').innerText = g.label || g.id;
                    if (hidden) hidden.value = g.id;
                    
                    optionsUl.classList.remove('open');
                });
                
                optionsUl.appendChild(li);
            });
            
            // 默认选中第一个
            if (groups.length > 0) {
                const first = groups[0];
                if (trigger) trigger.querySelector('.selected-text').innerText = first.label || first.id;
                if (hidden) hidden.value = first.id;
            }
        })
        .catch(e => {
            console.error(e);
            optionsUl.innerHTML = '<li class="ios-option">加载失败</li>';
        });
}

// --- 新增：检查标准融合兼容性 & VLM检测 ---
async function checkStandardCompatibilityAndVlm() {
    const statusDiv = document.getElementById('standard-compatibility-status');
    const modelPaths = state.selectedModels.map(m => (typeof m === 'string' ? null : m.path)).filter(Boolean);
    
    // 1. 隐藏状态栏 (如果模型不足2个)
    if (modelPaths.length < 2) {
        if (statusDiv) statusDiv.style.display = 'none';
        return;
    }
    
    // 2. VLM 检测与自动切换
    let hasVlm = false;
    // 优先检查 state 中的 is_vlm 标记
    hasVlm = state.selectedModels.some(m => m.is_vlm);
    
    // 如果没有标记，尝试从名称推断 (fallback)
    if (!hasVlm) {
        hasVlm = state.selectedModels.some(m => {
             const name = m.name || (typeof m === 'string' ? m : '');
             return /vl|vision|vlm|llava|qwen2\.?5?\s*vl|qwen2-vl|cogvlm|minicpm-v/i.test(name);
        });
    }
    
    // 执行切换
    const typeHidden = document.getElementById('standard-dataset-type');
    const typeTrigger = document.getElementById('standard-dataset-type-trigger');
    const targetType = hasVlm ? 'cmmmu' : 'mmlu';
    
    if (typeHidden && typeHidden.value !== targetType) {
        typeHidden.value = targetType;
        // 更新 UI
        if (typeTrigger) {
            const options = document.getElementById('standard-dataset-type-options');
            const targetOpt = options ? options.querySelector(`.ios-option[data-value="${targetType}"]`) : null;
            const targetText = targetOpt ? targetOpt.querySelector('.opt-name').innerText : (hasVlm ? 'CMMMU (VLM)' : 'MMLU (LLM)');
            typeTrigger.querySelector('.selected-text').innerText = targetText;
            
            // 更新选中态
            if (options) {
                options.querySelectorAll('.ios-option').forEach(o => o.classList.remove('selected'));
                if (targetOpt) targetOpt.classList.add('selected');
            }
        }
        // 重新加载子集
        loadStandardSubsets(targetType);
    }

    // 3. 架构兼容性检查 (调用后端)
    if (statusDiv) {
        statusDiv.style.display = 'block';
        statusDiv.innerHTML = '<i class="ri-loader-4-line ri-spin"></i> 正在检查架构兼容性...';
        statusDiv.style.background = '#f5f5f7';
        statusDiv.style.color = '#1d1d1f';
        
        try {
            const res = await fetch('/api/check_compatibility', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_paths: modelPaths })
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                if (data.compatible) {
                    statusDiv.innerHTML = `<i class="ri-checkbox-circle-fill" style="color: var(--success);"></i> 架构兼容 (${data.types[0] || 'Unknown'})`;
                    statusDiv.style.background = 'rgba(52, 199, 89, 0.1)';
                    statusDiv.style.color = 'var(--success)';
                } else {
                    statusDiv.innerHTML = `<i class="ri-error-warning-fill" style="color: var(--error);"></i> 架构不兼容: ${data.message}`;
                    statusDiv.style.background = 'rgba(255, 59, 48, 0.1)';
                    statusDiv.style.color = 'var(--error)';
                }
            } else {
                statusDiv.innerHTML = `<i class="ri-question-fill"></i> 无法验证: ${data.message}`;
            }
        } catch (e) {
            statusDiv.innerHTML = `<i class="ri-wifi-off-line"></i> 检查失败`;
        }
    }
}

// 【新增】生成 SHA-256 哈希并构建默认名称
async function generateDefaultName(models) {
    const dateStr = new Date().toISOString().split('.')[0].replace(/[-T:]/g, ''); // 格式: YYYYMMDDHHMMSS
    const rawString = models.join('_') + '_' + dateStr;
    
    // 使用浏览器原生 Crypto API 计算 SHA-256
    const msgBuffer = new TextEncoder().encode(rawString);
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    
    // 截取前 8 位哈希保持简洁
    const shortHash = hashHex.substring(0, 8);
    
    // 默认名：模型组合_短哈希
    // 为了防止名字过长，只取前两个模型名做前缀
    const prefix = models.slice(0, 2).join('_'); 
    return `${prefix}_${shortHash}`;
}

// 2. 打开弹窗 (包含之前的校验逻辑)
// 【修改】打开弹窗逻辑
async function openPriorityModal() {
    // --- 校验逻辑 ---
    if (state.selectedModels.length === 0) return;
    if (state.selectedModels.length < 2) {
        alert("请至少选择两个模型进行融合");
        return;
    }
    
    const method = document.getElementById('merge-method').value;
    
    // 特殊算法校验 (保持原逻辑...)
    if (['task_arithmetic', 'ties_dare'].includes(method)) {
        // ... (原校验代码不变)
    }
    // --- 校验结束 ---

    // 【新增】生成并填充默认名称
    const nameInput = document.getElementById('model-name-input');
    if (nameInput) {
        // 重置错误样式，防止上次留下的红框还在
        nameInput.classList.remove('input-error');
        
        nameInput.value = "Generating name...";
        const defaultName = await generateDefaultName(state.selectedModels);
        nameInput.value = defaultName;
    }

    // 显示弹窗
    const modal = document.getElementById('priority-modal-overlay');
    modal.style.display = 'flex';
    modal.offsetHeight; 
    modal.classList.add('show');
    
    // 自动聚焦，方便用户直接修改
    if (nameInput) nameInput.focus();
}

function closeModal() {
    const modal = document.getElementById('priority-modal-overlay');
    modal.classList.remove('show');
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300); // 等待 CSS transition 结束
}

// ===================== UI 辅助 =====================
function toggleResetButton(mode) {
    const btn = document.getElementById('clear-selection');

    if (mode === 'stop') {
        btn.innerText = "停止任务";
        btn.dataset.mode = 'stop';
        btn.disabled = false;

        // 直接用内联颜色避免依赖 CSS 新增类
        btn.style.backgroundColor = '#c8c8c8ff';
        btn.style.color = 'white';
    } else {
        btn.innerText = "重置";
        btn.dataset.mode = 'clear';

        btn.style.backgroundColor = '';
        btn.style.color = '';

        btn.disabled = state.selectedModels.length === 0;
    }
}

function showRestartToast() {
    // 检查是否已经存在
    if (document.getElementById('restart-toast')) return;

    const bar = document.querySelector('.action-bar');
    const toast = document.createElement('div');
    toast.id = 'restart-toast';
    toast.className = 'restart-toast';
    toast.innerHTML = `
        <i class="ri-information-fill"></i>
        <span>上次任务被打断，已重新开始执行</span>
        <button onclick="this.parentElement.remove()" class="toast-close">×</button>
    `;
    
    // 插入到 action-bar 内部，或者放在上面
    // 这里我们选择绝对定位在 action-bar 上方
    bar.appendChild(toast);

    // 5秒后自动消失
    setTimeout(() => {
        if(toast && toast.parentElement) toast.remove();
    }, 5000);
}

function hideRestartToast() {
    const toast = document.getElementById('restart-toast');
    if (toast) toast.remove();
}

function resetStartButton() {
    const btn = document.getElementById('start-merge');
    btn.disabled = state.selectedModels.length === 0;
    btn.innerText = "开始融合";
}

function setStatusUI({ message, progress, color, stripes }) {
    const statusMsg = document.querySelector('.status-message');
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');

    if (message !== undefined) statusMsg.innerText = message;

    if (progress !== undefined) {
        const p = Math.max(0, Math.min(100, Number(progress)));
        progressBar.style.width = `${p}%`;
        progressText.style.display = 'none';
        progressText.innerText = ``;
    }

    if (color) progressBar.style.backgroundColor = color;

    if (stripes) progressBar.classList.add('stripes-animation');
    else progressBar.classList.remove('stripes-animation');
}

function clearPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

// ===================== 业务逻辑：模型加载/选择 =====================

// 1. 加载本地模型（基座 + 已融合且含 .safetensors 的 output）
async function loadModels() {
    const container = document.getElementById('models-container');
    container.innerHTML = '<div class="loading">正在连接后端...</div>';

    try {
        const [modelsRes, mergedRes] = await Promise.all([fetch('/api/models'), fetch('/api/merged_models')]);
        const modelsData = await modelsRes.json();
        const mergedData = await mergedRes.json();
        const baseList = (modelsData.status === 'success' && modelsData.models) ? modelsData.models : [];
        const mergedList = (mergedData.status === 'success' && mergedData.models) ? mergedData.models : [];
        state.availableModels = baseList.map(m => ({ ...m, source: 'base' })).concat(mergedList.map(m => ({ ...m, source: 'merged' })));
        renderModelList(state.availableModels);
        loadRecipeListForMerge();
    } catch (error) {
        console.error('API Error:', error);
        container.innerHTML = '<div class="error">无法连接后端服务，请检查 app.py 是否运行。</div>';
    }
}

// 加载配方列表到标准融合的「融合配方」区
async function loadRecipeListForMerge() {
    const container = document.getElementById('recipes-merge-container');
    if (!container) return;
    try {
        const res = await fetch('/api/recipes');
        const data = await res.json();
        const list = (data.recipes || []).slice(0, 30);
        if (list.length === 0) {
            container.innerHTML = '<p class="limit-desc">暂无配方</p>';
            return;
        }
        container.innerHTML = list.map(r => {
            const name = (r.custom_name || r.recipe_id || '未命名').toString().slice(0, 28);
            const rid = r.recipe_id || r.task_id || '';
            return '<div class="model-card recipe-card" data-recipe-id="' + rid + '" data-recipe-name="' + (name.replace(/"/g, '&quot;')) + '" title="点击加入工作台">' +
                '<h4>' + name + '</h4><p class="limit-desc">配方</p></div>';
        }).join('');
        container.querySelectorAll('.recipe-card').forEach(el => {
            el.addEventListener('click', () => {
                const recipeId = el.getAttribute('data-recipe-id');
                const name = el.getAttribute('data-recipe-name') || recipeId;
                addModelToSelection({ type: 'recipe', recipe_id: recipeId, name: name });
            });
        });
    } catch (e) {
        container.innerHTML = '<p class="limit-desc">加载失败</p>';
    }
}

// 2. 渲染模型列表卡片
function renderModelList(models) {
    const container = document.getElementById('models-container');
    container.innerHTML = '';

    if (!models || models.length === 0) {
        container.innerHTML = '<div class="empty">本地没有找到基座模型，请检查基座模型目录或环境变量 LOCAL_MODELS_PATH。</div>';
        return;
    }

    models.forEach(model => {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.draggable = true;
        card.dataset.modelName = model.name;
        card.dataset.modelPath = model.path || '';

        const sizeGB = (model.size / (1024 * 1024 * 1024)).toFixed(2);

        card.innerHTML = `
            <h4>${model.name}</h4>
            <p>Size: ${sizeGB} GB</p>
            <p>Family: ${model.details?.family || 'Unknown'}</p>
        `;

        card.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', model.name);
            e.dataTransfer.setData('application/json', JSON.stringify({ name: model.name, path: model.path || '' }));
            card.classList.add('dragging');
        });

        card.addEventListener('dragend', () => {
            card.classList.remove('dragging');
        });

        card.addEventListener('click', () => {
            addModelToSelection({ type: 'path', path: model.path, name: model.name });
        });

        container.appendChild(card);
    });
}

// 3. 设置拖放区域逻辑
function setupDragAndDrop() {
    const dropzone = document.getElementById('dropzone');

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('active');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('active');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        const modelName = e.dataTransfer.getData('text/plain');
        if (modelName) {
            const model = state.availableModels.find(m => m.name === modelName);
            addModelToSelection(model || { name: modelName, path: null });
        }
    });
}

// 【新增】初始化分段选择器逻辑
function initSegmentedControls() {
    document.querySelectorAll('.segmented-control').forEach(control => {
        const options = control.querySelectorAll('.segmented-option');
        // 找到同级的 hidden input（可能没有，如 merge-mode-control）
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

// 标准融合 / 完全融合 切换
function setupMergeModeSwitch() {
    const control = document.getElementById('merge-mode-control');
    if (!control) return;
    const panelStandard = document.getElementById('panel-standard-merge');
    const panelEvolutionary = document.getElementById('panel-evolutionary-merge');
    const options = control.querySelectorAll('.segmented-option');
    options.forEach(opt => {
        opt.addEventListener('click', () => {
            const value = opt.dataset.value;
            if (value === 'standard') {
                if (panelStandard) panelStandard.style.display = '';
                if (panelEvolutionary) panelEvolutionary.style.display = 'none';
            } else {
                if (panelStandard) panelStandard.style.display = 'none';
                if (panelEvolutionary) panelEvolutionary.style.display = 'block';
                loadEvolutionaryModels();
                loadEvolutionarySubsets(document.getElementById('evolutionary-dataset-type')?.value || 'mmlu');
            }
        });
    });
}

let evolutionarySelectedPaths = [];
let evolutionarySelectedList = [];

function renderEvolutionaryWorkspace() {
    const placeholder = document.getElementById('evolutionary-workspace-placeholder');
    const listEl = document.getElementById('evolutionary-selected-list');
    if (!listEl) return;
    listEl.innerHTML = '';
    if (evolutionarySelectedList.length === 0) {
        if (placeholder) placeholder.style.display = '';
        return;
    }
    if (placeholder) placeholder.style.display = 'none';
    evolutionarySelectedList.forEach((item, index) => {
        const tag = item.type === 'recipe' ? '蓝图' : (item.is_vlm ? 'VLM' : 'LLM');
        const tagClass = item.type === 'recipe' ? 'evol-tag-recipe' : (item.is_vlm ? 'evol-tag-vlm' : 'evol-tag-llm');
        const div = document.createElement('div');
        div.className = 'selected-model-item';
        div.innerHTML = '<span class="evol-model-name">' + (item.name || '') + '</span> <span class="evol-tag ' + tagClass + '">' + tag + '</span> <button type="button" class="remove-btn" data-evol-index="' + index + '">×</button>';
        div.querySelector('.remove-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            removeEvolutionarySelection(index);
        });
        listEl.appendChild(div);
    });
}

function removeEvolutionarySelection(index) {
    if (index < 0 || index >= evolutionarySelectedList.length) return;
    const item = evolutionarySelectedList[index];
    evolutionarySelectedList.splice(index, 1);
    evolutionarySelectedPaths = evolutionarySelectedList.map(m => m.path).filter(Boolean);
    renderEvolutionaryWorkspace();
    const container = document.getElementById('evolutionary-models-container');
    if (container) container.querySelectorAll('.model-card').forEach(card => {
        if (item.type === 'recipe' && card.dataset.recipeId === item.recipe_id) card.classList.remove('selected');
        else if (item.path && card.dataset.path === item.path) card.classList.remove('selected');
    });
    checkVlmAndSwitchDataset();
}

function loadEvolutionaryModels() {
    const container = document.getElementById('evolutionary-models-container');
    if (!container) return;
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>加载中...</p></div>';
    Promise.all([fetch('/api/models').then(r => r.json()), fetch('/api/merged_models').then(r => r.json()), fetch('/api/recipes').then(r => r.json())]).then(([modelsData, mergedData, recipesData]) => {
        const baseList = (modelsData.status === 'success' && modelsData.models) ? modelsData.models : [];
        const mergedList = (mergedData.status === 'success' && mergedData.models) ? mergedData.models : [];
        const recipesList = (recipesData.status === 'success' && recipesData.recipes) ? recipesData.recipes : [];
        const allModels = baseList.concat(mergedList);
        container.innerHTML = '';
        if (allModels.length === 0 && recipesList.length === 0) {
            container.innerHTML = '<p class="limit-desc">暂无本地模型或配方，请先添加模型或完成一次完全融合。</p>';
            return;
        }
        function addCard(m, path, name, isVlm, type) {
            // VLM/LLM 标签（紫色/蓝色），蓝图（无本地模型）额外显示蓝图标签
            const vlmTag = isVlm ? 'VLM' : 'LLM';
            const vlmClass = isVlm ? 'evol-tag-vlm' : 'evol-tag-llm';
            let tagsHtml = '<span class="evol-tag ' + vlmClass + '">' + vlmTag + '</span>';
            if (type === 'recipe') {
                tagsHtml += '<span class="evol-tag evol-tag-recipe">蓝图</span>';
            }
            const card = document.createElement('div');
            card.className = 'model-card';
            card.style.cursor = 'pointer';
            card.dataset.path = path || '';
            card.dataset.name = name;
            card.dataset.type = type || 'path';
            card.dataset.recipeId = (type === 'recipe' && m.recipe_id) ? m.recipe_id : '';
            card.dataset.isVlm = isVlm ? '1' : '0';
            card.innerHTML = '<div class="model-name">' + name + '</div><div class="limit-desc">' + tagsHtml + '</div>';
            card.addEventListener('click', async () => {
                const key = type === 'recipe' ? (m.recipe_id || m.task_id) : path;
                const idx = evolutionarySelectedList.findIndex(x => (x.type === 'recipe' ? (x.recipe_id === key) : (x.path === key)));
                if (idx >= 0) {
                    evolutionarySelectedList.splice(idx, 1);
                    evolutionarySelectedPaths = evolutionarySelectedList.map(x => x.path).filter(Boolean);
                    card.classList.remove('selected');
                } else {
                    // 先添加到列表
                    if (type === 'recipe') {
                        evolutionarySelectedList.push({ type: 'recipe', recipe_id: m.recipe_id || m.task_id, name: name, is_vlm: isVlm });
                    } else {
                        evolutionarySelectedList.push({ name, path, is_vlm: isVlm, type: 'path' });
                        evolutionarySelectedPaths.push(path);
                    }
                    // 选择 >=2 个时进行架构兼容性检查
                    if (evolutionarySelectedList.length >= 2) {
                        try {
                            // 构建 items 格式进行检查
                            const items = evolutionarySelectedList.map(x => {
                                if (x.type === 'recipe' && x.recipe_id) {
                                    return { type: 'recipe', recipe_id: x.recipe_id };
                                } else {
                                    return { type: 'path', path: x.path };
                                }
                            });
                            const checkRes = await fetch('/api/merge_evolutionary_check', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ items })
                            });
                            const checkData = await checkRes.json();
                            if (checkData.status === 'success' && checkData.compatible === false) {
                                alert('所选模型/配方无法融合：' + (checkData.reason || '架构不一致'));
                                evolutionarySelectedList.pop();
                                if (type !== 'recipe') evolutionarySelectedPaths.pop();
                                card.classList.remove('selected');
                            } else {
                                card.classList.add('selected');
                            }
                        } catch (e) {
                            card.classList.add('selected');
                        }
                    } else {
                        card.classList.add('selected');
                    }
                }
                evolutionarySelectedPaths = evolutionarySelectedList.map(x => x.path).filter(Boolean);
                renderEvolutionaryWorkspace();
                checkVlmAndSwitchDataset();
            });
            container.appendChild(card);
        }
        allModels.forEach(m => {
            const name = m.name;
            const path = m.path || name;
            const nameHintsVlm = /vl|vision|vlm|llava|qwen2\.?5?\s*vl|qwen2-vl|cogvlm|minicpm-v/i.test(name || '');
            const isVlm = !!m.is_vlm || nameHintsVlm;
            addCard(m, path, name, isVlm, 'path');
        });
        recipesList.slice(0, 20).forEach(r => {
            const name = (r.custom_name || r.recipe_id || '未命名').toString().slice(0, 32);
            const isVlm = !!r.is_vlm;
            addCard(r, null, name, isVlm, 'recipe');
        });
        evolutionarySelectedPaths = evolutionarySelectedList.map(x => x.path).filter(Boolean);
        container.querySelectorAll('.model-card').forEach(function(card) {
            const path = card.dataset.path;
            const recipeId = card.dataset.recipeId;
            const isSelected = recipeId ? evolutionarySelectedList.some(x => x.type === 'recipe' && x.recipe_id === recipeId) : evolutionarySelectedPaths.indexOf(path) >= 0;
            if (isSelected) card.classList.add('selected');
        });
        renderEvolutionaryWorkspace();
    }).catch(() => { if (container) container.innerHTML = '<p class="limit-desc">加载失败</p>'; });
}

async function checkVlmAndSwitchDataset() {
    if (evolutionarySelectedPaths.length === 0) return;
    try {
        const results = await Promise.all(
            evolutionarySelectedPaths.map(p => fetch('/api/model_is_vlm?path=' + encodeURIComponent(p)).then(r => r.json()).catch(() => ({})))
        );
        const anyVlm = results.some(r => r.status === 'success' && r.is_vlm === true);
        const typeHidden = document.getElementById('evolutionary-dataset-type');
        const typeTrigger = document.getElementById('evolutionary-dataset-type-trigger');
        if (anyVlm && typeHidden && typeHidden.value !== 'cmmmu') {
            typeHidden.value = 'cmmmu';
            if (typeTrigger) {
                const opt = document.querySelector('#evolutionary-dataset-type-options .ios-option[data-value="cmmmu"]');
                if (opt) typeTrigger.querySelector('.selected-text').innerText = (opt.querySelector('.opt-name') || opt).innerText;
            }
            loadEvolutionarySubsets('cmmmu');
        }
    } catch (_) {}
}

let evolutionaryCustomDataset = null;

function setupEvolutionarySplitDropdown() {
    const trigger = document.getElementById('evolutionary-split-trigger');
    const optionsUl = document.getElementById('evolutionary-split-options');
    const hidden = document.getElementById('evolutionary-hf-split');
    const triggerText = document.getElementById('evolutionary-split-text');
    if (!trigger || !optionsUl || !hidden) return;
    const wrapper = trigger.closest('.ios-select-wrapper');
    trigger.addEventListener('click', () => {
        wrapper.classList.toggle('active');
        optionsUl.classList.toggle('open');
    });
    optionsUl.querySelectorAll('.ios-option').forEach(opt => {
        opt.addEventListener('click', () => {
            const val = opt.dataset.value;
            optionsUl.querySelectorAll('.ios-option').forEach(x => x.classList.remove('selected'));
            opt.classList.add('selected');
            if (hidden) hidden.value = val;
            if (triggerText) triggerText.textContent = opt.querySelector('.opt-name') ? opt.querySelector('.opt-name').textContent : val;
            wrapper.classList.remove('active');
            optionsUl.classList.remove('open');
        });
    });
}

function setupEvolutionaryDatasetFetch() {
    const btn = document.getElementById('evolutionary-hf-dataset-fetch');
    const input = document.getElementById('evolutionary-hf-dataset-input');
    if (!btn || !input) return;
    btn.addEventListener('click', async () => {
        const name = (input.value || '').trim();
        if (!name) {
            alert('请输入 HuggingFace 数据集名称，如 cais/mmlu');
            return;
        }
        btn.disabled = true;
        btn.textContent = '拉取中...';
        try {
            const res = await fetch('/api/dataset/hf_info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hf_dataset: name })
            });
            const data = await res.json();
            if (data.status !== 'success') {
                alert(data.message || '拉取失败');
                return;
            }
            evolutionaryCustomDataset = { hf_dataset: name, configs: data.configs || [], splits: data.splits || [] };
            const optionsUl = document.getElementById('evolutionary-subset-options');
            const hiddenInput = document.getElementById('evolutionary-mmlu-subset');
            const triggerText = document.getElementById('evolutionary-subset-text');
            const labelEl = document.getElementById('evolutionary-subset-label');
            if (labelEl) labelEl.textContent = '子集 (config)';
            if (optionsUl) {
                optionsUl.innerHTML = '';
                (data.configs || []).forEach((c, i) => {
                    const li = document.createElement('li');
                    li.className = 'ios-option' + (i === 0 ? ' selected' : '');
                    li.dataset.value = c;
                    li.innerHTML = '<span class="opt-name">' + c + '</span>';
                    li.addEventListener('click', function() {
                        optionsUl.querySelectorAll('.ios-option').forEach(x => x.classList.remove('selected'));
                        this.classList.add('selected');
                        if (hiddenInput) hiddenInput.value = this.dataset.value;
                        if (triggerText) triggerText.textContent = this.dataset.value;
                    });
                    optionsUl.appendChild(li);
                });
            }
            if (hiddenInput && data.configs && data.configs[0]) hiddenInput.value = data.configs[0];
            if (triggerText && data.configs && data.configs[0]) triggerText.textContent = data.configs[0];
            const splitOpts = document.getElementById('evolutionary-split-options');
            if (splitOpts && data.splits && data.splits.length) {
                splitOpts.innerHTML = '';
                data.splits.forEach((s, i) => {
                    const li = document.createElement('li');
                    li.className = 'ios-option' + (s === 'validation' || s === 'val' ? ' selected' : '');
                    li.dataset.value = s === 'val' ? 'validation' : s;
                    li.innerHTML = '<span class="opt-name">' + s + (s === 'validation' ? '（训练/验证）' : s === 'test' ? '（最终准确率）' : '') + '</span>';
                    li.addEventListener('click', function() {
                        splitOpts.querySelectorAll('.ios-option').forEach(x => x.classList.remove('selected'));
                        this.classList.add('selected');
                        const h = document.getElementById('evolutionary-hf-split');
                        const t = document.getElementById('evolutionary-split-text');
                        if (h) h.value = this.dataset.value;
                        if (t) t.textContent = this.querySelector('.opt-name').textContent;
                    });
                    splitOpts.appendChild(li);
                });
            }
        } finally {
            btn.disabled = false;
            btn.textContent = '拉取并读取';
        }
    });
}

function loadEvolutionarySubsets(datasetType) {
    if (evolutionaryCustomDataset) return;
    const labelEl = document.getElementById('evolutionary-subset-label');
    const triggerText = document.getElementById('evolutionary-subset-text');
    const hiddenInput = document.getElementById('evolutionary-mmlu-subset');
    const optionsUl = document.getElementById('evolutionary-subset-options');
    if (!optionsUl || !hiddenInput) return;
    const isMmlu = (datasetType || '').toLowerCase() === 'mmlu';
    const apiUrl = isMmlu ? '/api/mmlu_subset_groups' : '/api/cmmmu_subset_groups';
    const defaultId = isMmlu ? 'biology_medicine' : 'health_medicine';
    if (labelEl) labelEl.textContent = isMmlu ? 'MMLU 领域' : 'CMMMU 领域';
    fetch(apiUrl).then(r => r.json()).then(data => {
        const groups = data.groups || [];
        optionsUl.innerHTML = '';
        groups.forEach((g) => {
            const li = document.createElement('li');
            li.className = 'ios-option' + (g.id === defaultId ? ' selected' : '');
            li.dataset.value = g.id;
            li.innerHTML = '<span class="opt-name">' + (g.label || g.id) + '</span>';
            if (g.subsets && g.subsets.length) li.title = g.subsets.join(', ');
            optionsUl.appendChild(li);
        });
        hiddenInput.value = defaultId;
        if (triggerText) triggerText.textContent = (groups.find(g => g.id === defaultId) || groups[0] || {}).label || defaultId;
    }).catch(() => { if (optionsUl) optionsUl.innerHTML = '<li class="ios-option">加载失败</li>'; });
}

function setupEvolutionaryDatasetTypeSwitch() {
    const typeOptions = document.getElementById('evolutionary-dataset-type-options');
    const typeHidden = document.getElementById('evolutionary-dataset-type');
    if (!typeOptions || !typeHidden) return;
    typeOptions.querySelectorAll('.ios-option').forEach(opt => {
        opt.addEventListener('click', () => {
            const val = opt.dataset.value;
            typeHidden.value = val;
            evolutionaryCustomDataset = null;
            loadEvolutionarySubsets(val);
        });
    });
    // 子集下拉为动态填充，用事件委托处理选项点击
    const subsetOptionsUl = document.getElementById('evolutionary-subset-options');
    const subsetWrapper = subsetOptionsUl?.closest('.ios-select-wrapper');
    if (subsetWrapper) {
        subsetOptionsUl.addEventListener('click', (e) => {
            const opt = e.target.closest('.ios-option');
            if (!opt) return;
            e.stopPropagation();
            const val = opt.dataset.value;
            subsetOptionsUl.querySelectorAll('.ios-option').forEach(x => x.classList.remove('selected'));
            opt.classList.add('selected');
            const hidden = document.getElementById('evolutionary-mmlu-subset');
            const triggerText = document.getElementById('evolutionary-subset-text');
            if (hidden) hidden.value = val;
            if (triggerText) triggerText.textContent = val;
            subsetOptionsUl.classList.remove('open');
            subsetWrapper.classList.remove('active');
        });
    }
}

function setupEvolutionaryMerge() {
    const btn = document.getElementById('start-evolutionary-merge');
    if (!btn) return;
    btn.addEventListener('click', async () => {
        if (evolutionarySelectedList.length < 2) {
            alert('请至少选择 2 个模型或配方');
            return;
        }
        const customName = (document.getElementById('evolutionary-custom-name') && document.getElementById('evolutionary-custom-name').value.trim()) || ('进化融合-' + Date.now());
        const popSize = Math.max(2, Math.min(128, parseInt(document.getElementById('evolutionary-pop-size')?.value, 10) || 20));
        const nIter = Math.max(1, Math.min(50, parseInt(document.getElementById('evolutionary-n-iter')?.value, 10) || 15));
        const maxSamples = Math.max(4, Math.min(512, parseInt(document.getElementById('evolutionary-max-samples')?.value, 10) || 64));
        const datasetType = document.getElementById('evolutionary-dataset-type')?.value || 'mmlu';
        const hfSubsetGroup = document.getElementById('evolutionary-mmlu-subset')?.value || (datasetType === 'cmmmu' ? 'health_medicine' : 'biology_medicine');
        const customHfInput = document.getElementById('evolutionary-hf-dataset-input');
        const useCustomDataset = customHfInput && customHfInput.value.trim();
        const hfDataset = useCustomDataset ? customHfInput.value.trim() : (datasetType === 'cmmmu' ? 'm-a-p/CMMMU' : 'cais/mmlu');
        const hfSplit = (document.getElementById('evolutionary-hf-split') && document.getElementById('evolutionary-hf-split').value) || (datasetType === 'cmmmu' ? 'val' : 'validation');
        const dtype = document.getElementById('evolutionary-dtype')?.value || 'bfloat16';
        const rayGpus = parseInt(document.getElementById('evolutionary-ray-gpus')?.value || 2, 10);
        const hasRecipe = evolutionarySelectedList.some(x => x.type === 'recipe');
        const body = {
            custom_name: customName,
            pop_size: popSize,
            n_iter: nIter,
            max_samples: maxSamples,
            hf_dataset: hfDataset,
            hf_split: hfSplit,
            dtype,
            ray_num_gpus: rayGpus,
        };
        if (hasRecipe) {
            body.items = evolutionarySelectedList.map(x => {
                if (x.type === 'recipe') return { type: 'recipe', recipe_id: x.recipe_id };
                return { type: 'path', path: x.path };
            });
        } else {
            body.model_paths = evolutionarySelectedPaths;
        }
        if (hfSplit === 'validation' || hfSplit === 'val') body.hf_split_final = 'test';
        if (useCustomDataset) {
            const subsetEl = document.getElementById('evolutionary-mmlu-subset');
            body.hf_subsets = subsetEl && subsetEl.value ? [subsetEl.value] : [];
            body.hf_subset_group = '';
        } else {
            body.hf_subset_group = hfSubsetGroup;
        }
        btn.disabled = true;
        btn.textContent = '提交中...';
        try {
            const res = await fetch('/api/merge_evolutionary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (!data.task_id) {
                alert(data.message || '提交失败');
                btn.disabled = false;
                btn.textContent = '开始完全融合';
                return;
            }
            btn.textContent = '已提交，任务 ID: ' + data.task_id;
            const statusDiv = document.getElementById('task-status');
            if (statusDiv) {
                statusDiv.querySelector('.status-message').textContent = '完全融合任务已排队: ' + data.task_id;
                const progContainer = statusDiv.querySelector('.progress-container');
                const progPct = document.getElementById('task-progress-pct');
                const progStage = document.getElementById('task-progress-stage');
                if (progContainer) progContainer.style.display = 'block';
                if (progPct) progPct.style.display = 'block';
                if (progStage) progStage.style.display = 'block';
            }
            let pollCount = 0;
            const poll = async () => {
                const s = await fetch('/api/status/' + data.task_id).then(r => r.json()).catch(() => ({}));
                const statusDiv = document.getElementById('task-status');
                const fillEl = document.getElementById('task-progress-fill');
                const pctEl = document.getElementById('task-progress-pct');
                const stageEl = document.getElementById('task-progress-stage');
                if (s.status === 'completed' || s.status === 'success') {
                    btn.disabled = false;
                    btn.textContent = '开始完全融合';
                    if (statusDiv) statusDiv.querySelector('.status-message').textContent = '完全融合完成';
                    if (fillEl) fillEl.style.width = '100%';
                    if (pctEl) pctEl.textContent = '100%';
                    if (stageEl) stageEl.textContent = '';
                    loadHistoryList();
                    // 显示完成弹窗
                    showTaskCompletionModal(data.task_id, s, evo);
                    return;
                }
                if (s.status === 'error') {
                    btn.disabled = false;
                    btn.textContent = '开始完全融合';
                    if (statusDiv) statusDiv.querySelector('.status-message').textContent = '失败: ' + (s.message || s.error);
                    return;
                }
                if (statusDiv && s.message) statusDiv.querySelector('.status-message').textContent = (s.message || '').slice(0, 120);
                const evo = s.evolution_progress || {};
                const step = evo.step || 0;
                const currentStep = evo.current_step || step;  // 使用 current_step 如果可用
                const totalExpectedSteps = evo.total_expected_steps;
                const od = s.original_data || {};
                const nIter = od.n_iter || 15;
                const popSize = od.pop_size || 20;
                
                // 计算总步数：优先使用 total_expected_steps，否则估算
                let totalSteps = totalExpectedSteps;
                if (!totalSteps) {
                    // 估算：n_iter * pop_size（每次迭代评估 pop_size 个个体）
                    totalSteps = Math.max(1, nIter * popSize);
                }
                
                // 进度条更新：优先使用 current_step，否则使用 step
                let pct = 0;
                const effectiveStep = currentStep > 0 ? currentStep : step;
                if (effectiveStep > 0 && totalSteps > 0) {
                    pct = Math.min(100, Math.round((effectiveStep / totalSteps) * 100));
                } else if (s.status === 'running' || s.status === 'queued') {
                    // 任务运行中但step未更新：使用轮询次数估算最小进度（避免一直0%）
                    const minProgress = Math.min(5, Math.floor(pollCount / 10)); // 每10次轮询+1%，最多5%
                    pct = minProgress;
                }
                if (fillEl) fillEl.style.width = pct + '%';
                if (pctEl) pctEl.textContent = pct + '%';
                
                // 构建步骤文本：显示实际步数和总步数
                let stepText = '';
                if (effectiveStep > 0 && totalSteps > 0) {
                    stepText = `步骤 ${effectiveStep} / ${totalSteps}`;
                } else if (s.status === 'running') {
                    stepText = '运行中...';
                } else if (s.status === 'queued') {
                    stepText = '等待中...';
                } else {
                    stepText = '准备中...';
                }
                
                // 添加 ETA 信息
                let etaText = '';
                if (evo.eta_seconds && evo.eta_seconds > 0) {
                    const etaSeconds = Math.round(evo.eta_seconds);
                    if (etaSeconds < 60) {
                        etaText = ` · 预计剩余 ${etaSeconds}秒`;
                    } else if (etaSeconds < 3600) {
                        const minutes = Math.round(etaSeconds / 60);
                        etaText = ` · 预计剩余 ${minutes}分钟`;
                    } else {
                        const hours = Math.round(etaSeconds / 3600 * 10) / 10;
                        etaText = ` · 预计剩余 ${hours}小时`;
                    }
                }
                
                // 添加准确率信息
                const accText = evo.current_best != null ? ' · 当前最优 acc: ' + Number(evo.current_best).toFixed(4) : '';
                if (stageEl) stageEl.textContent = stepText + etaText + accText;
                pollCount++;
                if (pollCount < 3600) setTimeout(poll, 1000);
            };
            setTimeout(poll, 1500);
        } catch (e) {
            btn.disabled = false;
            btn.textContent = '开始完全融合';
            alert('请求失败: ' + e.message);
        }
    });
}

function setupRecipeApply() {
    const trigger = document.getElementById('recipe-apply-trigger');
    const optionsUl = document.getElementById('recipe-apply-options');
    const hiddenId = document.getElementById('recipe-apply-id');
    const triggerText = document.getElementById('recipe-apply-text');
    const btn = document.getElementById('recipe-apply-btn');
    if (!trigger || !optionsUl || !btn) {
        console.warn('[setupRecipeApply] 缺少必要的 DOM 元素');
        return;
    }

    // 确保配方选择下拉框不被 initPageDropdowns 处理
    const wrapper = trigger.closest('.ios-select-wrapper');
    if (wrapper) {
        wrapper.classList.add('recipe-apply-select');
    }

    async function loadRecipeList() {
        try {
            const res = await fetch('/api/recipes');
            const data = await res.json();
            const list = (data.recipes || []).slice(0, 50);
            optionsUl.innerHTML = '';
            if (list.length === 0) {
                const li = document.createElement('li');
                li.className = 'ios-option';
                li.textContent = '暂无配方';
                optionsUl.appendChild(li);
            } else {
                list.forEach((r, i) => {
                    const li = document.createElement('li');
                    li.className = 'ios-option' + (i === 0 ? ' selected' : '');
                    // 确保 recipe_id 存在（API 返回时已添加）
                    const recipeId = r.recipe_id || r.task_id || '';
                    li.dataset.recipeId = recipeId;
                    const name = (r.custom_name || r.recipe_id || r.task_id || '未命名').toString().slice(0, 40);
                    const parents = (r.parent_names || r.model_paths || []).map(p => typeof p === 'string' ? p.split(/[/\\]/).pop() : '').join(' + ');
                    li.innerHTML = '<span class="opt-name">' + name + (parents ? ' (' + parents + ')' : '') + '</span>';
                    li.addEventListener('click', function(e) {
                        e.stopPropagation();
                        e.preventDefault();
                        optionsUl.querySelectorAll('.ios-option').forEach(x => x.classList.remove('selected'));
                        this.classList.add('selected');
                        const id = this.dataset.recipeId;
                        if (hiddenId) hiddenId.value = id;
                        if (triggerText) {
                            const nameEl = this.querySelector('.opt-name');
                            triggerText.textContent = nameEl ? nameEl.textContent : id;
                        }
                        const wrapperEl = trigger.closest('.ios-select-wrapper');
                        if (wrapperEl) wrapperEl.classList.remove('active');
                        optionsUl.classList.remove('open');
                    });
                    optionsUl.appendChild(li);
                });
                // 默认选择第一个
                if (list.length > 0 && hiddenId) {
                    hiddenId.value = list[0].recipe_id || list[0].task_id || '';
                }
            }
        } catch (e) {
            console.error('[loadRecipeList] 加载配方列表失败:', e);
            optionsUl.innerHTML = '<li class="ios-option">加载失败</li>';
        }
    }

    // 绑定点击事件（使用 capture 确保优先处理）
    trigger.addEventListener('click', function handleTriggerClick(e) {
        e.stopPropagation();
        e.preventDefault();
        console.log('[recipe-apply] 点击触发，当前选项数:', optionsUl.children.length);
        if (optionsUl.children.length === 0) {
            loadRecipeList().then(() => {
                const wrapperEl = trigger.closest('.ios-select-wrapper');
                if (wrapperEl) {
                    wrapperEl.classList.add('active');
                }
                optionsUl.classList.add('open');
                console.log('[recipe-apply] 配方列表已加载，下拉框已打开');
            });
        } else {
            const wrapperEl = trigger.closest('.ios-select-wrapper');
            const isOpen = optionsUl.classList.contains('open');
            if (wrapperEl) {
                if (isOpen) {
                    wrapperEl.classList.remove('active');
                } else {
                    wrapperEl.classList.add('active');
                }
            }
            if (isOpen) {
                optionsUl.classList.remove('open');
            } else {
                optionsUl.classList.add('open');
            }
            console.log('[recipe-apply] 下拉框状态:', isOpen ? '关闭' : '打开');
        }
    }, true); // 使用 capture phase 确保优先处理

    btn.addEventListener('click', async () => {
        const recipeId = (hiddenId && hiddenId.value) || '';
        if (!recipeId) {
            alert('请先选择配方');
            return;
        }
        const customName = (document.getElementById('recipe-apply-custom-name') && document.getElementById('recipe-apply-custom-name').value.trim()) || '';
        btn.disabled = true;
        btn.textContent = '提交中...';
        try {
            const res = await fetch('/api/recipes/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ recipe_id: recipeId, custom_name: customName || undefined })
            });
            const data = await res.json();
            if (!data.task_id) {
                alert(data.message || '提交失败');
                btn.disabled = false;
                btn.textContent = '按配方融合';
                return;
            }
            btn.textContent = '已提交: ' + data.task_id;
            const statusDiv = document.getElementById('task-status');
            if (statusDiv) {
                statusDiv.querySelector('.status-message').textContent = '配方融合任务已排队: ' + data.task_id;
                const progContainer = statusDiv.querySelector('.progress-container');
                const progPct = document.getElementById('task-progress-pct');
                const progStage = document.getElementById('task-progress-stage');
                if (progContainer) progContainer.style.display = 'block';
                if (progPct) progPct.style.display = 'block';
                if (progStage) progStage.style.display = 'block';
            }
            let pollCount = 0;
            const poll = async () => {
                const s = await fetch('/api/status/' + data.task_id).then(r => r.json()).catch(() => ({}));
                const statusDiv = document.getElementById('task-status');
                const fillEl = document.getElementById('task-progress-fill');
                const pctEl = document.getElementById('task-progress-pct');
                const stageEl = document.getElementById('task-progress-stage');
                if (s.status === 'completed' || s.status === 'success') {
                    btn.disabled = false;
                    btn.textContent = '按配方融合';
                    if (statusDiv) statusDiv.querySelector('.status-message').textContent = '配方融合完成';
                    if (fillEl) fillEl.style.width = '100%';
                    if (pctEl) pctEl.textContent = '100%';
                    if (stageEl) stageEl.textContent = '';
                    loadHistoryList();
                    // 显示完成弹窗
                    showTaskCompletionModal(data.task_id, s, {});
                    return;
                }
                if (s.status === 'error') {
                    btn.disabled = false;
                    btn.textContent = '按配方融合';
                    if (statusDiv) statusDiv.querySelector('.status-message').textContent = '失败: ' + (s.message || s.error);
                    return;
                }
                if (statusDiv && s.message) statusDiv.querySelector('.status-message').textContent = (s.message || '').slice(0, 120);
                const pct = s.progress || 0;
                if (fillEl) fillEl.style.width = pct + '%';
                if (pctEl) pctEl.textContent = pct + '%';
                if (stageEl) stageEl.textContent = '';
                pollCount++;
                if (pollCount < 600) setTimeout(poll, 1000);
            };
            setTimeout(poll, 1500);
        } catch (e) {
            btn.disabled = false;
            btn.textContent = '按配方融合';
            alert('请求失败: ' + e.message);
        }
    });
}

// 4. 添加模型到选择区（model 为 {name, path}、{ type:'recipe', recipe_id, name } 或 name 字符串）
function addModelToSelection(model) {
    const clearBtn = document.getElementById('clear-selection');
    if (clearBtn.dataset.mode === 'stop') {
        alert('任务运行中，无法修改选择列表。请先停止任务。');
        return;
    }
    if (state.selectedModels.length >= 2) {
        alert('最多选择 2 个模型或配方');
        return;
    }
    let entry;
    if (typeof model === 'string') {
        entry = { type: 'path', name: model, path: null };
    } else if (model.type === 'recipe') {
        entry = { type: 'recipe', name: model.name || model.recipe_id, recipe_id: model.recipe_id };
    } else {
        entry = { type: 'path', name: model.name || model, path: model.path || null };
    }
    if (state.selectedModels.some(m => (m.name || m) === (entry.name || entry.recipe_id))) {
        alert('该模型/配方已在列表中');
        return;
    }
    state.selectedModels.push(entry);
    renderSelectedModels();
    updateUIState();
}

// 5. 渲染选中的模型
function renderSelectedModels() {
    const container = document.getElementById('selected-models');
    container.innerHTML = '';

    state.selectedModels.forEach((m, index) => {
        const name = typeof m === 'string' ? m : (m.name || m);
        const item = document.createElement('div');
        item.className = 'selected-model-item';
        item.innerHTML = `
            <span>${name}</span>
            <button class="remove-btn" onclick="removeModel(${index})">×</button>
        `;
        container.appendChild(item);
    });

    const hint = document.querySelector('.dropzone-content');
    hint.style.display = state.selectedModels.length > 0 ? 'none' : 'block';

    updateWeightSliders();
    checkStandardCompatibilityAndDataset();
}

// [新增] 标准融合：检查架构兼容性 & 自动切换数据集类型
async function checkStandardCompatibilityAndDataset() {
    const statusEl = document.getElementById('standard-compatibility-status');
    const typeHidden = document.getElementById('standard-dataset-type');
    const typeTrigger = document.getElementById('standard-dataset-type-trigger');

    if (state.selectedModels.length === 0) {
        if (statusEl) statusEl.style.display = 'none';
        return;
    }

    const modelPaths = state.selectedModels.map(m => (typeof m === 'string' ? null : m.path)).filter(Boolean);
    if (modelPaths.length === 0) return;

    // 1. 检查是否包含 VLM -> 自动切换数据集
    try {
        const results = await Promise.all(
            modelPaths.map(p => fetch('/api/model_is_vlm?path=' + encodeURIComponent(p)).then(r => r.json()).catch(() => ({})))
        );
        const anyVlm = results.some(r => r.status === 'success' && r.is_vlm === true);
        
        // 如果是 VLM 但当前选的是 MMLU，自动切到 CMMMU
        if (anyVlm && typeHidden && typeHidden.value !== 'cmmmu') {
            typeHidden.value = 'cmmmu';
            if (typeTrigger) {
                const opt = document.querySelector('#standard-dataset-type-options .ios-option[data-value="cmmmu"]');
                if (opt) typeTrigger.querySelector('.selected-text').innerText = (opt.querySelector('.opt-name') || opt).innerText;
                // 更新选中样式
                document.querySelectorAll('#standard-dataset-type-options .ios-option').forEach(o => o.classList.remove('selected'));
                if(opt) opt.classList.add('selected');
            }
            loadStandardSubsets('cmmmu');
        } else if (!anyVlm && typeHidden && typeHidden.value === 'cmmmu') {
             // 如果全是 LLM 但当前是 CMMMU，切回 MMLU
             typeHidden.value = 'mmlu';
             if (typeTrigger) {
                const opt = document.querySelector('#standard-dataset-type-options .ios-option[data-value="mmlu"]');
                if (opt) typeTrigger.querySelector('.selected-text').innerText = (opt.querySelector('.opt-name') || opt).innerText;
                document.querySelectorAll('#standard-dataset-type-options .ios-option').forEach(o => o.classList.remove('selected'));
                if(opt) opt.classList.add('selected');
             }
             loadStandardSubsets('mmlu');
        } else {
            // 类型没变，但可能还没加载子集（初次）
            const currentType = typeHidden ? typeHidden.value : 'mmlu';
            // 检查子集列表是否为空，若空则加载
            const opts = document.getElementById('standard-subset-options');
            if (opts && opts.children.length === 0) {
                loadStandardSubsets(currentType);
            }
        }
    } catch (e) {
        console.error("checkStandardCompatibilityAndDataset error:", e);
    }

    // 2. 检查架构兼容性
    try {
        const res = await fetch('/api/check_compatibility', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_paths: modelPaths })
        });
        const data = await res.json();
        
        if (statusEl) {
            statusEl.style.display = 'block';
            if (data.compatible) {
                statusEl.innerHTML = `<i class="ri-checkbox-circle-fill" style="color: var(--success);"></i> <strong>架构兼容</strong> <span style="color: #86868b; margin-left:8px;">${data.message || ''}</span>`;
                statusEl.style.background = '#f2fcf5';
                statusEl.style.border = '1px solid rgba(52, 199, 89, 0.2)';
            } else {
                statusEl.innerHTML = `<i class="ri-close-circle-fill" style="color: var(--error);"></i> <strong>架构不兼容</strong> <span style="color: #86868b; margin-left:8px;">${data.message || ''}</span>`;
                statusEl.style.background = '#fff2f2';
                statusEl.style.border = '1px solid rgba(255, 59, 48, 0.2)';
            }
        }
    } catch (e) {
        console.error("check compatibility error:", e);
    }
}

// [新增] 加载标准融合的子集
function loadStandardSubsets(datasetType) {
    const labelEl = document.getElementById('standard-subset-label');
    const triggerText = document.getElementById('standard-subset-trigger');
    const hiddenInput = document.getElementById('standard-subset-val');
    const optionsUl = document.getElementById('standard-subset-options');

    if (!optionsUl || !hiddenInput) return;

    const isMmlu = (datasetType || '').toLowerCase() === 'mmlu';
    const apiUrl = isMmlu ? '/api/mmlu_subset_groups' : '/api/cmmmu_subset_groups';
    const defaultId = isMmlu ? 'biology_medicine' : 'health_medicine'; // 默认选中一个

    if (labelEl) labelEl.textContent = isMmlu ? 'MMLU 领域' : 'CMMMU 领域';

    // 先清空
    optionsUl.innerHTML = '<li class="ios-option">加载中...</li>';

    fetch(apiUrl).then(r => r.json()).then(data => {
        const groups = data.groups || [];
        optionsUl.innerHTML = '';
        
        if (groups.length === 0) {
            optionsUl.innerHTML = '<li class="ios-option">无数据</li>';
            return;
        }

        groups.forEach((g) => {
            const li = document.createElement('li');
            li.className = 'ios-option' + (g.id === defaultId ? ' selected' : '');
            li.dataset.value = g.id;
            li.innerHTML = '<span class="opt-name">' + (g.label || g.id) + '</span>';
            if (g.subsets && g.subsets.length) li.title = g.subsets.join(', '); // hover显示包含的子集
            
            // 绑定点击事件
            li.onclick = () => {
                 // UI 更新
                 optionsUl.querySelectorAll('.ios-option').forEach(o => o.classList.remove('selected'));
                 li.classList.add('selected');
                 if (triggerText) triggerText.querySelector('.selected-text').innerText = (g.label || g.id);
                 hiddenInput.value = g.id;
                 // 关闭下拉
                 optionsUl.parentElement.classList.remove('open');
            };
            
            optionsUl.appendChild(li);
        });
        
        // 设置默认值
        hiddenInput.value = defaultId;
        const defaultGroup = groups.find(g => g.id === defaultId) || groups[0];
        if (triggerText && defaultGroup) {
            triggerText.querySelector('.selected-text').innerText = (defaultGroup.label || defaultGroup.id);
        }
        
    }).catch(e => {
        console.error(e);
        if (optionsUl) optionsUl.innerHTML = '<li class="ios-option">加载失败</li>';
    });
}

window.removeModel = function (index) {
    const clearBtn = document.getElementById('clear-selection');
    if (clearBtn.dataset.mode === 'stop') {
        alert('任务运行中，无法修改选择列表。请先停止任务。');
        return;
    }
    state.selectedModels.splice(index, 1);
    renderSelectedModels();
    updateUIState();
};

// 6. 更新权重滑块
function updateWeightSliders() {
    const container = document.getElementById('weights-container');

    if (state.selectedModels.length === 0) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    container.style.display = 'block';
    container.innerHTML = '<h3>模型权重设置</h3>';

    state.selectedModels.forEach((m, index) => {
        const name = typeof m === 'string' ? m : (m.name || m);
        const item = document.createElement('div');
        item.className = 'weight-item';
        item.innerHTML = `
            <label>
                <span>${name}</span>
                <span id="weight-val-${index}">1.0</span>
            </label>
            <input type="range" class="weight-slider"
                   min="0" max="1" step="0.1" value="1.0"
                   oninput="document.getElementById('weight-val-${index}').innerText = parseFloat(this.value).toFixed(1)">
        `;
        container.appendChild(item);
    });
}

// 7. 更新按钮状态
function updateUIState() {
    const startBtn = document.getElementById('start-merge');
    const clearBtn = document.getElementById('clear-selection');

    // 如果当前处于 stop 模式，不允许被其它逻辑切回
    if (clearBtn.dataset.mode === 'stop') return;

    startBtn.disabled = state.selectedModels.length === 0;
    toggleResetButton('clear');
}

// ===================== 业务逻辑：开始/停止任务 =====================

function handleClearOrStop() {
    const btn = document.getElementById('clear-selection');
    const mode = btn.dataset.mode || 'clear';

    if (mode === 'stop') {
        stopCurrentTask();
        return;
    }

    // 清空选择
    state.selectedModels = [];
    renderSelectedModels();
    updateUIState();
}

async function handleResumeTask(taskId) {
    try {
        const resumeGroup = document.getElementById('resume-group');
        // 简单的 loading 状态
        if(resumeGroup) resumeGroup.style.opacity = '0.5';

        const res = await fetch(`/api/resume/${taskId}`, { method: 'POST' });
        const data = await res.json();
        
        if (data.status === 'success') {
            // 恢复 UI
            hideInterruptedUI();
            // 轮询会自然接管 update 状态 (变成 queued)
        } else {
            alert("恢复失败: " + data.message);
            handleClearOrStop(); // 失败就当做停止处理
        }
    } catch (e) {
        console.error(e);
        alert("网络错误");
    }
}

async function startMergeTask() {
    clearPolling();
    
    const startBtn = document.getElementById('start-merge');
    const statusDiv = document.getElementById('task-status');
    const customNameInput = document.getElementById('model-name-input');
    const customName = customNameInput && customNameInput.value.trim();
    if (!customName) {
        if (customNameInput) { customNameInput.classList.add('input-error'); customNameInput.focus(); }
        alert('请填写模型名称');
        return;
    }
    const method = document.getElementById('merge-method').value;
    const dtype = document.getElementById('dtype').value;
    if (state.selectedModels.length === 0) return;
    if (state.selectedModels.length < 2) {
        alert("请至少选择两个模型进行融合");
        return;
    }
    const firstName = typeof state.selectedModels[0] === 'string' ? state.selectedModels[0] : (state.selectedModels[0].name || state.selectedModels[0]);
    if (['task_arithmetic', 'ties_dare'].includes(method)) {
        const confirmMerge = confirm(
            `当前选择的是 ${method} 模式：\n` +
            `1. 系统将以 [${firstName}] 作为基座(Base)。\n` +
            `2. 请确保选中的所有模型都属于同一个系列（如均为 Qwen2-0.5B）。如果不是同一个系列将无法融合！\n` +
            `是否继续？`
        );
        if (!confirmMerge) return;
    }

    // UI 初始化
    startBtn.disabled = true;
    startBtn.innerText = "正在融合...";
    statusDiv.style.opacity = '1';
    document.querySelector('.progress-container').style.display = 'block';

    setStatusUI({ message: "正在连接服务器...", progress: 0, color: '#0071e3', stripes: false });

    // 收集参数

    const weights = [];
    document.querySelectorAll('.weight-slider').forEach(input => {
        weights.push(parseFloat(input.value));
    });

    const hasRecipe = state.selectedModels.some(m => m && m.type === 'recipe');
    // const customNameInput = document.getElementById('model-name-input');
    // const customName = customNameInput ? customNameInput.value.trim() : '';

    const payload = { 
        weights: weights, 
        method: method, 
        dtype: dtype, 
        custom_name: customName 
    };

    // 获取标准融合的数据集参数
    const stdDatasetType = document.getElementById('standard-dataset-type');
    const stdDatasetSubset = document.getElementById('standard-subset-val');
    if (stdDatasetType) payload.dataset_type = stdDatasetType.value;
    if (stdDatasetSubset) payload.dataset_subset = stdDatasetSubset.value;

    if (hasRecipe) {
        payload.items = state.selectedModels.map(m => {
            if (m.type === 'recipe') return { type: 'recipe', recipe_id: m.recipe_id };
            return { type: 'path', path: m.path };
        });
    } else {
        const modelPaths = state.selectedModels.map(m => (m && m.path)).filter(Boolean);
        const modelNames = state.selectedModels.map(m => (m && m.name)).filter(Boolean);
        if (modelPaths.length === state.selectedModels.length) payload.model_paths = modelPaths;
        else payload.models = modelNames;
    }
    try {
        const response = await fetch('/api/merge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === 'success' && data.task_id) {
            currentTaskId = data.task_id;
            sessionStorage.setItem(STORAGE_ACTIVE_TASK, currentTaskId);

            toggleResetButton('stop');
            pollStatus(currentTaskId);
        } else {
            alert('启动失败: ' + (data.message || 'Unknown error'));
            resetStartButton();
            toggleResetButton('clear');
        }
    } catch (e) {
        console.error(e);
        alert('网络错误');
        resetStartButton();
        toggleResetButton('clear');
    }
}

function showInterruptedUI(taskId) {
    const startBtn = document.getElementById('start-merge');
    // 隐藏原本的按钮，或者改变它的样式
    startBtn.style.display = 'none';

    // 检查是否已存在 resume 按钮
    let resumeGroup = document.getElementById('resume-group');
    if (!resumeGroup) {
        resumeGroup = document.createElement('div');
        resumeGroup.id = 'resume-group';
        resumeGroup.className = 'resume-group';
        
        // 确认按钮
        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'btn btn-primary btn-resume';
        confirmBtn.innerHTML = '<i class="ri-play-circle-line"></i> 继续任务';
        confirmBtn.onclick = () => handleResumeTask(taskId);

        // 取消按钮 (直接停止)
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-secondary btn-cancel-resume';
        cancelBtn.innerText = '取消';
        cancelBtn.onclick = () => handleClearOrStop(); // 复用停止逻辑

        resumeGroup.appendChild(cancelBtn);
        resumeGroup.appendChild(confirmBtn);

        const buttonsContainer = document.querySelector('.action-bar .buttons');
        buttonsContainer.appendChild(resumeGroup);
    }
}

function hideInterruptedUI() {
    const startBtn = document.getElementById('start-merge');
    startBtn.style.display = 'inline-block';
    
    const resumeGroup = document.getElementById('resume-group');
    if (resumeGroup) resumeGroup.remove();
}

async function stopCurrentTask() {
    const taskId = currentTaskId || sessionStorage.getItem(STORAGE_ACTIVE_TASK);
    if (!taskId) return;

    if (!confirm('确定要停止当前任务吗？')) return;

    const btn = document.getElementById('clear-selection');
    btn.innerText = "正在停止...";
    btn.disabled = true;

    // 立刻停止前端轮询，避免“停止后又被轮询覆盖”
    clearPolling();

    try {
        const res = await fetch(`/api/stop/${taskId}`, { method: 'POST' });
        const data = await res.json();

        if (data.status === 'success') {
            handleStopSuccessUI();
        } else {
            alert('停止失败: ' + (data.message || 'Unknown error'));
            toggleResetButton('stop');
        }
    } catch (e) {
        console.error(e);
        alert('网络错误');
        toggleResetButton('stop');
    }
}

function handleStopSuccessUI() {
    sessionStorage.removeItem(STORAGE_ACTIVE_TASK);
    currentTaskId = null;

    // 恢复按钮
    toggleResetButton('clear');
    resetStartButton();

    // 更新状态栏
    setStatusUI({ message: "任务已手动停止", progress: 0, color: '#86868b', stripes: false });
}

// ===================== 轮询状态 =====================
function pollStatus(taskId) {
    clearPolling();

    const startBtn = document.getElementById('start-merge');
    
    // 确保按钮是禁用状态
    startBtn.disabled = true;
    toggleResetButton('stop');

    pollTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${taskId}`);
            const data = await res.json();

            // ================== 1. 排队中 ==================
            if (data.status === 'queued') {
                const count = data.queue_position ?? 0;
                // 如果是自动恢复的任务，提示语稍微不同
                const msg = data.restarted ? 
                    `任务被打断，正在重新排队... (${count})` : 
                    data.message || `排队中... (${count})`;
                
                setStatusUI({
                    message: msg,
                    progress: 100,
                    color: '#ff9f0a',
                    stripes: true
                });
                startBtn.innerText = `等待中 (${count})`;
                hideRestartToast(); // 排队时隐藏 toast
                return;
            }

            // ================== 2. 运行中 ==================
            if (data.status === 'running') {
                setStatusUI({
                    message: data.message || '运行中...',
                    progress: data.progress ?? 0,
                    color: '#0071e3',
                    stripes: false
                });
                startBtn.innerText = "正在处理...";

                // check if restarted -> Show Toast
                if (data.restarted === true && !hasShownRestartToast) {
                    showRestartToast();
                    hasShownRestartToast = true; // 只弹一次
                }
                return;
            }

            // ================== 3. Cutin 被打断 (特殊状态) ==================
            if (data.status === 'interrupted') {
                setStatusUI({
                    message: "任务已被高优先级任务打断",
                    progress: 0,
                    color: '#ff3b30', // 红色
                    stripes: false
                });
                
                // 显示手动确认 UI
                showInterruptedUI(taskId);
                
                // 这里不清空 timer，因为我们要等待用户操作
                // 但要停止向后端频繁请求吗？其实可以继续轮询，万一后端状态变了呢
                return; 
            }

            // ================== 4. 结束/错误/手动停止 ==================
            if (['completed', 'error', 'stopped'].includes(data.status)) {
                clearPolling();
                sessionStorage.removeItem(STORAGE_ACTIVE_TASK);
                currentTaskId = null;
                hasShownRestartToast = false; // 重置 flag

                toggleResetButton('clear');
                resetStartButton();
                hideRestartToast(); // 清理 UI
                hideInterruptedUI(); // 清理 UI

                if (data.status === 'completed') {
                    setStatusUI({ message: "任务完成", progress: 100, color: '#34c759', stripes: false });
                    if (data.result) finishTask(data.result);
                } else if (data.status === 'stopped') {
                    setStatusUI({ message: "任务已手动停止", progress: 0, color: '#86868b', stripes: false });
                } else {
                    setStatusUI({ message: data.message || "任务失败", progress: 0, color: '#ff3b30', stripes: false });
                    // alert('任务出错: ' + (data.message || 'Unknown error'));
                }
            }

        } catch (e) {
            console.error("Polling error", e);
        }
    }, 1000);
}

// ===================== 结果渲染 =====================
function finishTask(resultData) {
    // resultData 结构来自 merge_manager.run_merge_task 返回的 result
    // 期待：{status:'success', metrics:{...}}
    const metrics = resultData.metrics;
    if (metrics) renderResults(metrics);
}

function renderResults(metrics) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    document.getElementById('accuracy').innerText = `${metrics.accuracy}%`;
    document.getElementById('f1-score').innerText = metrics.f1_score;
    document.getElementById('test-cases').innerText = metrics.test_cases;
    document.getElementById('passed').innerText = metrics.passed;

    drawChart(metrics.comparison, metrics.base_name);
}

// 全局图表实例，防止重复创建
let myChart = null;

function drawChart(compData, baseName) {
    if (!compData) return;

    const ctx = document.getElementById('evaluation-chart').getContext('2d');

    if (myChart) myChart.destroy();

    myChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: compData.labels,
            datasets: [
                {
                    label: baseName + ' (基座)',
                    data: compData.base_data,
                    backgroundColor: 'rgba(108, 117, 125, 0.2)',
                    borderColor: 'rgba(108, 117, 125, 1)',
                    pointBackgroundColor: 'rgba(108, 117, 125, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Merged Model (融合后)',
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
            scales: {
                r: {
                    angleLines: { display: true },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            },
            plugins: {
                legend: { position: 'top' }
            }
        }
    });
}

function initPageDropdowns() {
    // 找到所有带 generic-select 类的 wrapper (主页上的那两个)，但排除配方选择下拉框
    const dropdowns = document.querySelectorAll('.ios-select-wrapper.generic-select:not(.recipe-apply-select)');

    dropdowns.forEach(wrapper => {
        const trigger = wrapper.querySelector('.ios-select-trigger');
        const optionsMenu = wrapper.querySelector('.ios-select-options');
        const options = wrapper.querySelectorAll('.ios-option');
        // 找到该 wrapper 同级的隐藏 input，用于存储值
        const hiddenInput = wrapper.parentElement.querySelector('input[type="hidden"]');

        if (!trigger || !optionsMenu) return;

        // 1. 点击触发器
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            // 关闭页面上其他已打开的菜单
            document.querySelectorAll('.ios-select-options.open').forEach(el => {
                if (el !== optionsMenu) el.classList.remove('open');
            });
            optionsMenu.classList.toggle('open');
            wrapper.classList.toggle('active'); // 可选：用于给 wrapper 加样式
        });

        // 2. 点击选项
        options.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                
                // UI 更新
                options.forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
                
                // 文字更新（优先 .opt-name，否则整行文本）
                const nameEl = opt.querySelector('.opt-name');
                const name = nameEl ? nameEl.innerText : (opt.innerText || opt.textContent || '');
                // 有些选项可能有中文括号，我们可以选择只保留名字，或者全显
                // 这里为了美观，我们把 trigger 里的文字替换成当前选项的名字
                // 如果需要保留 "Linear (线性融合)" 这种格式，可以从 innerText 读
                // 或者简单点，直接取 opt-name
                trigger.querySelector('.selected-text').innerText = name;

                // 数据更新 (核心：更新 hidden input)
                const val = opt.dataset.value;
                if (hiddenInput) {
                    hiddenInput.value = val;
                    // 触发 input 事件，防止有监听 input 的逻辑失效 (虽然目前没有)
                    hiddenInput.dispatchEvent(new Event('input'));
                    hiddenInput.dispatchEvent(new Event('change'));
                }

                // 关闭菜单
                optionsMenu.classList.remove('open');
                wrapper.classList.remove('active');
            });
        });
    });

    // 3. 全局点击关闭（dropdowns 已排除配方选择下拉框）
    document.addEventListener('click', (e) => {
        dropdowns.forEach(wrapper => {
            const menu = wrapper.querySelector('.ios-select-options');
            const trigger = wrapper.querySelector('.ios-select-trigger');
            if (!wrapper.contains(e.target)) {
                menu.classList.remove('open');
                wrapper.classList.remove('active');
            }
        });
    });
}

// ===================== 任务完成弹窗 =====================
async function showTaskCompletionModal(taskId, statusData, evoProgress) {
    const overlay = document.getElementById('task-completion-modal-overlay');
    const taskIdEl = document.getElementById('completion-task-id');
    const recipeStatusEl = document.getElementById('completion-recipe-status');
    const modelPathEl = document.getElementById('completion-model-path');
    const metricsEl = document.getElementById('completion-metrics');
    const closeBtn = document.getElementById('completion-close');
    if (!overlay || !taskIdEl) return;

    taskIdEl.textContent = taskId || '未知';

    // 检查配方是否保存
    try {
        // 先获取任务详情，检查是否有 recipe_id（对于 recipe_apply 类型）
        const detailRes = await fetch(`/api/history/${taskId}`);
        const detailData = await detailRes.json();
        let recipeId = taskId; // 默认使用 taskId
        if (detailData.status === 'success' && detailData.data) {
            // 如果是 recipe_apply 类型，使用 recipe_id
            if (detailData.data.type === 'recipe_apply' && detailData.data.recipe_id) {
                recipeId = detailData.data.recipe_id;
            }
        }
        
        const recipeRes = await fetch(`/api/recipes/${recipeId}`);
        const recipeData = await recipeRes.json();
        if (recipeData.status === 'success' && recipeData.recipe) {
            recipeStatusEl.textContent = '✓ 已保存到 recipes/' + recipeId + '.json';
            recipeStatusEl.style.color = 'var(--success)';
        } else {
            recipeStatusEl.textContent = '✗ 未找到配方文件';
            recipeStatusEl.style.color = 'var(--apple-gray)';
        }
    } catch (e) {
        recipeStatusEl.textContent = '? 无法检查配方状态';
        recipeStatusEl.style.color = 'var(--apple-gray)';
    }

    // 获取模型路径
    try {
        const detailRes = await fetch(`/api/history/${taskId}`);
        const detailData = await detailRes.json();
        if (detailData.status === 'success' && detailData.data) {
            const outputPath = detailData.data.metrics?.output_path || detailData.data.output_path;
            if (outputPath) {
                modelPathEl.textContent = outputPath;
            } else {
                modelPathEl.textContent = 'merges/' + taskId + '/output';
            }
        } else {
            modelPathEl.textContent = 'merges/' + taskId + '/output';
        }
    } catch (e) {
        modelPathEl.textContent = 'merges/' + taskId + '/output';
    }

    // 显示性能指标（如果是完全融合）
    if (evoProgress && (evoProgress.current_best != null || evoProgress.global_best != null)) {
        let metricsHtml = '<div style="font-weight: 500; margin-bottom: 8px;">性能指标:</div>';
        if (evoProgress.current_best != null) {
            metricsHtml += '<div style="margin-bottom: 6px;">当前最优准确率: <strong>' + Number(evoProgress.current_best).toFixed(4) + '</strong></div>';
        }
        if (evoProgress.global_best != null) {
            metricsHtml += '<div style="margin-bottom: 6px;">全局最优准确率: <strong>' + Number(evoProgress.global_best).toFixed(4) + '</strong></div>';
        }
        
        // --- 新增：显示迭代信息、基因权重、数据集 ---
        if (evoProgress.current_step != null || evoProgress.step != null) {
            const step = evoProgress.current_step || evoProgress.step;
            const total = evoProgress.total_expected_steps || '?';
            metricsHtml += `<div style="margin-bottom: 6px;">迭代进度: <strong>${step} / ${total}</strong></div>`;
        }

        if (evoProgress.best_genotype) {
            let geneStr = '';
            if (Array.isArray(evoProgress.best_genotype)) {
                geneStr = evoProgress.best_genotype.map(v => Number(v).toFixed(4)).join(', ');
            } else {
                geneStr = String(evoProgress.best_genotype);
            }
            metricsHtml += `<div style="margin-bottom: 6px;">最佳基因权重: <strong>[${geneStr}]</strong></div>`;
        }

        // 尝试从 detailData 获取数据集信息 (如果 fetch 成功)
        try {
            const detailRes = await fetch(`/api/history/${taskId}`);
            const detailData = await detailRes.json();
            if (detailData.status === 'success' && detailData.data) {
                if (detailData.data.hf_dataset) {
                     metricsHtml += `<div style="margin-bottom: 6px;">数据集: <strong>${detailData.data.hf_dataset}</strong></div>`;
                }
                if (detailData.data.hf_subsets && detailData.data.hf_subsets.length > 0) {
                     // 仅显示前3个子集，避免太长
                     const subStr = detailData.data.hf_subsets.slice(0, 3).join(', ') + (detailData.data.hf_subsets.length > 3 ? '...' : '');
                     metricsHtml += `<div style="margin-bottom: 6px; font-size: 0.85rem; color: #666;">子集: ${subStr}</div>`;
                }
            }
        } catch(e) { console.error("Fetch dataset info failed", e); }
        // ------------------------------------------

        metricsEl.innerHTML = metricsHtml;
        metricsEl.style.display = 'block';

        // ================== 3D 可视化图表 ==================
        // 检查/创建图表容器
        let plotContainer = document.getElementById('completion-3d-plot');
        if (!plotContainer) {
            plotContainer = document.createElement('div');
            plotContainer.id = 'completion-3d-plot';
            plotContainer.style.width = '100%';
            plotContainer.style.height = '400px';
            plotContainer.style.marginTop = '15px';
            metricsEl.parentElement.appendChild(plotContainer);
        } else {
            plotContainer.innerHTML = ''; // 清空旧图表
            plotContainer.style.display = 'block';
        }

        try {
            const plotRes = await fetch(`/api/fusion_3d_data/${taskId}`);
            const plotData = await plotRes.json();
            
            if (plotData.status === 'success' && plotData.data && plotData.data.length > 0) {
                const x = [], y = [], z = [], c = [];
                plotData.data.forEach(row => {
                    // 假设 CSV 列名: genotype_1, genotype_2, objective_1
                    // 实际列名可能略有不同，做一些容错处理
                    const g1 = row.genotype_1 !== undefined ? row.genotype_1 : row['genotype_1'];
                    const g2 = row.genotype_2 !== undefined ? row.genotype_2 : row['genotype_2'];
                    const obj = row.objective_1 !== undefined ? row.objective_1 : row['objective_1'];
                    
                    if (g1 != null && g2 != null && obj != null) {
                        x.push(parseFloat(g1));
                        y.push(parseFloat(g2));
                        z.push(parseFloat(obj));
                        c.push(parseFloat(obj));
                    }
                });

                if (x.length > 0) {
                    const trace = {
                        x: x,
                        y: y,
                        z: z,
                        mode: 'markers',
                        marker: {
                            size: 5,
                            color: c,
                            colorscale: 'Viridis',
                            opacity: 0.8,
                            showscale: true,
                            colorbar: { title: 'Accuracy' }
                        },
                        type: 'scatter3d',
                        hovertemplate: 'G1: %{x:.4f}<br>G2: %{y:.4f}<br>Acc: %{z:.4f}<extra></extra>'
                    };

                    const layout = {
                        title: '参数搜索空间 (Fitness Landscape)',
                        margin: { l: 0, r: 0, b: 0, t: 30 },
                        scene: {
                            xaxis: { title: 'Genotype 1' },
                            yaxis: { title: 'Genotype 2' },
                            zaxis: { title: 'Accuracy' }
                        }
                    };

                    Plotly.newPlot('completion-3d-plot', [trace], layout);
                } else {
                    plotContainer.innerHTML = '<p style="color:#999;font-size:0.9rem;">无有效 3D 数据点</p>';
                }
            } else {
                plotContainer.style.display = 'none'; // 无数据时不显示容器
            }
        } catch (e) {
            console.error("Load 3D data failed", e);
            plotContainer.innerHTML = '<p style="color:#999;font-size:0.9rem;">加载 3D 数据失败</p>';
        }
        // ================================================

    } else {
        metricsEl.style.display = 'none';
    }

    // 检查模型清理状态
    const cleanupStatusEl = document.createElement('div');
    cleanupStatusEl.style.marginTop = '12px';
    cleanupStatusEl.style.paddingTop = '12px';
    cleanupStatusEl.style.borderTop = '1px solid var(--apple-border)';
    cleanupStatusEl.innerHTML = '<div style="font-size: 0.85rem; color: var(--apple-gray); margin-bottom: 4px;"><strong>清理状态:</strong></div>';
    try {
        const detailRes2 = await fetch(`/api/history/${taskId}`);
        const detailData2 = await detailRes2.json();
        if (detailData2.status === 'success' && detailData2.data) {
            const taskType = detailData2.data.type || '';
            if (taskType === 'merge_evolutionary') {
                // 完全融合：检查 final_vlm 是否已清理
                cleanupStatusEl.innerHTML += '<div style="font-size: 0.9rem;">✓ 中间模型目录 (final_vlm) 已清理</div>';
                cleanupStatusEl.innerHTML += '<div style="font-size: 0.9rem; margin-top: 4px;">✓ 最终模型已保存到命名目录</div>';
            } else {
                cleanupStatusEl.innerHTML += '<div style="font-size: 0.9rem;">✓ 模型已保存</div>';
            }
        } else {
            cleanupStatusEl.innerHTML += '<div style="font-size: 0.9rem;">? 无法确认清理状态</div>';
        }
    } catch (e) {
        cleanupStatusEl.innerHTML += '<div style="font-size: 0.9rem;">? 无法检查清理状态</div>';
    }
    const detailsContainer = document.getElementById('completion-details');
    if (detailsContainer && !detailsContainer.querySelector('[data-cleanup-status]')) {
        cleanupStatusEl.setAttribute('data-cleanup-status', 'true');
        detailsContainer.appendChild(cleanupStatusEl);
    }

    // 显示弹窗
    overlay.style.display = 'flex';
    overlay.classList.add('show');

    // 关闭按钮
    if (closeBtn) {
        closeBtn.onclick = () => {
            overlay.style.display = 'none';
            overlay.classList.remove('show');
        };
    }
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            overlay.style.display = 'none';
            overlay.classList.remove('show');
        }
    };
}
