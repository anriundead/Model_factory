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
const STORAGE_ACTIVE_EVAL = 'mergekit_eval_active_task';

function initEval() {
    loadAllModels();
    setupEvalUI();
    setupDatasetSource();
    setupSidebar(); 
    setupDeleteModal();
    initSegmentedControls();
    setupTestsetRefresh();
    restoreActiveEvalTask();
}

async function restoreActiveEvalTask() {
    const candidates = [];
    const tid = sessionStorage.getItem(STORAGE_ACTIVE_EVAL);
    if (tid) candidates.push(tid);
    // 无会话 ID 时，或会话 ID 失效时，回退到历史记录中查找“正在运行/排队”的评估任务
    try {
        const r = await fetch('/api/test_history');
        const d = await r.json();
        const list = (d.history || []).filter(it => (it && it.type === 'eval_only'));
        list.forEach(it => {
            if (it && (it.status === 'running' || it.status === 'queued') && it.id) {
                candidates.push(it.id);
            }
        });
    } catch (e) {}

    const uniq = [...new Set(candidates.filter(Boolean))];
    for (const cid of uniq) {
        try {
            const r = await fetch(`/api/status/${cid}`);
            const s = await r.json();
            const status = s ? s.status : null;
            const isActive = s ? (s.is_active !== false) : false;
            const isEval = !!(s && (s.type === 'eval_only' || ((s.original_data || {}).type === 'eval_only') || s.eval_progress));
            if ((status === 'running' || status === 'queued') && isActive && isEval) {
                currentTaskId = cid;
                try { sessionStorage.setItem(STORAGE_ACTIVE_EVAL, currentTaskId); } catch (e) {}
                pollStatus(currentTaskId);
                return;
            }
        } catch (e) {}
    }

    // 没有可恢复的活跃任务：清理残留状态，确保进度条隐藏
    currentTaskId = null;
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    try { sessionStorage.removeItem(STORAGE_ACTIVE_EVAL); } catch (e) {}
    const progressContainer = document.querySelector('.progress-container');
    if (progressContainer) progressContainer.style.display = 'none';
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
let activeRepoFetchKey = null;
const TESTSETS_CACHE_KEY = 'testsets_cache_v3';
const HF_INFO_SESS_KEY = 'hf_info_sess_v1';
const HF_INFO_FETCH_MS = 5000;

function _hfInfoSessLoad(ds, sub, tsid) {
    try {
        const k = (ds || '') + '|' + (sub || '') + '|' + (tsid || '');
        const raw = sessionStorage.getItem(HF_INFO_SESS_KEY);
        const obj = raw ? JSON.parse(raw) : {};
        const e = obj[k];
        if (e && e.data && e.ts && (Date.now() - e.ts) < 3600000) return e.data;
    } catch (err) {}
    return null;
}
function _hfInfoSessSave(ds, sub, tsid, data) {
    try {
        const k = (ds || '') + '|' + (sub || '') + '|' + (tsid || '');
        const raw = sessionStorage.getItem(HF_INFO_SESS_KEY);
        const obj = raw ? JSON.parse(raw) : {};
        obj[k] = { ts: Date.now(), data };
        sessionStorage.setItem(HF_INFO_SESS_KEY, JSON.stringify(obj));
    } catch (err) {}
}

function applyHfInfoToEvalSubsetSplitUI(d, subsetVal, splitVal, t, fetchKey) {
    const subsetUl = document.getElementById('eval-subset-options');
    const splitUl = document.getElementById('eval-split-options');
    const subsetTrigger = document.getElementById('eval-subset-trigger');
    const splitTrigger = document.getElementById('eval-split-trigger');
    const subsetWrapper = subsetTrigger.closest('.ios-select-wrapper');
    const splitWrapper = splitTrigger.closest('.ios-select-wrapper');
    subsetWrapper.classList.remove('loading');
    splitWrapper.classList.remove('loading');
    if (activeRepoFetchKey !== fetchKey ||
        (document.getElementById('eval-testset-id').value || '') !== (t.testset_id || '') ||
        (document.getElementById('eval-hf-dataset').value || '') !== (t.hf_dataset || '')) {
        return false;
    }
    if (d.status !== 'success') return false;
    if (d.configs && d.configs.length) {
        const configs = d.configs.slice();
        const configsSet = new Set(configs);
        if (subsetVal && !configsSet.has(subsetVal)) configs.unshift(subsetVal);
        const nextSubset = subsetVal || (configs[0] || 'all');
        subsetUl.innerHTML = configs.map(c => '<li class="ios-option' + (c === nextSubset ? ' selected' : '') + '" data-value="' + c + '">' + c + '</li>').join('');
        subsetTrigger.querySelector('.selected-text').textContent = nextSubset || 'all';
        document.getElementById('eval-subset').value = nextSubset;
    } else {
        const nextSubset = subsetVal || 'all';
        subsetUl.innerHTML = '<li class="ios-option' + (nextSubset === 'all' ? ' selected' : '') + '" data-value="all">all</li>' + (subsetVal && subsetVal !== 'all' ? '<li class="ios-option selected" data-value="' + subsetVal + '">' + subsetVal + '</li>' : '');
        subsetTrigger.querySelector('.selected-text').textContent = nextSubset;
        document.getElementById('eval-subset').value = nextSubset;
    }
    if (d.splits && d.splits.length) {
        const splits = d.splits.slice();
        if (splitVal && !splits.includes(splitVal)) splits.unshift(splitVal);
        let nextSplit = splitVal;
        if (!nextSplit) nextSplit = splits.includes('test') ? 'test' : splits[0];
        splitUl.innerHTML = splits.map(s => '<li class="ios-option' + (s === nextSplit ? ' selected' : '') + '" data-value="' + s + '">' + s + '</li>').join('');
        splitTrigger.querySelector('.selected-text').textContent = nextSplit;
        document.getElementById('eval-split').value = nextSplit;
    } else {
        let defaultSplitsHtml = '<li class="ios-option' + (splitVal === 'all' ? ' selected' : '') + '" data-value="all">all</li>';
        defaultSplitsHtml += '<li class="ios-option' + (splitVal === 'train' ? ' selected' : '') + '" data-value="train">train</li>';
        defaultSplitsHtml += '<li class="ios-option' + (splitVal === 'validation' ? ' selected' : '') + '" data-value="validation">validation</li>';
        defaultSplitsHtml += '<li class="ios-option' + (splitVal === 'test' ? ' selected' : '') + '" data-value="test">test</li>';
        splitUl.innerHTML = defaultSplitsHtml;
        if (splitVal && !['all', 'train', 'validation', 'test'].includes(splitVal)) {
            splitUl.innerHTML = '<li class="ios-option selected" data-value="' + splitVal + '">' + splitVal + '</li>' + splitUl.innerHTML;
        }
        splitTrigger.querySelector('.selected-text').textContent = splitVal || 'all';
        document.getElementById('eval-split').value = splitVal || 'all';
    }
    return true;
}

function evalRefetchSplitsOnly(hfDataset, hfSubset, testsetId, currentSplitVal) {
    const splitUl = document.getElementById('eval-split-options');
    const splitTrigger = document.getElementById('eval-split-trigger');
    const splitWrapper = splitTrigger.closest('.ios-select-wrapper');
    const fetchKey = (testsetId || '') + '|' + hfDataset + '|sub:' + (hfSubset || '');
    activeRepoFetchKey = fetchKey;
    splitTrigger.querySelector('.selected-text').innerHTML = '<span class="loading-text"><span class="loading-dots"></span> 更新分割...</span>';
    splitWrapper.classList.add('loading');
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HF_INFO_FETCH_MS);
    fetch('/api/dataset/hf_info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            hf_dataset: hfDataset,
            hf_subset: hfSubset || undefined,
            testset_id: testsetId || undefined,
        }),
        signal: controller.signal,
    })
        .then(r => r.json())
        .then(d => {
            clearTimeout(timeoutId);
            splitWrapper.classList.remove('loading');
            if (activeRepoFetchKey !== fetchKey) return;
            _hfInfoSessSave(hfDataset, hfSubset, testsetId, d);
            if (d.status !== 'success' || !d.splits || !d.splits.length) {
                splitTrigger.querySelector('.selected-text').textContent = currentSplitVal || 'test';
                return;
            }
            const splits = d.splits.slice();
            let next = currentSplitVal;
            if (!next || !splits.includes(next)) next = splits.includes('test') ? 'test' : splits[0];
            splitUl.innerHTML = splits.map(s => '<li class="ios-option' + (s === next ? ' selected' : '') + '" data-value="' + s + '">' + s + '</li>').join('');
            splitTrigger.querySelector('.selected-text').textContent = next;
            document.getElementById('eval-split').value = next;
        })
        .catch(() => {
            clearTimeout(timeoutId);
            splitWrapper.classList.remove('loading');
            if (activeRepoFetchKey !== fetchKey) return;
            splitTrigger.querySelector('.selected-text').textContent = currentSplitVal || 'test';
        });
}

function loadTestsetsCache() {
    try {
        const raw = sessionStorage.getItem(TESTSETS_CACHE_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (e) {
        return null;
    }
}
function saveTestsetsCache(list) {
    try { sessionStorage.setItem(TESTSETS_CACHE_KEY, JSON.stringify(list || [])); } catch (e) {}
}
function setupTestsetRefresh() {
    const btn = document.getElementById('eval-testset-refresh');
    if (!btn) return;
    btn.addEventListener('click', () => loadTestsetRepoOptions(true));
}
async function loadTestsetRepoOptions(forceRefresh) {
    try {
        if (!forceRefresh) {
            const cached = loadTestsetsCache();
            const invalidCache = cached && cached.some(t => !t || !t.testset_id || String(t.testset_id).includes('/'));
            if (cached && cached.length && !invalidCache) {
                testsetsList = cached;
                renderTestsetRepoOptions();
                if (cached.some(t => !t.sample_count || Number(t.sample_count) <= 0)) {
                    setTimeout(() => loadTestsetRepoOptions(true), 0);
                }
                return;
            }
        }
        const res = await fetch('/api/testset/list?refresh=1');
        const data = await res.json();
        testsetsList = (data.testsets || []) || [];
        saveTestsetsCache(testsetsList);
        renderTestsetRepoOptions();
    } catch (e) {
        console.error(e);
        document.getElementById('eval-repo-options').innerHTML = '<li class="ios-option" style="color:#999;">加载失败</li>';
    }
}
function renderTestsetRepoOptions() {
        const ul = document.getElementById('eval-repo-options');
        ul.innerHTML = '';
        if (!testsetsList.length) {
            ul.innerHTML = '<li class="ios-option" style="color:#999;">暂无测试集</li>';
            document.getElementById('eval-repo-trigger').querySelector('.selected-text').textContent = '请选择测试集';
            document.getElementById('eval-testset-id').value = '';
            document.getElementById('eval-hf-dataset').value = '';
            return;
        }
        const prevSelectedId = document.getElementById('eval-testset-id').value || '';
        const selectedTestset = testsetsList.find(t => (t.testset_id || '') === prevSelectedId) || testsetsList[0];
        testsetsList.forEach((t, i) => {
            const li = document.createElement('li');
            li.className = 'ios-option' + ((selectedTestset && (t.testset_id || '') === (selectedTestset.testset_id || '')) ? ' selected' : '');
            li.dataset.testsetId = t.testset_id || '';
            li.dataset.hfDataset = t.hf_dataset || '';
            li.dataset.hfSubset = t.hf_subset || '';
            li.dataset.hfSplit = t.hf_split || 'train';
            const name = (t.name || t.hf_dataset || t.testset_id);
            const isVlmBench = !!t.is_vlm_benchmark;
            const badge = isVlmBench
                ? '<span class="testset-type-badge vlm">VLM</span>'
                : '<span class="testset-type-badge llm">LLM</span>';
            li.innerHTML = '<span class="opt-name">' + name + '</span>' + badge;
            li.addEventListener('click', () => selectRepoTestset(t));
            ul.appendChild(li);
        });
        const nextSelectedId = selectedTestset ? (selectedTestset.testset_id || '') : '';
        document.getElementById('eval-repo-trigger').querySelector('.selected-text').textContent = selectedTestset ? (selectedTestset.name || selectedTestset.hf_dataset || selectedTestset.testset_id) : '请选择测试集';
        document.getElementById('eval-testset-id').value = nextSelectedId;
        document.getElementById('eval-hf-dataset').value = selectedTestset ? (selectedTestset.hf_dataset || '') : '';
        try {
            const hint = document.getElementById('vlm-compat-hint');
            if (hint) hint.style.display = (selectedTestset && selectedTestset.is_vlm_benchmark) ? 'block' : 'none';
        } catch (e) {}
        if (selectedTestset && nextSelectedId && nextSelectedId !== prevSelectedId) {
            fillSubsetSplitFromTestset(selectedTestset);
        }
}

function selectRepoTestset(t) {
    document.getElementById('eval-repo-trigger').querySelector('.selected-text').textContent = t.name || t.hf_dataset || t.testset_id;
    document.getElementById('eval-testset-id').value = t.testset_id || '';
    document.getElementById('eval-hf-dataset').value = t.hf_dataset || '';
    try {
        const hint = document.getElementById('vlm-compat-hint');
        if (hint) {
            hint.style.display = t && t.is_vlm_benchmark ? 'block' : 'none';
        }
    } catch (e) {}
    fillSubsetSplitFromTestset(t);
    document.querySelectorAll('#eval-repo-options .ios-option').forEach((o, i) => o.classList.toggle('selected', o.dataset.testsetId === (t.testset_id || '')));
    document.getElementById('eval-repo-options').classList.remove('open');
}

function fillSubsetSplitFromTestset(t) {
    const subsetVal = (t.hf_subset || '').trim();
    const splitVal = (t.hf_split || 'train').trim();
    const subsetUl = document.getElementById('eval-subset-options');
    const splitUl = document.getElementById('eval-split-options');
    const subsetTrigger = document.getElementById('eval-subset-trigger');
    const splitTrigger = document.getElementById('eval-split-trigger');
    const subsetWrapper = subsetTrigger.closest('.ios-select-wrapper');
    const splitWrapper = splitTrigger.closest('.ios-select-wrapper');
    document.getElementById('eval-subset').value = subsetVal;
    document.getElementById('eval-split').value = splitVal;
    
    // 如果有 hf_dataset，显示加载状态
    if (t.hf_dataset) {
        const fetchKey = (t.testset_id || '') + '|' + (t.hf_dataset || '');
        activeRepoFetchKey = fetchKey;

        let instant = null;
        if (t.cached_configs && t.cached_splits && t.cached_configs.length && t.cached_splits.length) {
            instant = { status: 'success', configs: t.cached_configs, splits: t.cached_splits };
        } else {
            instant = _hfInfoSessLoad(t.hf_dataset, subsetVal, t.testset_id);
        }
        if (instant && instant.status === 'success') {
            applyHfInfoToEvalSubsetSplitUI(instant, subsetVal, splitVal, t, fetchKey);
            return;
        }

        subsetTrigger.querySelector('.selected-text').innerHTML = '<span class="loading-text"><span class="loading-dots"></span> 加载子集中...</span>';
        splitTrigger.querySelector('.selected-text').innerHTML = '<span class="loading-text"><span class="loading-dots"></span> 加载分割中...</span>';
        subsetWrapper.classList.add('loading');
        splitWrapper.classList.add('loading');
        subsetUl.innerHTML = '';
        splitUl.innerHTML = '';
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), HF_INFO_FETCH_MS);
        fetch('/api/dataset/hf_info', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                hf_dataset: t.hf_dataset,
                hf_subset: subsetVal || undefined,
                testset_id: t.testset_id || undefined,
            }),
            signal: controller.signal,
        })
            .then(r => r.json())
            .then(d => {
                clearTimeout(timeoutId);
                _hfInfoSessSave(t.hf_dataset, subsetVal, t.testset_id, d);
                if (!applyHfInfoToEvalSubsetSplitUI(d, subsetVal, splitVal, t, fetchKey)) {
                    setDefaultSubsetSplit(subsetVal, splitVal);
                }
            })
            .catch(() => {
                clearTimeout(timeoutId);
                if (activeRepoFetchKey !== fetchKey) return;
                subsetWrapper.classList.remove('loading');
                splitWrapper.classList.remove('loading');
                subsetTrigger.querySelector('.selected-text').textContent = '加载超时，已用默认值';
                splitTrigger.querySelector('.selected-text').textContent = '加载超时，已用默认值';
                setDefaultSubsetSplit(subsetVal, splitVal);
            });
    } else {
        // 没有 hf_dataset 时直接设置默认值
        setDefaultSubsetSplit(subsetVal, splitVal);
    }
    
    function setDefaultSubsetSplit(subsetVal, splitVal) {
        const nextSubset = subsetVal || 'all';
        subsetUl.innerHTML = '<li class="ios-option' + (nextSubset === 'all' ? ' selected' : '') + '" data-value="all">all</li>';
        splitUl.innerHTML = '<li class="ios-option' + (splitVal === 'all' ? ' selected' : '') + '" data-value="all">all</li><li class="ios-option' + (splitVal === 'train' ? ' selected' : '') + '" data-value="train">train</li><li class="ios-option' + (splitVal === 'validation' ? ' selected' : '') + '" data-value="validation">validation</li><li class="ios-option' + (splitVal === 'test' ? ' selected' : '') + '" data-value="test">test</li>';
        subsetTrigger.querySelector('.selected-text').textContent = nextSubset;
        splitTrigger.querySelector('.selected-text').textContent = splitVal || 'all';
        document.getElementById('eval-subset').value = nextSubset;
        document.getElementById('eval-split').value = splitVal || 'all';
        if (subsetVal) {
            const li = document.createElement('li');
            li.className = 'ios-option selected';
            li.dataset.value = subsetVal;
            li.textContent = subsetVal;
            subsetUl.appendChild(li);
            subsetUl.querySelector('[data-value="all"]').classList.remove('selected');
        }
        if (splitVal && !['all', 'train', 'validation', 'test'].includes(splitVal)) {
            const li = document.createElement('li');
            li.className = 'ios-option selected';
            li.dataset.value = splitVal;
            li.textContent = splitVal;
            splitUl.insertBefore(li, splitUl.firstChild);
        }
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
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), HF_INFO_FETCH_MS);
        const res = await fetch('/api/dataset/hf_info', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hf_dataset: hfId }), signal: controller.signal });
        const data = await res.json();
        clearTimeout(timeoutId);
        if (data.status === 'success') {
            customConfigs = data.configs || [];
            customSplits = data.splits || ['train', 'validation', 'test'];
            const subUl = document.getElementById('eval-custom-subset-options');
            const splitUl = document.getElementById('eval-custom-split-options');
            const initialSubset = customConfigs.length ? customConfigs[0] : 'all';
            subUl.innerHTML = (customConfigs.length ? '' : '<li class="ios-option selected" data-value="all">all</li>') + (customConfigs.map(c => '<li class="ios-option' + (c === initialSubset ? ' selected' : '') + '" data-value="' + c + '">' + c + '</li>').join(''));
            const baseSplits = customSplits.length ? customSplits : ['all', 'train', 'validation', 'test'];
            splitUl.innerHTML = baseSplits.map(s => '<li class="ios-option' + (s === 'test' ? ' selected' : '') + '" data-value="' + s + '">' + s + '</li>').join('');
            document.getElementById('eval-custom-subset-trigger').querySelector('.selected-text').textContent = initialSubset;
            document.getElementById('eval-custom-split-trigger').querySelector('.selected-text').textContent = 'test';
            document.getElementById('eval-custom-subset').value = initialSubset;
            document.getElementById('eval-custom-split').value = 'test';
        }
    } catch (e) { }
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
    if (!listEl) return;
    listEl.innerHTML = '';
    
    try {
        const res = await fetch('/api/test_history');
        const data = await res.json();
        
        if (!data.history || data.history.length === 0) {
            listEl.innerHTML = '<div style="padding:20px; text-align:center; color:#999; font-size:0.8rem;">暂无测试记录</div>';
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
            const displayName = item.model_name || item.custom_name || item.task_id || item.id || '';
            const datasetLabel = item.dataset || item.hf_dataset || '评测任务';
            const statusLabel = item.status || '';

            // 3. 构建 HTML
            div.innerHTML = `
                <div class="h-content">
                    <div class="h-title">
                        <i class="ri-flask-line" style="font-size:0.8rem; margin-right:4px; opacity:0.7;"></i>
                        ${displayName}
                    </div>
                    <div class="h-meta">
                        <span>${timeStr}</span>
                        <span>•</span>
                        <span>${datasetLabel}</span>
                        ${statusLabel ? '<span>•</span><span>' + statusLabel + '</span>' : ''}
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
        const kindTag = model.is_vlm ? 'VLM' : 'LLM';
        // VLM 紫色，LLM 蓝色，更直观的标签样式
        const tagStyle = model.is_vlm
            ? 'background: linear-gradient(135deg, rgba(139, 92, 246, 0.18), rgba(168, 85, 247, 0.22)); color: #7c3aed; border: 1px solid rgba(139, 92, 246, 0.25);'
            : 'background: linear-gradient(135deg, rgba(0, 113, 227, 0.12), rgba(59, 130, 246, 0.16)); color: #2563eb; border: 1px solid rgba(0, 113, 227, 0.2);';

        card.innerHTML = `
            <div style="display:flex; align-items:center; gap:10px;">
                <i class="${type === 'local' ? 'ri-hard-drive-2-line' : 'ri-git-merge-line'}" style="color:var(--apple-blue); font-size:1.2rem;"></i>
                <div style="text-align:left; flex:1;">
                    <h4 style="font-size:0.8rem; margin:0;">${model.name}</h4>
                    <p style="font-size:0.75rem; margin:0; opacity:0.6;">${type === 'local' ? 'Base Model' : 'Fused Model'} <span style="margin-left:6px; font-size:0.7rem; padding:3px 10px; border-radius:8px; font-weight:600; ${tagStyle}">${kindTag}</span></p>
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
            
            // 重置状态显示
            document.querySelector('.status-message').innerText = "准备就绪";
            const progressContainer = document.querySelector('.progress-container');
            if (progressContainer) progressContainer.style.display = 'none';

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
    const sampling = document.getElementById('eval-sampling').value;
    const numGpus = parseInt(document.getElementById('eval-num-gpus')?.value || '0', 10);
    const payload = {
        model_path: evalState.selectedModel.path,
        model_name: evalState.selectedModel.name,
        task_type: 'eval_only',
        limit: limit,
        sampling: sampling,
        num_gpus: numGpus
    };
    if (source === 'builtin') {
        payload.dataset = document.getElementById('eval-dataset').value;
    } else if (source === 'repo') {
        const rawTestsetId = document.getElementById('eval-testset-id').value;
        const rawHfDataset = document.getElementById('eval-hf-dataset').value || null;
        let resolvedTestsetId = rawTestsetId;
        if (!resolvedTestsetId || String(resolvedTestsetId).includes('/')) {
            const match = testsetsList.find(t => t && (t.hf_dataset || '') === (rawHfDataset || ''));
            if (match && match.testset_id) {
                resolvedTestsetId = match.testset_id;
            }
        }
        // 前端兼容性校验：VLM 基准必须选 VLM 模型
        try {
            const picked = testsetsList.find(t => t && (t.testset_id || '') === (resolvedTestsetId || ''));
            const testsetIsVlm = !!(picked && picked.is_vlm_benchmark);
            const modelIsVlm = !!(evalState.selectedModel && evalState.selectedModel.is_vlm);
            if (testsetIsVlm && !modelIsVlm) {
                showToast('该测试集为 VLM 基准，请选择带视觉塔的模型（如 Qwen2.5-VL）', 'warning');
                startBtn.disabled = false;
                startBtn.innerText = '开始测试';
                clearBtn.innerText = '清空';
                return;
            }
            if (!testsetIsVlm && modelIsVlm) {
                showToast('当前使用 VLM 模型运行 LLM 基准，将走文本评测管线。', 'info');
            }
        } catch (e) {}
        payload.testset_id = resolvedTestsetId || null;
        payload.hf_dataset = rawHfDataset || null;
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
            try { sessionStorage.setItem(STORAGE_ACTIVE_EVAL, currentTaskId); } catch (e) {}
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
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    try { sessionStorage.removeItem(STORAGE_ACTIVE_EVAL); } catch (e) {}
    resetUI();
    setStatusUI("测试已停止", 0, false, '#86868b');
}

function normalizeEvalStatusMessage(message, status) {
    const raw = (message || '').toString().trim();
    if (!raw || raw === 'None' || raw === 'null' || raw === 'undefined' || raw === '0' || raw === '0/0') {
        return status === 'queued' ? '排队中...' : '正在评估...';
    }
    if (raw.length > 80) return raw.slice(0, 80);
    return raw;
}
function pollStatus(taskId) {
    if (pollTimer) clearInterval(pollTimer);
    let pollCount = 0;
    let errorCount = 0;
    pollTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${taskId}`);
            const data = await res.json();
            errorCount = 0;
            const isActive = data && (data.is_active !== false);
            
            if ((data.status === 'running' || data.status === 'queued') && isActive) {
                // 确保进度条容器可见（用于刷新页面后的恢复）
                const progressContainer = document.querySelector('.progress-container');
                if (progressContainer && progressContainer.style.display === 'none') {
                    progressContainer.style.display = 'block';
                }

                let msg = data.status === 'queued' ? `排队中 (${data.queue_position || 0})` : normalizeEvalStatusMessage(data.message, data.status);
                let prog = data.status === 'queued' ? 100 : (data.progress || 0);
                const stripe = data.status === 'queued';
                if (data.status === 'running' && (!data.progress || data.progress <= 0)) {
                    prog = Math.min(5, Math.floor(pollCount / 6) + 1);
                }
                const ep = data.eval_progress || null;
                if (ep && typeof ep.current === 'number' && typeof ep.total === 'number') {
                    const eta = typeof ep.eta_seconds === 'number' ? ep.eta_seconds : null;
                    if (eta !== null && isFinite(eta) && eta > 0) {
                        const m = Math.floor(eta / 60);
                        const s = Math.floor(eta % 60);
                        msg = `${msg} (${ep.current}/${ep.total}) ETA ${m}:${s.toString().padStart(2,'0')}`;
                    } else {
                        msg = `${msg} (${ep.current}/${ep.total})`;
                    }
                    if (!data.progress && typeof ep.percent === 'number') {
                        const pct = Math.max(0, Math.min(100, ep.percent));
                        prog = Math.floor(30 + (65 * pct / 100));
                    }
                }
                
                setStatusUI(msg, prog, stripe);
                document.getElementById('start-eval').innerText = "正在运行...";
            } 
            else if (['completed', 'error', 'stopped', 'interrupted', 'failed'].includes(data.status) || !isActive) {
                clearInterval(pollTimer);
                pollTimer = null;
                currentTaskId = null;
                 try { sessionStorage.removeItem(STORAGE_ACTIVE_EVAL); } catch (e) {}
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
                    const isStopped = data.status === 'stopped';
                    setStatusUI(data.message || (isStopped ? "测试已停止" : "任务失败"), 0, false, isStopped ? '#86868b' : '#ff3b30');
                }
            }
        } catch(e) {
            errorCount += 1;
            // 连续失败视为任务状态不可恢复，避免进度条常驻
            if (errorCount >= 3) {
                if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
                currentTaskId = null;
                try { sessionStorage.removeItem(STORAGE_ACTIVE_EVAL); } catch (e2) {}
                resetUI();
            }
            console.error(e);
        }
        pollCount++;
    }, 1000);
}

function resetUI() {
    const startBtn = document.getElementById('start-eval');
    const clearBtn = document.getElementById('clear-eval');
    startBtn.disabled = !evalState.selectedModel;
    startBtn.innerText = "开始测试";
    clearBtn.innerText = "重置";
    clearBtn.disabled = !evalState.selectedModel;
    const progressContainer = document.querySelector('.progress-container');
    if (progressContainer) progressContainer.style.display = 'none';
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
            hintEl.textContent = '当前为占位结果，评测未返回数据。图中三角形仅为占位显示。';
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
    // 全为 0 时多边形会塌缩到中心不可见：基准线用较大占位值(12)保证可见，目标模型用 2 表示暂无数据
    const BASE_PLACEHOLDER = 12;
    const MERGED_PLACEHOLDER = 2;
    const displayBase = baseAllZero ? baseData.map(() => BASE_PLACEHOLDER) : baseData;
    const displayMerged = mergedAllZero ? mergedData.map(() => MERGED_PLACEHOLDER) : mergedData;

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
                                if (baseAllZero) return baseName + ': (暂无数据)';
                                return baseName + ': ' + displayBase[idx];
                            }
                            if (mergedAllZero) return 'Target Model: (暂无数据)';
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

    const openSidebar = () => {
        sidebar.classList.add('open');
        overlay.classList.add('show');
        loadHistoryList(); // 打开侧栏时刷新测试历史，保证看到最新列表
    };
    const closeSidebar = () => {
        sidebar.classList.remove('open');
        overlay.classList.remove('show');
    };
    const toggle = () => {
        if (sidebar.classList.contains('open')) closeSidebar();
        else openSidebar();
    };

    menuBtn.addEventListener('click', toggle);
    closeBtn.addEventListener('click', toggle);
    overlay.addEventListener('click', toggle);
    
    loadHistoryList();

    // 页面重新可见时刷新测试历史列表，避免多 tab 或 bfcache 后看到旧数据
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') loadHistoryList();
    });
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
            if (input && input.id === 'eval-subset') {
                const hfDataset = (document.getElementById('eval-hf-dataset') || {}).value;
                const tsid = (document.getElementById('eval-testset-id') || {}).value;
                const curSplit = (document.getElementById('eval-split') || {}).value || 'test';
                if (hfDataset) evalRefetchSplitsOnly(hfDataset, val, tsid, curSplit);
            }
        });
    });

    document.addEventListener('click', () => {
        document.querySelectorAll('.ios-select-options').forEach(el => {
            el.classList.remove('open');
        });
    });
}

function _ensureToastContainer() {
    let el = document.getElementById('toast-container');
    if (!el) {
        el = document.createElement('div');
        el.id = 'toast-container';
        el.className = 'toast-container';
        document.body.appendChild(el);
    }
    return el;
}

function showToast(message, type) {
    const container = _ensureToastContainer();
    const toast = document.createElement('div');
    toast.className = 'toast ' + (type || 'info');
    toast.textContent = message || '';
    container.appendChild(toast);
    setTimeout(() => {
        try {
            toast.style.transition = 'all 0.2s ease';
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-6px)';
        } catch (e) {}
    }, 2600);
    setTimeout(() => {
        try { toast.remove(); } catch (e) {}
    }, 3000);
}
