let progressInterval = null;
let pollTimer = null;
let chartComp = null;
let chartTemp = null;
let lastDashboard = null;
const POLL_INTERVAL_MS = 250;

function showToast(msg) {
    const t = document.getElementById("error-toast");
    t.innerText = msg;
    t.style.display = "block";
    setTimeout(() => { t.style.display = "none"; }, 4000);
}

function toggleFullScreen() {
    const doc = window.document;
    const docEl = doc.documentElement;
    const requestFullScreen = docEl.requestFullscreen || docEl.mozRequestFullScreen || docEl.webkitRequestFullScreen || docEl.msRequestFullscreen;
    const cancelFullScreen = doc.exitFullscreen || doc.mozCancelFullScreen || doc.webkitExitFullscreen || doc.msExitFullscreen;

    if(!doc.fullscreenElement && !doc.mozFullScreenElement && !doc.webkitFullscreenElement && !doc.msFullscreenElement) {
        requestFullScreen.call(docEl);
        document.getElementById('fullscreen-btn').innerText = "退出全屏";
    } else {
        cancelFullScreen.call(doc);
        document.getElementById('fullscreen-btn').innerText = "全屏模式";
    }
}

function readComposition() {
    const xs = [1, 2, 3, 4, 5, 6].map(i => parseFloat(document.getElementById(`x${i}`).value));
    if (xs.some(v => Number.isNaN(v))) {
        throw new Error("组分输入包含非数字。请检查六个组分输入框。");
    }
    const sum = xs.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1.0) > 1e-4) {
        throw new Error("组分和必须等于 1！当前为：" + sum.toFixed(4));
    }
    return xs;
}

function buildFeedstockPayload() {
    return {
        composition: readComposition(),
        source: document.getElementById("source-select").value,
        confidence: parseFloat(document.getElementById("confidence").value),
        wet_mass_flow_kgps: parseFloat(document.getElementById("wet_mass_flow").value)
    };
}

async function calculate() {
    let data;
    try {
        data = buildFeedstockPayload();
    } catch (err) {
        showToast(err.message);
        return;
    }

    document.getElementById("loading-container").style.display = "block";
    const progressBar = document.getElementById("progress-bar");
    const loadingText = document.getElementById("loading-text");
    progressBar.style.width = "0%";
    loadingText.innerText = "正在提交组分校正...";

    let progress = 0;
    clearInterval(progressInterval);
    progressInterval = setInterval(() => {
        progress = Math.min(96, progress + 24);
        progressBar.style.width = progress + "%";
    }, 180);

    try {
        const response = await fetch("/api/feedstock", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        const resData = await response.json();
        if (!resData.success) {
            throw new Error(resData.error || "提交失败");
        }
        clearInterval(progressInterval);
        progressBar.style.width = "100%";
        loadingText.innerText = "组分已更新，监控数据已刷新";
        updateDashboard(resData);
        showToast("组分校正已提交");
    } catch (e) {
        clearInterval(progressInterval);
        progressBar.style.width = "0%";
        loadingText.innerText = "提交失败";
        showToast("提交失败: " + e.message);
        console.error(e);
    } finally {
        setTimeout(() => {
            document.getElementById("loading-container").style.display = "none";
        }, 900);
    }
}

async function startMonitoring() {
    try {
        const res = await fetch("/api/start", { method: "POST" });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || "启动失败");
        updateDashboard(data);
        startPolling();
        showToast("监控已启动");
    } catch (err) {
        showToast("启动失败: " + err.message);
    }
}

async function stopMonitoring() {
    try {
        const res = await fetch("/api/stop", { method: "POST" });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || "暂停失败");
        updateDashboard(data);
        showToast("监控已暂停");
    } catch (err) {
        showToast("暂停失败: " + err.message);
    }
}

async function resetMonitoring() {
    try {
        const res = await fetch("/api/reset", { method: "POST" });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || "复位失败");
        updateDashboard(data);
        startPolling();
        showToast("实时仿真已复位");
    } catch (err) {
        showToast("复位失败: " + err.message);
    }
}

async function fetchDashboard() {
    try {
        const response = await fetch("/api/dashboard?limit=360", { cache: "no-store" });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || "无法读取 dashboard");
        updateDashboard(data);
    } catch (err) {
        const status = document.getElementById("status-feas");
        status.innerText = "连接异常";
        status.style.color = "#ff4d4f";
        status.style.textShadow = "0 0 5px #ff4d4f";
        console.error(err);
    }
}

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(fetchDashboard, POLL_INTERVAL_MS);
}

function updateDashboard(data) {
    lastDashboard = data;
    const feed = data.feedstock;
    const furnace = data.furnace;
    const preheater = data.preheater;
    const control = data.control;
    const health = data.health;

    document.getElementById("res-tg").innerText = `${fmt(control.Tg_ref_C, 1)} / ${fmt(control.Tg_cmd_C, 1)} °C`;
    document.getElementById("res-vg").innerText = `${fmt(control.vg_ref_mps, 2)} / ${fmt(control.vg_cmd_mps, 2)} m/s`;
    document.getElementById("res-tm").innerText = fmt(preheater.T_solid_out_C, 1) + " °C";
    document.getElementById("res-w").innerText = fmt(preheater.omega_out * 100, 2) + " %";
    document.getElementById("res-qreq").innerText = fmt(control.Q_aux_heat_kW, 2) + " kW";
    document.getElementById("res-qsup").innerText = fmt(control.fan_circulation_power_kW, 2) + " kW";
    document.getElementById("res-ttar").innerText = fmt(furnace.T_set_C, 1) + " °C";
    document.getElementById("res-wtar").innerText = fmt(control.omega_target * 100, 2) + " %";
    document.getElementById("res-tset").innerText = `${fmt(furnace.T_stack_C, 1)} °C / ${fmt(furnace.v_stack_mps, 2)} m/s`;
    document.getElementById("res-mdot").innerText = fmt(control.mdot_stack_available_kgps, 4) + " kg/s";

    const status = document.getElementById("status-feas");
    if (health.ok) {
        status.innerText = control.operator_feasible ? "NMPC运行" : "需人工关注";
        setValueColor(status, "good");
    } else {
        status.innerText = control.recovery_guard_active ? "恢复保护" : "告警";
        setValueColor(status, "bad");
    }

    const tempDeviation = furnace.temperature_error_C;
    const devNode = document.getElementById("status-dev");
    devNode.innerText = (tempDeviation >= 0 ? "+" : "") + fmt(tempDeviation, 1) + " °C";
    setValueColor(devNode, Math.abs(tempDeviation) < 8 ? "good" : Math.abs(tempDeviation) < 18 ? "warn" : "bad");

    const safetyNode = document.getElementById("status-safety");
    safetyNode.innerText = fmt(control.safety_margin_C, 1) + " °C";
    setValueColor(safetyNode, control.safety_margin_C > 18 ? "good" : control.safety_margin_C > 5 ? "warn" : "bad");

    const staleNode = document.getElementById("status-stale");
    staleNode.innerText = health.stale ? "暂停" : `t=${fmt(data.time_s, 1)}s / ${control.operator_source || "NMPC"}`;
    setValueColor(staleNode, health.stale ? "warn" : "good");

    const note = document.getElementById("res-nmpc-note");
    if (note) note.innerText = `${control.operator_source || "--"} · ${fmt(control.nmpc_last_solve_ms, 1)} ms · ${control.nmpc_candidates || "--"}候选`;

    renderCharts(feed.composition, data.history || []);
}

function updateUI(legacyResult) {
    // Compatibility hook for older /api/solve responses.
    if (legacyResult && legacyResult.dashboard) {
        updateDashboard(legacyResult.dashboard);
        return;
    }
    fetchDashboard();
}

function renderCharts(composition, history) {
    if (!chartComp) {
        chartComp = echarts.init(document.getElementById('chart-composition'));
    }
    if (!chartTemp) {
        chartTemp = echarts.init(document.getElementById('chart-temperature'));
    }

    const names = ['菜叶', '西瓜皮', '橙子皮', '肉', '杂项混合', '米饭'];
    const compOption = {
        tooltip: { trigger: 'item', formatter: '{b} : {c} ({d}%)' },
        series: [{
            name: '组分比例',
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: { borderRadius: 5, borderColor: '#050b14', borderWidth: 2 },
            label: { show: true, color: '#d8f8ff', formatter: '{b}\n{d}%' },
            data: names.map((name, i) => ({ value: composition[i], name }))
        }]
    };

    const recent = history.slice(-240);
    const times = recent.map(row => fmt(row.time_s, 0));
    const tempOption = {
        title: { text: '炉内与控制关键量实时趋势', left: 'center', top: 12, textStyle: { color: '#00d2ff', fontSize: 16 } },
        tooltip: { trigger: 'axis' },
        legend: { top: 42, textStyle: { color: '#d8f8ff' } },
        grid: { top: 85, left: '3%', right: '4%', bottom: '5%', containLabel: true },
        xAxis: { type: 'category', data: times, axisLabel: { color: '#88a4b3' } },
        yAxis: { type: 'value', axisLabel: { color: '#88a4b3' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
        series: [
            { name: 'T_avg 炉温', type: 'line', smooth: true, symbol: 'none', data: recent.map(row => round(row.T_avg_C, 1)) },
            { name: 'T_stack 烟囱', type: 'line', smooth: true, symbol: 'none', data: recent.map(row => round(row.T_stack_C, 1)) },
            { name: 'T_set 目标', type: 'line', smooth: true, symbol: 'none', data: recent.map(row => round(row.T_set_C, 1)) },
            { name: '安全下限', type: 'line', smooth: true, symbol: 'none', data: recent.map(row => round(row.T_compliance_min_C, 1)) },
            { name: 'Tg_cmd 指令', type: 'line', smooth: true, symbol: 'none', data: recent.map(row => round(row.Tg_cmd_C, 1)) },
            { name: '扰动 dTavg', type: 'line', smooth: true, symbol: 'none', data: recent.map(row => round(row.disturbance_Tavg_C || 0, 1)) }
        ]
    };

    chartComp.setOption(compOption);
    chartTemp.setOption(tempOption);
}

function setValueColor(node, kind) {
    const colors = {
        good: ["#00ff99", "0 0 5px #00ff99"],
        warn: ["#ffa500", "0 0 5px #ffa500"],
        bad: ["#ff4d4f", "0 0 5px #ff4d4f"]
    };
    const picked = colors[kind] || colors.good;
    node.style.color = picked[0];
    node.style.textShadow = picked[1];
}

function fmt(value, digits) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "--";
    return n.toFixed(digits);
}

function round(value, digits) {
    const factor = Math.pow(10, digits);
    return Math.round(Number(value) * factor) / factor;
}

window.addEventListener('resize', function() {
    if (chartComp) chartComp.resize();
    if (chartTemp) chartTemp.resize();
});

window.addEventListener('DOMContentLoaded', function() {
    fetchDashboard();
    startPolling();
});
