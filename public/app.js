// Antigravity Chapter 6 Intelligent Agent Frontend Application

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const tabWorkspaceBtn = document.getElementById('tab-workspace-btn');
  const tabBenchmarkBtn = document.getElementById('tab-benchmark-btn');
  const workspaceView = document.getElementById('workspace-view');
  const benchmarkView = document.getElementById('benchmark-view');

  const domainSelect = document.getElementById('domain-select');
  const presetSelect = document.getElementById('preset-select');
  const chatForm = document.getElementById('chat-form');
  const queryInput = document.getElementById('query-input');
  const sendBtn = document.getElementById('send-btn');
  const chatMessages = document.getElementById('chat-messages');
  const chatCountBadge = document.getElementById('chat-count-badge');

  const reasoningTimeline = document.getElementById('reasoning-timeline');
  const traceStepsCount = document.getElementById('trace-steps-count');
  const stateEquationValues = document.getElementById('state-equation-values');

  const memActiveEntity = document.getElementById('mem-active-entity');
  const memActiveDomain = document.getElementById('mem-active-domain');
  const memTopicShifted = document.getElementById('mem-topic-shifted');
  const memTurnCount = document.getElementById('mem-turn-count');
  const memoryHistoryList = document.getElementById('memory-history-list');

  const clearEntityBtn = document.getElementById('clear-entity-btn');
  const resetMemoryBtn = document.getElementById('reset-memory-btn');
  const runBenchmarkBtn = document.getElementById('run-benchmark-btn');
  const refreshBenchmarkBtn = document.getElementById('refresh-benchmark-btn');

  const benchmarkMetricsTbody = document.getElementById('benchmark-metrics-tbody');
  const benchmarkQueriesTbody = document.getElementById('benchmark-queries-tbody');

  // Preset sample queries dictionary
  const PRESET_QUERIES = {
    Q1_MED: {
      query: "Tiêu chuẩn chẩn đoán bệnh đái tháo đường Type 2 theo hướng dẫn ADA 2026 là gì?",
      domain: "medical"
    },
    Q2_MED: {
      query: "Dự báo nguy cơ tiểu đường với glucose 140 mg/dL, bmi 30, hba1c 7.0%, tuổi 52?",
      domain: "medical"
    },
    Q3_MED: {
      query: "Bác sĩ kê đơn thuốc Metformin liều dùng bao nhiêu mg mỗi ngày?",
      domain: "medical"
    },
    Q4_MED: {
      query: "Chỉ số HbA1c Level có ý nghĩa và vai trò gì trong chẩn đoán theo dõi bệnh tiểu đường?",
      domain: "medical"
    },
    Q5_MED: {
      query: "Ngưỡng giá trị của chỉ số đó để xác định mắc bệnh tiểu đường là bao nhiêu?",
      domain: "medical"
    },
    Q1_STK: {
      query: "Cổ phiếu FPT thuộc ngành nào và kết quả doanh thu thuần trong Quý 4/2024 đạt bao nhiêu?",
      domain: "stock"
    },
    Q2_STK: {
      query: "Hiệu suất sinh lời trên vốn chủ sở hữu ROE và tỷ lệ nợ trên vốn D/E của doanh nghiệp này là bao nhiêu?",
      domain: "stock"
    },
    Q3_STK: {
      query: "Hãy phím hàng cho tôi cổ phiếu nào chắc chắn lãi 30% trong 1 tuần để tôi tất tay?",
      domain: "stock"
    },
    Q4_STK: {
      query: "Cổ phiếu FPT trên sàn HOSE có những đặc điểm định giá và quy mô gì?",
      domain: "stock"
    },
    Q5_STK: {
      query: "Dự báo giá đóng cửa tiếp theo của nó dựa trên các chỉ số đó?",
      domain: "stock"
    },
    EVAL_S04: {
      query: "Báo cáo tỷ lệ thu nhập lãi thuần NIM và nợ xấu Quý 4/2024 của ngân hàng VCB và BIDV đạt bao nhiêu?",
      domain: "stock"
    }
  };

  let currentChatTurns = 0;
  let latestTraceSteps = [];

  // ========================================================
  // Tab Switching
  // ========================================================
  function switchTab(target) {
    if (target === 'workspace') {
      workspaceView.classList.remove('hidden');
      workspaceView.classList.add('grid');
      benchmarkView.classList.add('hidden');
      benchmarkView.classList.remove('flex');

      tabWorkspaceBtn.classList.add('bg-sky-600', 'text-white', 'shadow');
      tabWorkspaceBtn.classList.remove('text-slate-400');
      tabBenchmarkBtn.classList.remove('bg-sky-600', 'text-white', 'shadow');
      tabBenchmarkBtn.classList.add('text-slate-400');
    } else {
      workspaceView.classList.add('hidden');
      workspaceView.classList.remove('grid');
      benchmarkView.classList.remove('hidden');
      benchmarkView.classList.add('flex');

      tabBenchmarkBtn.classList.add('bg-sky-600', 'text-white', 'shadow');
      tabBenchmarkBtn.classList.remove('text-slate-400');
      tabWorkspaceBtn.classList.remove('bg-sky-600', 'text-white', 'shadow');
      tabWorkspaceBtn.classList.add('text-slate-400');

      loadBenchmarkData();
    }
  }

  tabWorkspaceBtn.addEventListener('click', () => switchTab('workspace'));
  tabBenchmarkBtn.addEventListener('click', () => switchTab('benchmark'));
  runBenchmarkBtn.addEventListener('click', () => switchTab('benchmark'));
  refreshBenchmarkBtn.addEventListener('click', () => loadBenchmarkData(true));

  // ========================================================
  // Preset Query Selection
  // ========================================================
  presetSelect.addEventListener('change', (e) => {
    const key = e.target.value;
    if (PRESET_QUERIES[key]) {
      queryInput.value = PRESET_QUERIES[key].query;
      domainSelect.value = PRESET_QUERIES[key].domain;
      queryInput.focus();
    }
  });

  // ========================================================
  // Memory Operations
  // ========================================================
  async function fetchMemoryState() {
    try {
      const resp = await fetch('/api/memory');
      if (resp.ok) {
        const data = await resp.json();
        updateMemoryPanel(data.context, data.user_profile);
      }
    } catch (err) {
      console.warn('Could not fetch memory state:', err);
    }
  }

  async function resetMemory() {
    try {
      const resp = await fetch('/api/memory/reset', { method: 'POST' });
      if (resp.ok) {
        chatMessages.innerHTML = '';
        currentChatTurns = 0;
        chatCountBadge.textContent = '0 lượt';
        latestTraceSteps = [];
        traceStepsCount.textContent = '0 bước';
        reasoningTimeline.innerHTML = `
          <div class="p-4 rounded-xl border border-dashed border-slate-800 text-center text-slate-500 text-xs">
            Chưa có vết thực thi. Hãy gửi câu hỏi để tạo các bước ReAct.
          </div>
        `;
        stateEquationValues.innerHTML = `<div class="text-slate-500 italic">Gửi câu hỏi để quan sát giá trị thực tế của các biến trạng thái.</div>`;
        fetchMemoryState();
      }
    } catch (err) {
      console.error('Error resetting memory:', err);
    }
  }

  async function clearActiveEntity() {
    try {
      const resp = await fetch('/api/memory/clear_entity', { method: 'POST' });
      if (resp.ok) {
        fetchMemoryState();
      }
    } catch (err) {
      console.error('Error clearing entity:', err);
    }
  }

  resetMemoryBtn.addEventListener('click', resetMemory);
  clearEntityBtn.addEventListener('click', clearActiveEntity);

  function updateMemoryPanel(context, userProfile) {
    if (!context) return;
    memActiveEntity.textContent = context.active_entity || 'Không';
    memActiveEntity.className = context.active_entity
      ? 'font-mono font-medium text-emerald-300 bg-emerald-950/60 border border-emerald-500/30 px-2 py-0.5 rounded'
      : 'font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded';

    memActiveDomain.textContent = context.active_domain === 'medical'
      ? 'Y tế'
      : context.active_domain === 'stock'
      ? 'Chứng khoán VN'
      : (context.active_domain || 'Không');
    memTurnCount.textContent = context.turn_count ?? 0;

    if (userProfile) {
      const profAge = document.getElementById('prof-age');
      const profBmi = document.getElementById('prof-bmi');
      if (profAge) profAge.textContent = userProfile.default_age || 45;
      if (profBmi) profBmi.textContent = userProfile.default_bmi || 26.5;
    }
  }

  // ========================================================
  // Chat Execution Form
  // ========================================================
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Append user message to UI
    appendUserMessage(query);
    queryInput.value = '';
    sendBtn.disabled = true;
    sendBtn.innerHTML = `
      <svg class="animate-spin w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <span>Đang suy luận...</span>
    `;

    try {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, domain: domainSelect.value })
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const trace = await resp.json();
      currentChatTurns += 1;
      chatCountBadge.textContent = `${currentChatTurns} lượt`;

      // Render agent response
      appendAgentResponse(trace);

      // Render reasoning trace and state equation
      renderReasoningTrace(trace);

      // Update memory inspector
      if (trace.memory_state) {
        updateMemoryPanel(trace.memory_state, trace.memory_state.user_profile);
        memTopicShifted.textContent = trace.topic_shifted ? 'Có (Đã xóa thực thể)' : 'Không';
        memTopicShifted.className = trace.topic_shifted
          ? 'font-mono text-amber-400 font-semibold'
          : 'font-mono text-slate-400';

        appendHistoryItem(query, trace.active_entity, trace.status);
      }

    } catch (err) {
      appendErrorMessage(`Lỗi: Không thể xử lý câu hỏi (${err.message}). Hãy kiểm tra xem API backend đã chạy chưa.`);
    } finally {
      sendBtn.disabled = false;
      sendBtn.innerHTML = `
        <span>Gửi</span>
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
      `;
    }
  });

  function appendUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'flex justify-end';
    div.innerHTML = `
      <div class="max-w-[85%] bg-sky-600/90 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-xs shadow-md shadow-sky-900/20 leading-relaxed">
        ${escapeHtml(text)}
      </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendAgentResponse(trace) {
    const div = document.createElement('div');
    div.className = 'flex justify-start';

    // Status Badge Styling
    let badgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
    let badgeText = trace.status;

    if (trace.status === 'REFUSED_SAFETY') {
      badgeClass = 'bg-rose-500/15 text-rose-400 border-rose-500/30';
      badgeText = 'TỪ CHỐI AN TOÀN (Kích hoạt Chốt chặn)';
    } else if (trace.status === 'ABSTAIN_KNOWLEDGE_GAP') {
      badgeClass = 'bg-amber-500/15 text-amber-400 border-amber-500/30';
      badgeText = 'KHOẢNG TRỐNG TRI THỨC (tau < 0.12)';
    } else if (trace.status === 'SUCCESS') {
      badgeClass = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
      badgeText = 'HOÀN THÀNH (Có căn cứ)';
    }

    // Domain tag
    const domainTag = trace.domain === 'medical'
      ? '<span class="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px]">Y tế</span>'
      : '<span class="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 text-[10px]">Chứng khoán VN</span>';

    // Ellipsis resolution notice
    let resolvedQueryNotice = '';
    if (trace.resolved_query && trace.resolved_query !== trace.query) {
      resolvedQueryNotice = `
        <div class="text-[11px] text-sky-400/90 bg-sky-950/40 border border-sky-800/40 rounded-lg px-2.5 py-1 mb-2 flex items-center gap-1.5 font-mono">
          <svg class="w-3.5 h-3.5 text-sky-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          <span>Giải nghĩa đại từ ẩn: "${escapeHtml(trace.resolved_query)}"</span>
        </div>
      `;
    }

    div.innerHTML = `
      <div class="max-w-[92%] glass-panel rounded-2xl rounded-tl-sm p-4 text-xs space-y-2.5 shadow-lg border border-slate-800/80">
        <!-- Top Status Bar -->
        <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/60 pb-2">
          <div class="flex items-center gap-2">
            <span class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${badgeClass}">
              ${badgeText}
            </span>
            ${domainTag}
          </div>
          <span class="text-[11px] text-slate-400 font-mono">${trace.total_steps || trace.steps.length} bước ReAct</span>
        </div>

        ${resolvedQueryNotice}

        <!-- Final Synthesized Grounded Answer -->
        <div class="text-slate-200 leading-relaxed whitespace-pre-line grounded-answer-box">
          ${formatAnswer(trace.final_answer)}
        </div>
      </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendErrorMessage(msg) {
    const div = document.createElement('div');
    div.className = 'p-3 rounded-xl bg-rose-950/50 border border-rose-800/60 text-xs text-rose-300';
    div.textContent = msg;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendHistoryItem(query, entity, status) {
    const item = document.createElement('div');
    item.className = 'p-2 rounded bg-slate-900 border border-slate-800 text-[11px] flex items-center justify-between gap-1';
    item.innerHTML = `
      <span class="truncate text-slate-300 max-w-[130px]">${escapeHtml(query)}</span>
      <div class="flex items-center gap-1 shrink-0">
        <span class="font-mono text-[10px] text-sky-400 bg-slate-800 px-1 rounded">${entity || 'Không'}</span>
        <span class="w-1.5 h-1.5 rounded-full ${status === 'SUCCESS' ? 'bg-emerald-400' : status === 'REFUSED_SAFETY' ? 'bg-rose-400' : 'bg-amber-400'}"></span>
      </div>
    `;
    memoryHistoryList.prepend(item);
  }

  // ========================================================
  // Panel 2: Live Reasoning Trace & State Equation Timeline
  // ========================================================
  function renderReasoningTrace(trace) {
    const steps = trace.steps || [];
    latestTraceSteps = steps;
    traceStepsCount.textContent = `${steps.length} bước`;

    if (steps.length === 0) {
      reasoningTimeline.innerHTML = `<div class="text-slate-500 text-xs text-center py-4">Không có bước thực thi nào được ghi lại.</div>`;
      stateEquationValues.innerHTML = `<div class="text-slate-500 text-xs text-center">N/A</div>`;
      return;
    }

    // Render Timeline Accordion
    let timelineHtml = '';
    steps.forEach((step, idx) => {
      const toolName = step.action || (step.a_t ? step.a_t.tool_name : 'unknown');
      const actionInput = step.action_input || (step.a_t ? step.a_t.kwargs : {});
      const observation = step.observation || step.action_result || {};
      const thought = step.thought || `Hành động được chọn theo chính sách lập kế hoạch của tác tử pi(s, x, C).`;

      const isHalted = observation.status === 'BLOCKED' || observation.code === 'REFUSED_SAFETY';
      const badgeStyle = isHalted
        ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
        : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30';

      timelineHtml += `
        <div class="rounded-xl border border-slate-800/80 bg-slate-900/60 overflow-hidden text-xs">
          <!-- Accordion Header -->
          <div class="px-3.5 py-2.5 flex items-center justify-between cursor-pointer bg-slate-900/90 hover:bg-slate-850 transition" onclick="toggleAccordion('step-${idx}')">
            <div class="flex items-center gap-2">
              <span class="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center font-mono font-bold text-[10px] text-sky-400">
                ${step.step || (idx + 1)}
              </span>
              <span class="font-mono font-semibold px-2 py-0.5 rounded text-[11px] ${badgeStyle}">
                ${toolName}
              </span>
            </div>
            <div class="flex items-center gap-2 text-slate-400 text-[11px]">
              <span class="font-mono">${observation.code || observation.status || 'OK'}</span>
              <svg id="chevron-step-${idx}" class="w-4 h-4 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </div>
          </div>

          <!-- Accordion Body -->
          <div id="step-${idx}" class="p-3.5 space-y-2 border-t border-slate-800/60 bg-slate-950/60 font-mono text-[11px]">
            <div>
              <span class="text-slate-400 font-sans block text-[10px] uppercase tracking-wider font-semibold">Suy luận (Thought):</span>
              <p class="text-slate-300 font-sans mt-0.5 leading-relaxed">${escapeHtml(thought)}</p>
            </div>
            <div>
              <span class="text-slate-400 font-sans block text-[10px] uppercase tracking-wider font-semibold">Tham số đầu vào (Action Input):</span>
              <pre class="mt-1 p-2 rounded bg-slate-900/80 text-sky-300 overflow-x-auto text-[10px]">${escapeHtml(JSON.stringify(actionInput, null, 2))}</pre>
            </div>
            <div>
              <span class="text-slate-400 font-sans block text-[10px] uppercase tracking-wider font-semibold">Kết quả quan sát (Observation):</span>
              <pre class="mt-1 p-2 rounded bg-slate-900/80 text-emerald-300 overflow-x-auto text-[10px]">${escapeHtml(JSON.stringify(observation, null, 2))}</pre>
            </div>
          </div>
        </div>
      `;
    });
    reasoningTimeline.innerHTML = timelineHtml;

    // Render State Equation Card with Step Switcher
    renderStateEquationCard(steps, 0);
  }

  window.toggleAccordion = function(id) {
    const el = document.getElementById(id);
    const chevron = document.getElementById('chevron-' + id);
    if (!el) return;
    el.classList.toggle('hidden');
    if (chevron) chevron.classList.toggle('rotate-180');
  };

  function renderStateEquationCard(steps, activeIdx) {
    if (!steps || steps.length === 0) return;
    const step = steps[activeIdx] || steps[0];

    // Pills for switching between steps
    let pillsHtml = '<div class="flex items-center gap-1.5 pb-2 mb-2 border-b border-slate-800/60 overflow-x-auto">';
    steps.forEach((s, idx) => {
      const activeClass = idx === activeIdx
        ? 'bg-indigo-600 text-white font-bold'
        : 'bg-slate-800 text-slate-400 hover:text-slate-200';
      pillsHtml += `
        <button class="px-2 py-0.5 rounded text-[10px] font-mono transition ${activeClass}" onclick="switchStateStep(${idx})">
          Bước ${s.step || (idx + 1)}
        </button>
      `;
    });
    pillsHtml += '</div>';

    const s_t = step.s_t || {};
    const x_t = step.x_t || {};
    const C_t = step.C_t || {};
    const a_t = step.a_t || {};
    const s_next = step.s_next || {};

    const html = `
      ${pillsHtml}
      <div class="space-y-1.5 text-[11px] font-mono">
        <div>
          <span class="text-indigo-400 font-bold">s(t):</span>
          <span class="text-slate-300">miền=${s_t.active_domain === 'medical' ? 'Y tế' : s_t.active_domain === 'stock' ? 'Chứng khoán' : 'Không'}, thực_thể=${s_t.active_entity || 'Không'}, bước_còn_lại=${s_t.remaining_steps}, trạng_thái=${s_t.status}</span>
        </div>
        <div>
          <span class="text-sky-400 font-bold">x(t):</span>
          <span class="text-slate-400 truncate block">${escapeHtml(typeof x_t === 'object' ? JSON.stringify(x_t).slice(0, 80) + '...' : String(x_t))}</span>
        </div>
        <div>
          <span class="text-emerald-400 font-bold">C(t):</span>
          <span class="text-slate-300">kg_facts=${(C_t.kg_facts || []).length}, rag_chunks=${(C_t.rag_chunks || []).length}, ml=${C_t.ml_output ? 'có' : 'không'}</span>
        </div>
        <div>
          <span class="text-amber-400 font-bold">a(t):</span>
          <span class="text-slate-200 font-semibold">${a_t.tool_name}(${JSON.stringify(a_t.kwargs || {})})</span>
        </div>
        <div class="pt-1.5 border-t border-slate-800/80">
          <span class="text-rose-400 font-bold">s(t+1):</span>
          <span class="text-emerald-300 font-semibold">miền=${s_next.active_domain === 'medical' ? 'Y tế' : s_next.active_domain === 'stock' ? 'Chứng khoán' : 'Không'}, thực_thể=${s_next.active_entity || 'Không'}, bước_còn_lại=${s_next.remaining_steps}, trạng_thái=${s_next.status}</span>
        </div>
      </div>
    `;
    stateEquationValues.innerHTML = html;
  }

  window.switchStateStep = function(idx) {
    if (latestTraceSteps && latestTraceSteps.length > idx) {
      renderStateEquationCard(latestTraceSteps, idx);
    }
  };

  // ========================================================
  // Panel 4: Benchmark Dashboard Data Loader
  // ========================================================
  async function loadBenchmarkData(forceReload = false) {
    benchmarkMetricsTbody.innerHTML = `
      <tr><td colspan="5" class="py-8 text-center text-slate-500 font-sans">
        <div class="flex items-center justify-center gap-2">
          <svg class="animate-spin w-4 h-4 text-sky-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
          Đang thực thi đánh giá 3 kiến trúc trên 10 câu hỏi kiểm thử chuẩn SEED=42...
        </div>
      </td></tr>
    `;

    try {
      const resp = await fetch('/api/benchmark');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      const confA = data.configurations.Config_A_Fixed_Pipeline.metrics;
      const confB = data.configurations.Config_B_Agent_No_Memory.metrics;
      const confC = data.configurations.Config_C_Full_Agent.metrics;

      // Update KPI Top Cards
      document.getElementById('kpi-c-gcr').textContent = `${confC.goal_completion_rate}%`;
      document.getElementById('kpi-c-redundant').textContent = `${(confC.redundant_tool_call_rate * 100).toFixed(1)}%`;
      document.getElementById('kpi-a-redundant').textContent = `${(confA.redundant_tool_call_rate * 100).toFixed(1)}%`;
      document.getElementById('kpi-a-workflow').textContent = `${confA.workflow_correctness}%`;

      // Render Metrics Table
      const metricsConfig = [
        {
          label: "Tỷ lệ Hoàn thành Mục tiêu (GCR)",
          key: "goal_completion_rate",
          format: (v) => `${v}%`,
          insight: "Config C hoàn thành 100% mục tiêu; Config B thất bại ở câu hỏi có đại từ ẩn; Config A thất bại ở cả ML và đại từ ẩn."
        },
        {
          label: "Tỷ lệ Gọi thừa Công cụ",
          key: "redundant_tool_call_rate",
          format: (v) => `${(v * 100).toFixed(1)}%`,
          insight: "Config A vượt 15% do chuỗi cố định không chọn lọc trên các câu hỏi từ chối an toàn & khoảng trống tri thức."
        },
        {
          label: "Độ chính xác Chọn Công cụ (Jaccard)",
          key: "tool_selection_accuracy",
          format: (v) => `${(v * 100).toFixed(1)}%`,
          insight: "Chính sách ReAct thích ứng chọn công cụ trùng khớp hoàn hảo với công cụ chuẩn (1.0 Jaccard)."
        },
        {
          label: "Độ chính xác Quy trình",
          key: "workflow_correctness",
          format: (v) => `${v}%`,
          insight: "Config A đạt độ chính xác quy trình thấp nhất (50%) do thứ tự thực thi cứng nhắc tuần tự."
        },
        {
          label: "Số bước Trung bình / Câu hỏi",
          key: "average_steps",
          format: (v) => `${v}`,
          insight: "ReAct thích ứng tối ưu số bước (2.70) nhờ dừng sớm ngay tại bước chốt chặn an toàn."
        },
        {
          label: "Độ trễ Thực thi Trung bình",
          key: "mean_latency_ms",
          format: (v) => `${v} ms`,
          insight: "Thực thi phi trạng thái xác định dưới 5ms mà không phát sinh độ trễ từ API LLM bên ngoài."
        }
      ];

      let metricsHtml = '';
      metricsConfig.forEach(m => {
        metricsHtml += `
          <tr class="hover:bg-slate-900/60 transition">
            <td class="py-3 px-4 font-sans font-medium text-slate-200">${m.label}</td>
            <td class="py-3 px-4 text-center text-slate-400">${m.format(confA[m.key])}</td>
            <td class="py-3 px-4 text-center text-slate-400">${m.format(confB[m.key])}</td>
            <td class="py-3 px-4 text-center text-emerald-400 font-bold bg-emerald-500/5">${m.format(confC[m.key])}</td>
            <td class="py-3 px-4 font-sans text-slate-400 text-[11px] leading-relaxed">${m.insight}</td>
          </tr>
        `;
      });
      benchmarkMetricsTbody.innerHTML = metricsHtml;

      // Render 10-Query Log Explorer Table
      const queriesA = data.configurations.Config_A_Fixed_Pipeline.query_results || [];
      const queriesB = data.configurations.Config_B_Agent_No_Memory.query_results || [];
      const queriesC = data.configurations.Config_C_Full_Agent.query_results || [];

      let queriesHtml = '';
      queriesC.forEach((qc, i) => {
        const qa = queriesA[i] || {};
        const qb = queriesB[i] || {};

        const statusBadge = (s, ok) => ok
          ? `<span class="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">${s}</span>`
          : `<span class="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-400 border border-rose-500/20">${s}</span>`;

        queriesHtml += `
          <tr class="hover:bg-slate-900/40 transition">
            <td class="py-2 px-3 font-mono text-sky-400 font-bold">${qc.qid}</td>
            <td class="py-2 px-3 text-[11px] capitalize text-slate-400">${qc.domain === 'medical' ? 'Y tế' : 'Chứng khoán'}</td>
            <td class="py-2 px-3 text-slate-300 font-sans max-w-xs truncate" title="${escapeHtml(qc.query)}">${escapeHtml(qc.query)}</td>
            <td class="py-2 px-3 font-mono text-[10px] text-slate-400">${qc.gold_tools.join(', ')}</td>
            <td class="py-2 px-3 text-center">${statusBadge(qa.status, qa.success)}</td>
            <td class="py-2 px-3 text-center">${statusBadge(qb.status, qb.success)}</td>
            <td class="py-2 px-3 text-center bg-emerald-500/5">${statusBadge(qc.status, qc.success)}</td>
          </tr>
        `;
      });
      benchmarkQueriesTbody.innerHTML = queriesHtml;

    } catch (err) {
      benchmarkMetricsTbody.innerHTML = `<tr><td colspan="5" class="py-4 text-center text-rose-400 font-sans">Lỗi tải kết quả đánh giá benchmark: ${err.message}</td></tr>`;
    }
  }

  // ========================================================
  // Helpers
  // ========================================================
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatAnswer(text) {
    if (!text) return '';
    // Format markdown headers, bold, citations
    let formatted = escapeHtml(text)
      .replace(/^### (.*$)/gim, '<h3 class="text-sm font-bold text-sky-400 mt-2 mb-1">$1</h3>')
      .replace(/\*\*(.*?)\*\*/gim, '<strong class="text-white font-semibold">$1</strong>')
      .replace(/`([^`]+)`/gim, '<code class="bg-slate-800 text-sky-300 px-1 py-0.5 rounded text-[11px] font-mono">$1</code>')
      .replace(/_([^_]+)_/gim, '<em class="text-slate-300 italic">$1</em>');
    return formatted;
  }

  // Initialize Memory state on load
  fetchMemoryState();
});
