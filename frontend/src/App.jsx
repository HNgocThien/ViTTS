import React, { useState, useEffect, useRef } from 'react';
import { Mic, Activity, Headphones, Play, Database, Settings, RefreshCw, Volume2, StopCircle, CheckCircle, AlertCircle, Music, ArrowLeft, ArrowRight, RotateCcw, ChevronDown, List, Plus, PlayCircle } from 'lucide-react';
import config from './config';

const { TRAIN_URL, INFER_URL, COLLECT_URL } = config;

function App() {
  const [activeTab, setActiveTab] = useState('label');
  const [datasets, setDatasets] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);

  const fetchData = () => {
    fetch(`${INFER_URL}/datasets`).then(r => r.json()).then(d => setDatasets(d.datasets || [])).catch(console.error);
    fetch(`${INFER_URL}/checkpoints`).then(r => r.json()).then(c => setCheckpoints(c.checkpoints || [])).catch(console.error);
  };

  useEffect(() => { fetchData(); }, []);

  return (
    <div className="dashboard-container">
      <div className="sidebar">
        <h1>🎙️ F5-TTS Studio</h1>
        <div className="nav-section-label">PIPELINE</div>
        {[
          { key: 'label', icon: <Mic size={18} />, label: '1. Labeling' },
          { key: 'train', icon: <Activity size={18} />, label: '2. Training' },
          { key: 'infer', icon: <Headphones size={18} />, label: '3. Inference' },
        ].map(({ key, icon, label }) => (
          <div
            key={key}
            className={`nav-item ${activeTab === key ? 'active' : ''}`}
            onClick={() => setActiveTab(key)}
          >
            {icon}
            <span>{label}</span>
          </div>
        ))}

        <div className="sidebar-footer">
          <button className="btn-icon" onClick={fetchData} title="Refresh datasets & checkpoints">
            <RefreshCw size={16} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      <div className="main-content">
        {activeTab === 'label' && <LabelingTab />}
        {activeTab === 'train' && <TrainingTab datasets={datasets} />}
        {activeTab === 'infer' && <InferenceTab checkpoints={checkpoints} datasets={datasets} activeTab={activeTab} />}
      </div>
    </div>
  );
}

// ─── Labeling Tab ──────────────────────────────────────────────────────────────

function LabelingTab() {
  return (
    <div>
      <div className="page-header">
        <h2 className="page-title"><Mic size={28} /> Data Collection</h2>
        <p className="page-subtitle">Record your voice natively. Audio files are automatically formatted to LibriSpeech standard.</p>
      </div>
      <NativeLabelingStudio />
    </div>
  );
}



function NativeLabelingStudio() {
  const [datasetName, setDatasetName] = useState(() => localStorage.getItem('tts_dataset_name') || "");
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [isStarted, setIsStarted] = useState(() => localStorage.getItem('tts_is_started') === 'true');
  const [showDatasetList, setShowDatasetList] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [promptId, setPromptId] = useState(() => Number(localStorage.getItem('tts_prompt_id')) || 0);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState("");

  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Persist session state
  useEffect(() => {
    localStorage.setItem('tts_dataset_name', datasetName);
    localStorage.setItem('tts_is_started', isStarted);
    localStorage.setItem('tts_prompt_id', promptId);
  }, [datasetName, isStarted, promptId]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${config.COLLECT_URL}/datasets`);
      const data = await res.json();
      setAvailableDatasets(data.datasets || []);
    } catch (e) { console.error("Failed to fetch datasets", e); }
  };

  const fetchPrompt = async (targetId = null) => {
    if (!datasetName) return;
    try {
      const url = targetId
        ? `${config.COLLECT_URL}/prompt?uuid=${datasetName}&id=${targetId}`
        : `${config.COLLECT_URL}/prompt?uuid=${datasetName}`;

      const res = await fetch(url);
      const data = await res.json();
      if (data.success) {
        setPrompt(data.data.prompt);
        setPromptId(data.data.prompt_id);
        setTotal(data.data.total);
        setStatus("");
        setRecordedBlob(null);

        // If server returned an existing audio URL, show it in preview
        if (data.data.audio_url) {
          setPreviewUrl(`${config.INFER_URL}${data.data.audio_url}?t=${Date.now()}`);
        } else {
          if (previewUrl && !previewUrl.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
          setPreviewUrl(null);
        }
      } else {
        setPrompt("");
        setStatus(data.message || "No more prompts");
      }
    } catch (e) {
      setStatus("Error connecting to backend");
    }
  };

  useEffect(() => {
    fetchDatasets();
    if (isStarted) {
      // Use persisted prompt ID if available, else fetch next
      fetchPrompt(promptId > 0 ? promptId : null);
    }
  }, []);

  const handleStartSession = () => {
    if (!datasetName.trim()) {
      alert("Please enter or select a dataset name.");
      return;
    }
    setIsStarted(true);
    fetchPrompt(); // Fetch next unrecorded
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setRecordedBlob(audioBlob);
        setPreviewUrl(URL.createObjectURL(audioBlob));
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordedBlob(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    } catch (e) {
      alert("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const handleNext = async () => {
    // 1. If we have a NEW recording, upload it first
    if (recordedBlob) {
      setIsUploading(true);
      setStatus("Uploading...");

      const formData = new FormData();
      formData.append("uuid", datasetName);
      formData.append("prompt", prompt);
      formData.append("prompt_id", promptId);
      formData.append("audio", recordedBlob, "recording.wav");

      try {
        const res = await fetch(`${config.COLLECT_URL}/audio`, {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (!data.success) {
          setStatus("Failed to save audio");
          setIsUploading(false);
          return; // Stop if upload failed
        }
        setStatus("Saved!");
      } catch (e) {
        setStatus("Upload failed");
        setIsUploading(false);
        return;
      } finally {
        setIsUploading(false);
      }
    }

    // 2. Move to next sentence
    if (promptId < total) {
      fetchPrompt(promptId + 1);
    } else {
      setStatus("End of dataset raggiunra!");
    }
  };

  const handlePrev = () => {
    if (promptId > 1) {
      fetchPrompt(promptId - 1);
    }
  };

  const handleRecollect = () => {
    setRecordedBlob(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setStatus("Ready to re-record");
  };

  if (!isStarted) {
    return (
      <div className="glass-card" style={{ maxWidth: '600px', margin: '2rem auto', padding: '3rem' }}>
        <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Database size={24} /> Step 1: Initialize Dataset
        </h3>

        <div style={{ position: 'relative', marginBottom: '2rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Dataset Name</label>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <div style={{ flex: 1, position: 'relative' }}>
              <input
                type="text"
                placeholder="Enter new name..."
                value={datasetName}
                onChange={e => setDatasetName(e.target.value)}
                className="form-control"
                style={{ width: '100%', paddingRight: '10px' }}
              />
            </div>
            <div style={{ position: 'relative' }}>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ height: '45px', display: 'flex', alignItems: 'center', gap: '0.5rem', whiteSpace: 'nowrap' }}
                onClick={() => setShowDatasetList(!showDatasetList)}
              >
                <List size={18} />
                <span>Existing</span>
                <ChevronDown size={16} style={{ transition: 'transform 0.2s', transform: showDatasetList ? 'rotate(180deg)' : 'rotate(0)' }} />
              </button>

              {showDatasetList && (
                <div className="glass-card" style={{
                  position: 'absolute', top: '100%', right: 0, zIndex: 100, width: '250px',
                  marginTop: '0.5rem', maxHeight: '250px', overflowY: 'auto', padding: '0.5rem',
                  boxShadow: '0 10px 25px rgba(0,0,0,0.3)', border: '1px solid var(--border-color)'
                }}>
                  <div style={{ padding: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)', borderBottom: '1px solid var(--border-color)', marginBottom: '0.5rem' }}>
                    AVAILABLE DATASETS
                  </div>
                  {availableDatasets.length > 0 ? availableDatasets.map(d => (
                    <div
                      key={d}
                      className="nav-item"
                      style={{ padding: '0.75rem', borderRadius: '8px', cursor: 'pointer', marginBottom: '2px' }}
                      onClick={() => { setDatasetName(d); setShowDatasetList(false); }}
                    >
                      <Database size={14} style={{ marginRight: '0.75rem', opacity: 0.6 }} />
                      <span style={{ fontWeight: 500 }}>{d}</span>
                    </div>
                  )) : <p style={{ padding: '1rem', color: 'var(--text-muted)', textAlign: 'center' }}>No datasets found</p>}
                </div>
              )}
            </div>
          </div>
        </div>

        <button
          type="button"
          className="btn btn-primary btn-full"
          onClick={handleStartSession}
          disabled={!datasetName.trim()}
          style={{ height: '55px', fontSize: '1.1rem' }}
        >
          <PlayCircle size={20} /> Confirm & Start
        </button>
      </div>
    );
  }

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '3rem', marginTop: '1rem', textAlign: 'center' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '2rem', alignItems: 'center' }}>
        <div style={{ textAlign: 'left' }}>
          <h4 style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Current Dataset</h4>
          <code style={{ fontSize: '1.3rem', color: 'var(--primary-color)', fontWeight: 'bold' }}>{datasetName}</code>
        </div>
        <button type="button" className="btn btn-secondary" onClick={() => { setIsStarted(false); localStorage.removeItem('tts_is_started'); }}>
          <RotateCcw size={14} /> Reset / Change
        </button>
      </div>

      <div style={{
        width: '100%',
        maxWidth: '800px',
        height: '250px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-main)',
        borderRadius: '16px',
        padding: '2.5rem',
        border: '1px solid var(--border-color)',
        marginBottom: '1rem',
        boxShadow: 'inset 0 2px 10px rgba(0,0,0,0.2)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Animated background subtle effect */}
        {isRecording && (
          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '4px', background: 'var(--danger-color)', animation: 'pulse 1.5s infinite' }} />
        )}

        <h1 style={{ fontSize: prompt.length > 50 ? '1.8rem' : '2.8rem', margin: 0, lineHeight: 1.3, fontWeight: 600 }}>
          {prompt || (status.includes("No more") ? "🎉 All Done!" : "Loading Prompt...")}
        </h1>
      </div>

      <div style={{ display: 'flex', width: '100%', maxWidth: '800px', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem 1rem', borderRadius: '100px', fontSize: '0.9rem' }}>
          Progress: <strong style={{ color: 'var(--primary-color)' }}>{promptId} / {total || "?"}</strong>
        </div>
        {status && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: status === 'Saved!' ? 'var(--success-color)' : 'var(--primary-color)', fontWeight: 'bold' }}>
            {status === 'Saved!' && <CheckCircle size={16} />}
            {status}
          </div>
        )}
      </div>

      <div className="controls-box" style={{
        background: 'rgba(255,255,255,0.03)',
        padding: '2.5rem',
        borderRadius: '32px',
        width: '100%',
        maxWidth: '700px',
        display: 'flex',
        flexDirection: 'column',
        gap: '2.5rem',
        alignItems: 'center',
        border: '1px solid rgba(255,255,255,0.05)'
      }}>

        {/* Review Section - ALWAYS VISIBLE */}
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            fontSize: '0.75rem',
            color: previewUrl ? 'var(--primary-color)' : 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            {previewUrl ? <><div className="badge-dot" style={{ background: 'var(--primary-color)' }} /> Review Recording</> : 'Awaiting Recording'}
          </div>
          <div style={{ width: '100%', position: 'relative' }}>
            <audio src={previewUrl || ''} controls style={{ width: '100%', opacity: previewUrl ? 1 : 0.3, pointerEvents: previewUrl ? 'all' : 'none', transition: 'opacity 0.3s' }} />
            {!previewUrl && (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none', color: 'rgba(255,255,255,0.2)', fontSize: '0.8rem' }}>
                Record to preview audio
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <button
            type="button"
            className="btn btn-secondary shadow-sm"
            title="Previous"
            onClick={handlePrev}
            disabled={isRecording || promptId <= 1}
            style={{ width: '60px', height: '60px', borderRadius: '16px', padding: 0 }}
          >
            <ArrowLeft size={28} />
          </button>

          {!recordedBlob ? (
            <button
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              className={`btn ${isRecording ? 'btn-danger' : 'btn-primary'}`}
              disabled={!prompt}
              style={{ width: '100px', height: '100px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: isRecording ? '0 0 30px rgba(239, 68, 68, 0.5)' : '0 10px 20px rgba(0,0,0,0.2)', transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)' }}
            >
              {isRecording ? <StopCircle size={45} /> : <Mic size={45} />}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleRecollect}
              className="btn btn-secondary shadow-hover"
              title="Discard and record again"
              style={{ width: '100px', height: '100px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.1)' }}
            >
              <RotateCcw size={45} />
            </button>
          )}

          <button
            type="button"
            className="btn btn-primary shadow-sm"
            title="Save & Next"
            onClick={handleNext}
            disabled={isRecording || isUploading}
            style={{ width: '60px', height: '60px', borderRadius: '16px', padding: 0, background: recordedBlob ? 'var(--success-color)' : '' }}
          >
            {isUploading ? <div className="loader" style={{ width: 24, height: 24 }} /> : <ArrowRight size={28} />}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Training Tab ──────────────────────────────────────────────────────────────

function TrainingTab({ datasets }) {
  const [config, setConfig] = useState({
    dataset: '',
    base_model: 'F5TTS-Base',
    epoch: 10,
    batch_size: 4,
    learning_rate: 7.5e-5,
    num_warmup_updates: 500,
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle | running | done | failed
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Poll backend status on mount in case a training was already in progress
  useEffect(() => {
    fetch(`${TRAIN_URL}/train_status`)
      .then(r => r.json())
      .then(d => {
        if (d.status === 'running') {
          setIsTraining(true);
          setTrainingStatus('running');
          attachStream();
        } else if (d.status !== 'idle') {
          setTrainingStatus(d.status);
        }
      })
      .catch(() => { });
    return () => eventSourceRef.current?.close();
  }, []);

  const attachStream = () => {
    if (eventSourceRef.current) eventSourceRef.current.close();
    const es = new EventSource(`${TRAIN_URL}/train_stream`);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const line = event.data;
      setLogs(prev => {
        const next = [...prev, line];
        return next.length > 200 ? next.slice(next.length - 200) : next;
      });

      if (line.includes('[STATUS] DONE')) {
        es.close();
        setIsTraining(false);
        setTrainingStatus('done');
      } else if (line.includes('[STATUS] FAILED')) {
        es.close();
        setIsTraining(false);
        setTrainingStatus('failed');
      }
    };

    es.onerror = () => {
      es.close();
      setIsTraining(false);
    };
  };

  const handleStart = async () => {
    if (!config.dataset) {
      alert('Please select a dataset first.');
      return;
    }
    setIsTraining(true);
    setTrainingStatus('running');
    setLogs([]);

    try {
      const res = await fetch(`${TRAIN_URL}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      setLogs([`Backend: ${data.status}`]);
      attachStream();
    } catch (e) {
      setLogs([`Network Error: Could not connect to backend.`]);
      setIsTraining(false);
      setTrainingStatus('failed');
    }
  };

  const handleStop = () => {
    eventSourceRef.current?.close();
    setIsTraining(false);
  };

  const statusBadge = {
    idle: null,
    running: <span className="badge badge-running"><span className="badge-dot" /> Running</span>,
    done: <span className="badge badge-done"><CheckCircle size={13} /> Done</span>,
    failed: <span className="badge badge-failed"><AlertCircle size={13} /> Failed</span>,
  }[trainingStatus];

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title"><Activity size={28} /> Train F5-TTS Model</h2>
        <p className="page-subtitle">Configure hyperparameters and start fine-tuning on your recorded dataset.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1.3fr)', gap: '2rem' }}>

        {/* Config card */}
        <div className="glass-card">
          <h2><Database size={18} /> Configuration</h2>

          <div className="form-group">
            <label>Dataset</label>
            <select className="form-control" value={config.dataset}
              onChange={e => setConfig({ ...config, dataset: e.target.value })}>
              <option value="">-- Choose Dataset --</option>
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
              {datasets.length === 0 && <option value="vietnamese_train">vietnamese_train</option>}
            </select>
          </div>

          <div className="form-group">
            <label>Base Model</label>
            <select className="form-control" value={config.base_model}
              onChange={e => setConfig({ ...config, base_model: e.target.value })}>
              <option value="F5TTS-Base">F5TTS-Base (Recommended)</option>

            </select>
          </div>

          <div className="flex-row">
            <div className="form-group" style={{ flex: 1 }}>
              <label>Epochs</label>
              <input type="number" className="form-control" min={1} value={config.epoch}
                onChange={e => setConfig({ ...config, epoch: Number(e.target.value) })} />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Batch Size</label>
              <input type="number" className="form-control" min={1} value={config.batch_size}
                onChange={e => setConfig({ ...config, batch_size: Number(e.target.value) })} />
            </div>
          </div>

          <div className="flex-row">
            <div className="form-group" style={{ flex: 1 }}>
              <label>Learning Rate</label>
              <input type="number" className="form-control" step="0.000001" value={config.learning_rate}
                onChange={e => setConfig({ ...config, learning_rate: Number(e.target.value) })} />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Warmup Steps</label>
              <input type="number" className="form-control" min={0} value={config.num_warmup_updates}
                onChange={e => setConfig({ ...config, num_warmup_updates: Number(e.target.value) })} />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button
              className="btn btn-primary btn-full"
              onClick={handleStart}
              disabled={isTraining}
            >
              {isTraining ? <><div className="loader" /> Training...</> : <><Activity size={16} /> Start Training</>}
            </button>
            {isTraining && (
              <button className="btn btn-danger" onClick={handleStop} title="Disconnect stream">
                <StopCircle size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Logs card */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={18} /> Real-time Logs
            </span>
            {statusBadge}
          </h2>
          <div className="logs-container" style={{ flex: 1 }}>
            {logs.length === 0
              ? <span style={{ color: '#4b5563' }}>Waiting to start...</span>
              : logs.map((log, i) => {
                let color = '#10b981';
                if (log.startsWith('[ERROR]') || log.includes('failed')) color = '#f87171';
                else if (log.startsWith('[SUCCESS]') || log.includes('DONE')) color = '#34d399';
                else if (log.startsWith('[INFO]')) color = '#60a5fa';
                else if (log.startsWith('[STATUS]')) color = '#a78bfa';
                return <div key={i} style={{ color, marginBottom: '2px' }}>&gt; {log}</div>;
              })}
            <div ref={logsEndRef} />
          </div>
        </div>

      </div>
    </div>
  );
}

// ─── Inference Tab ───────────────────────────────────────────────────────────────

function InferenceTab({ checkpoints, datasets, activeTab }) {
  const [text, setText] = useState('Nhớ ai bổi hổi bồi hồi, như đứng đống lửa, như ngồi đống than.');
  const [checkpoint, setCheckpoint] = useState('');
  const [dataset, setDataset] = useState('');
  const [refAudios, setRefAudios] = useState([]);
  const [refAudio, setRefAudio] = useState('');
  const [refText, setRefText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [statusMsg, setStatusMsg] = useState('');
  const [previewUrl, setPreviewUrl] = useState(null);
  const [inferStatus, setInferStatus] = useState('idle'); // idle | running | done | failed
  const [inferLogs, setInferLogs] = useState([]);
  const inferLogsEndRef = useRef(null);
  const inferEventSourceRef = useRef(null);

  useEffect(() => {
    inferLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [inferLogs]);

  // Recover state on mount
  useEffect(() => {
    fetch(`${INFER_URL}/infer_status`)
      .then(r => r.json())
      .then(d => {
        if (d.status === 'running') {
          setIsGenerating(true);
          setInferStatus('running');
          attachInferStream();
        } else if (d.status === 'done' && d.audio_url) {
          setAudioUrl(`${INFER_URL}${d.audio_url.replace('/api/infer/audio/', '/audio/')}?t=${Date.now()}`);
          setInferStatus('done');
        }
      })
      .catch(() => {});
    return () => inferEventSourceRef.current?.close();
  }, []);

  const attachInferStream = () => {
    if (inferEventSourceRef.current) inferEventSourceRef.current.close();
    const es = new EventSource(`${INFER_URL}/infer_stream`);
    inferEventSourceRef.current = es;

    es.onmessage = (event) => {
      const line = event.data;
      setInferLogs(prev => {
        const next = [...prev, line];
        return next.length > 200 ? next.slice(next.length - 200) : next;
      });

      if (line.includes('[SUCCESS]')) {
        es.close();
        setIsGenerating(false);
        setInferStatus('done');
        // Fetch the final URL
        fetch(`${INFER_URL}/infer_status`)
          .then(r => r.json())
          .then(d => {
             if (d.audio_url) {
                setAudioUrl(`${INFER_URL}${d.audio_url.replace('/api/infer/audio/', '/audio/')}?t=${Date.now()}`);
             }
          });
      } else if (line.includes('[ERROR]') || line.includes('[CRITICAL]')) {
        es.close();
        setIsGenerating(false);
        setInferStatus('failed');
      }
    };

    es.onerror = () => {
      es.close();
      setIsGenerating(false);
    };
  };

  // Recording Ref
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setRecordedBlob(blob);
        setPreviewUrl(URL.createObjectURL(blob));
        setDataset(''); // Override dataset selection
        setRefAudio('');
      };
      mediaRecorder.start();
      setIsRecording(true);
    } catch (e) { alert("Mic access denied"); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
    }
  };

  // Load ref audios when dataset changes
  useEffect(() => {
    if (!dataset) { setRefAudios([]); setRefAudio(''); return; }
    fetch(`${INFER_URL}/datasets/${dataset}/ref_audios`)
      .then(r => r.json())
      .then(d => { setRefAudios(d.ref_audios || []); setRefAudio(''); setPreviewUrl(null); })
      .catch(() => setRefAudios([]));
  }, [dataset, activeTab]);

  // Update preview when ref audio changes
  useEffect(() => {
    if (refAudio && dataset) {
      setPreviewUrl(`${INFER_URL}/datasets/${dataset}/audio/${refAudio}?t=${Date.now()}`);
    } else {
      setPreviewUrl(null);
    }
  }, [refAudio, dataset, activeTab]);

  const handleGenerate = async () => {
    if (!text.trim()) return;
    setIsGenerating(true);
    setAudioUrl(null);
    setStatusMsg('');

    let finalRefAudio = refAudio;
    let finalDataset = dataset;

    // 1. If we have a NEW recording, upload it first
    if (recordedBlob) {
      setIsUploading(true);
      setStatusMsg("Uploading reference recording...");
      const formData = new FormData();
      formData.append("audio", recordedBlob, "ref.wav");
      try {
        const upRes = await fetch(`${INFER_URL}/upload_ref`, { method: "POST", body: formData });
        const upData = await upRes.json();
        if (upData.success) {
          finalRefAudio = upData.ref_at;
          finalDataset = null; // Important: use absolute path returned
        }
      } catch (e) {
        console.error("Upload failed", e);
      } finally {
        setIsUploading(false);
      }
    }

    try {
      const res = await fetch(`${INFER_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          checkpoint: checkpoint || null,
          ref_audio: finalRefAudio || null,
          ref_text: refText || null,
          dataset: finalDataset || null,
        }),
      });
      const data = await res.json();
      setInferLogs([`Backend: ${data.status}`]);
      setInferStatus('running');
      attachInferStream();
    } catch (e) {
      setInferLogs([`Network error: Could not connect to backend.`]);
      setIsGenerating(false);
      setInferStatus('failed');
    }
  };

  const handleDownload = async () => {
    if (!audioUrl) return;
    try {
      const res = await fetch(audioUrl);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `output_${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Download failed", e);
      window.open(audioUrl, '_blank');
    }
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title"><Headphones size={28} /> Generate Audio</h2>
        <p className="page-subtitle">Choose a trained checkpoint, optionally provide a reference audio for voice conditioning, then synthesize text.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: '2rem' }}>

        {/* Settings */}
        <div className="glass-card">
          <h2><Settings size={18} /> Inference Settings</h2>

          <div className="form-group">
            <label>Model Checkpoint</label>
            <select className="form-control" value={checkpoint}
              onChange={e => setCheckpoint(e.target.value)}>
              <option value="">-- Auto: latest checkpoint --</option>
              {checkpoints.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Reference Dataset (for voice conditioning)</label>
            <select className="form-control" value={dataset}
              onChange={e => { setDataset(e.target.value); if (e.target.value) setRecordedBlob(null); }}>
              <option value="">-- None (generic voice) --</option>
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Or: Record Reference Audio</span>
              {recordedBlob && <span style={{ fontSize: '0.7rem', color: 'var(--success-color)' }}>● Recorded</span>}
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className={`btn ${isRecording ? 'btn-danger' : 'btn-secondary'} btn-full`}
                onClick={isRecording ? stopRecording : startRecording}
              >
                {isRecording ? <><StopCircle size={14} /> Stop Recording</> : <><Mic size={14} /> Start Recording</>}
              </button>
              {recordedBlob && (
                <button className="btn btn-secondary" onClick={() => { setRecordedBlob(null); setPreviewUrl(null); }}>
                  <RotateCcw size={14} />
                </button>
              )}
            </div>
          </div>

          {refAudios.length > 0 && (
            <div className="form-group">
              <label>Reference Audio <span style={{ color: '#6b7280', fontWeight: 400 }}>(conditioning sample)</span></label>
              <select className="form-control" value={refAudio}
                onChange={e => setRefAudio(e.target.value)}>
                <option value="">-- Pick a reference audio --</option>
                {refAudios.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
          )}

          {previewUrl && (
            <div className="form-group">
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Music size={14} /> Preview Reference Audio
              </label>
              <audio src={previewUrl} controls style={{ marginTop: '0.25rem' }} />
            </div>
          )}

          {(refAudio || recordedBlob) && (
            <div className="form-group">
              <label>Reference Transcription <span style={{ color: '#6b7280', fontWeight: 400 }}>(improves quality)</span></label>
              <input type="text" className="form-control" placeholder="Type the text spoken in the reference audio..."
                value={refText} onChange={e => setRefText(e.target.value)} />
            </div>
          )}

          <div className="form-group">
            <label>Text to Synthesize</label>
            <textarea
              className="form-control"
              placeholder="Type Vietnamese or English text here..."
              value={text}
              onChange={e => setText(e.target.value)}
              rows={4}
            />
          </div>

          <button
            className="btn btn-primary btn-full"
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
          >
            {isGenerating
              ? <><div className="loader" /> Synthesizing...</>
              : <><Play size={16} /> Generate Voice</>}
          </button>
        </div>

        {/* Output */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2><Volume2 size={18} /> Output</h2>

          {!audioUrl && !isGenerating && inferStatus !== 'failed' && (
            <div className="empty-state">
              <Volume2 size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
              <p>Generated audio will appear here.</p>
            </div>
          )}

          {isGenerating && (
            <div className="empty-state" style={{ padding: '1rem' }}>
              <div className="loader" style={{ width: 40, height: 40, borderWidth: 3 }} />
              <p style={{ marginTop: '1rem', color: '#60a5fa' }}>Synthesizing...</p>
            </div>
          )}

          {audioUrl && inferStatus === 'done' && !isGenerating && (
            <div className="audio-result">
              <div className="audio-wave-bg">
                <div className="wave-bar" /><div className="wave-bar" /><div className="wave-bar" />
                <div className="wave-bar" /><div className="wave-bar" /><div className="wave-bar" />
                <div className="wave-bar" /><div className="wave-bar" /><div className="wave-bar" />
              </div>
              <audio src={audioUrl} controls autoPlay style={{ width: '100%', marginTop: '1rem' }} />
              <button onClick={handleDownload} className="btn btn-secondary btn-full" style={{ marginTop: '1rem' }}>
                ⬇ Download WAV
              </button>
            </div>
          )}

          {/* Inference Logs Terminal */}
          <div style={{ marginTop: '1.5rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              <Activity size={14} /> Real-time Logs
              {inferStatus === 'running' && <span className="badge-dot" style={{ background: '#10b981', marginLeft: 'auto' }} />}
            </h4>
            <div className="logs-container" style={{ flex: 1, minHeight: '150px' }}>
              {inferLogs.length === 0
                ? <span style={{ color: '#4b5563' }}>Waiting to start...</span>
                : inferLogs.map((log, i) => {
                  let color = '#10b981';
                  if (log.includes('[ERROR]') || log.includes('[CRITICAL]') || log.includes('failed')) color = '#f87171';
                  else if (log.includes('[SUCCESS]') || log.includes('saved')) color = '#34d399';
                  else if (log.includes('[INFO]') || log.includes('Process')) color = '#60a5fa';
                  else if (log.includes('gen_text') || log.includes('Detected')) color = '#a78bfa';
                  return <div key={i} style={{ color, marginBottom: '2px', wordBreak: 'break-all' }}>&gt; {log}</div>;
                })}
              <div ref={inferLogsEndRef} />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
