import React, { useState, useEffect, useRef } from 'react';
import { Mic, Activity, Headphones, Play, Database, Settings, RefreshCw, Volume2, StopCircle, CheckCircle, AlertCircle, Music } from 'lucide-react';

const API_URL = 'http://localhost:8000';
const MRS_URL = 'http://localhost:3000';

function App() {
  const [activeTab, setActiveTab] = useState('label');
  const [datasets, setDatasets] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);

  const fetchData = () => {
    fetch(`${API_URL}/datasets`).then(r => r.json()).then(d => setDatasets(d.datasets || [])).catch(console.error);
    fetch(`${API_URL}/checkpoints`).then(r => r.json()).then(c => setCheckpoints(c.checkpoints || [])).catch(console.error);
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
          { key: 'test', icon: <Headphones size={18} />, label: '3. Testing' },
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
        {activeTab === 'test' && <TestingTab checkpoints={checkpoints} datasets={datasets} />}
      </div>
    </div>
  );
}

// ─── Labeling Tab ──────────────────────────────────────────────────────────────

function LabelingTab() {
  return (
    <div>
      <div className="page-header">
        <h2 className="page-title"><Mic size={28} /> Data Labeling</h2>
        <p className="page-subtitle">Record your voice using Mimic Recording Studio. Audio files are automatically saved and ready for training.</p>
      </div>
      <div className="glass-card" style={{ height: '72vh', padding: 0, overflow: 'hidden' }}>
        <iframe
          src={MRS_URL}
          width="100%"
          height="100%"
          style={{ border: 'none', background: '#fff', display: 'block' }}
          title="Mimic Recording Studio"
        />
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
    fetch(`${API_URL}/train/status`)
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
      .catch(() => {});
    return () => eventSourceRef.current?.close();
  }, []);

  const attachStream = () => {
    if (eventSourceRef.current) eventSourceRef.current.close();
    const es = new EventSource(`${API_URL}/train_stream`);
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
      const res = await fetch(`${API_URL}/train`, {
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
              <option value="E2TTS-Base">E2TTS-Base</option>
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

// ─── Testing Tab ───────────────────────────────────────────────────────────────

function TestingTab({ checkpoints, datasets }) {
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

  // Load ref audios when dataset changes
  useEffect(() => {
    if (!dataset) { setRefAudios([]); setRefAudio(''); return; }
    fetch(`${API_URL}/datasets/${dataset}/ref_audios`)
      .then(r => r.json())
      .then(d => { setRefAudios(d.ref_audios || []); setRefAudio(''); setPreviewUrl(null); })
      .catch(() => setRefAudios([]));
  }, [dataset]);

  // Update preview when ref audio changes
  useEffect(() => {
    if (refAudio && dataset) {
      setPreviewUrl(`${API_URL}/datasets/${dataset}/audio/${refAudio}`);
    } else {
      setPreviewUrl(null);
    }
  }, [refAudio, dataset]);

  const handleGenerate = async () => {
    if (!text.trim()) return;
    setIsGenerating(true);
    setAudioUrl(null);
    setStatusMsg('');

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          checkpoint: checkpoint || null,
          ref_audio: refAudio || null,
          ref_text: refText || null,
          dataset: dataset || null,
        }),
      });
      const data = await res.json();
      if (data.audio_path) {
        setAudioUrl(`${API_URL}${data.audio_path}?t=${Date.now()}`);
        setStatusMsg(data.warning || data.status || 'Done');
      } else {
        setStatusMsg('Error: No audio returned from backend.');
      }
    } catch (e) {
      setStatusMsg('Network error: Could not connect to backend.');
    } finally {
      setIsGenerating(false);
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
              onChange={e => setDataset(e.target.value)}>
              <option value="">-- None (generic voice) --</option>
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
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

          {refAudio && (
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
        <div className="glass-card">
          <h2><Volume2 size={18} /> Output</h2>

          {!audioUrl && !isGenerating && (
            <div className="empty-state">
              <Volume2 size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
              <p>Generated audio will appear here.</p>
            </div>
          )}

          {isGenerating && (
            <div className="empty-state">
              <div className="loader" style={{ width: 40, height: 40, borderWidth: 3 }} />
              <p style={{ marginTop: '1rem', color: '#60a5fa' }}>Synthesizing...</p>
            </div>
          )}

          {audioUrl && (
            <div className="audio-result">
              <div className="audio-wave-bg">
                <div className="wave-bar" /><div className="wave-bar" /><div className="wave-bar" />
                <div className="wave-bar" /><div className="wave-bar" /><div className="wave-bar" />
                <div className="wave-bar" /><div className="wave-bar" /><div className="wave-bar" />
              </div>
              <audio src={audioUrl} controls autoPlay />
              <a href={audioUrl} download className="btn btn-secondary btn-full" style={{ marginTop: '1rem' }}>
                ⬇ Download WAV
              </a>
              {statusMsg && (
                <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.75rem', textAlign: 'center' }}>
                  {statusMsg}
                </p>
              )}
            </div>
          )}

          {statusMsg && !audioUrl && (
            <p style={{ color: '#f87171', fontSize: '0.875rem' }}>{statusMsg}</p>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;
