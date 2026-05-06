const hostname = window.location.hostname;
const API_BASE = `http://${hostname}:8000/api`;

const config = {
  TRAIN_URL: `${API_BASE}/train`,
  INFER_URL: `${API_BASE}/infer`,
  COLLECT_URL: `${API_BASE}/collect`,
};

export default config;
