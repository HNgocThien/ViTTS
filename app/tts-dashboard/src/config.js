const config = {
  TRAIN_URL: 'http://localhost:8000', // tts-train container (GPU)
  INFER_URL: 'http://localhost:8001', // tts-infer container (no GPU)
  MRS_URL: 'http://localhost:3000'    // MRS labeling interface
};

export default config;
