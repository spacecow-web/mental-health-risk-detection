import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  analyzeWithActiveTransformer,
  deletePersistedRun,
  evaluateActiveModelOnCorpus,
  fetchDreadditCorpus,
  getPersistedEvaluations,
  getServerModelStatus,
  loadDefaultCorpus,
  setActiveServerModel,
  trainAndEvaluateTransformer,
  type ExplanationMethod,
  type PreprocessOptions,
} from './server/nlpEngine.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 3000);

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '15mb' }));

  // Bootstrap endpoint: loads default dataset, persisted evaluations, and model status automatically
  app.get('/api/bootstrap', async (_req, res) => {
    const modelStatus = getServerModelStatus();
    const persisted = getPersistedEvaluations();
    let defaultDataset: Awaited<ReturnType<typeof loadDefaultCorpus>> | null = null;
    let datasetError: string | null = null;
    try {
      defaultDataset = await loadDefaultCorpus(200);
    } catch (err) {
      datasetError = err instanceof Error ? err.message : 'Default dataset unavailable.';
    }
    res.json({
      modelStatus,
      defaultDataset,
      datasetError,
      persisted,
    });
  });

  // Health & active Transformer status
  app.get('/api/health', (_req, res) => {
    res.json(getServerModelStatus());
  });

  // Switch between built-in zero-shot Transformer and any fine-tuned checkpoint
  app.post('/api/model/select', (req, res) => {
    try {
      const modelId = String(req.body?.modelId || '');
      const status = setActiveServerModel(modelId);
      res.json(status);
    } catch (err) {
      res.status(400).json({
        detail: err instanceof Error ? err.message : 'Could not switch active model.',
      });
    }
  });

  // Run genuine Transformer inference + model-connected SHAP or LIME explainability
  app.post('/api/analyze', async (req, res) => {
    try {
      const text = String(req.body?.text ?? '');
      const method: ExplanationMethod = req.body?.method === 'LIME' ? 'LIME' : 'SHAP';
      const modelId = req.body?.modelId ? String(req.body.modelId) : undefined;
      const options: PreprocessOptions = {
        mask_urls: req.body?.mask_urls ?? true,
        anonymize_mentions: req.body?.anonymize_mentions ?? true,
        strip_hashtag_symbol: req.body?.strip_hashtag_symbol ?? true,
        normalize_whitespace: req.body?.normalize_whitespace ?? true,
        lowercase: req.body?.lowercase ?? false,
      };
      const result = await analyzeWithActiveTransformer({ text, method, modelId, options });
      res.json(result);
    } catch (err) {
      res.status(400).json({
        detail: err instanceof Error ? err.message : 'Transformer analysis failed.',
      });
    }
  });

  // Load default cached research dataset
  app.get('/api/datasets/default', async (req, res) => {
    try {
      const limit = Math.min(Math.max(Number(req.query.limit || 200), 40), 500);
      const dataset = await loadDefaultCorpus(limit);
      res.json(dataset);
    } catch (err) {
      res.status(502).json({
        detail: err instanceof Error ? err.message : 'Unable to load default dataset.',
      });
    }
  });

  // Fetch real public Dreaddit benchmark rows from Hugging Face Datasets Server (cached)
  app.get('/api/datasets/dreaddit', async (req, res) => {
    try {
      const limit = Math.min(Math.max(Number(req.query.limit || 200), 40), 300);
      const rows = await fetchDreadditCorpus(limit);
      res.json({
        datasetName: 'Dreaddit (Turcan & McKeown, 2019)',
        datasetSource: 'Hugging Face Hub (andreagasparini/dreaddit) — Peer-reviewed Reddit stress corpus',
        fileName: 'dreaddit_huggingface_sample.csv',
        rows,
      });
    } catch (err) {
      res.status(502).json({
        detail: err instanceof Error ? err.message : 'Unable to fetch Dreaddit dataset.',
      });
    }
  });

  // Evaluate active model strictly on held-out test split and persist results
  app.post('/api/evaluate', async (req, res) => {
    try {
      const rows = Array.isArray(req.body?.rows) ? req.body.rows : undefined;
      const datasetName = req.body?.datasetName ? String(req.body.datasetName) : undefined;
      const modelId = req.body?.modelId ? String(req.body.modelId) : undefined;
      const result = await evaluateActiveModelOnCorpus({ rows, datasetName, modelId });
      res.json(result);
    } catch (err) {
      res.status(400).json({
        detail: err instanceof Error ? err.message : 'Evaluation pipeline failed.',
      });
    }
  });

  // Delete a persisted evaluation run
  app.delete('/api/evaluations/:runId', (req, res) => {
    const updated = deletePersistedRun(req.params.runId);
    res.json(updated);
  });

  // Real leak-free 70/15/15 split, Transformer encoding, balanced training, and held-out evaluation
  app.post('/api/train', async (req, res) => {
    try {
      const rows = Array.isArray(req.body?.rows) ? req.body.rows : undefined;
      const backbone =
        req.body?.backbone === 'bert-base-encoder' ? 'bert-base-encoder' : 'distilbert-encoder';
      const datasetName = req.body?.datasetName ? String(req.body.datasetName) : undefined;
      const epochs = req.body?.epochs ? Number(req.body.epochs) : 12;
      const maxSamples = req.body?.maxSamples ? Number(req.body.maxSamples) : 240;

      const result = await trainAndEvaluateTransformer({
        rows,
        backbone,
        datasetName,
        epochs,
        maxSamples,
      });
      res.json(result);
    } catch (err) {
      res.status(400).json({
        detail: err instanceof Error ? err.message : 'Training pipeline failed.',
      });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      const indexFile = path.join(distDir, 'index.html');
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send('Production build not found. Run npm run build first.');
      }
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MindTrace AI server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start MindTrace AI server:', err);
  process.exit(1);
});
