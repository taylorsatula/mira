# Offline Model Preparation

Offline mode routes chat and internal LLM calls to local OpenAI-compatible endpoints instead of hosted providers. The installer configures MIRA for those endpoints, but the local model servers and model files must exist before first startup.

## Default Local Endpoints

The deploy scripts expect:

- Main model server: `http://localhost:3090/v1/chat/completions`
- Small model server: `http://localhost:3092/v1/chat/completions`
- Health check: `curl http://localhost:3090/health`

Default model storage:

```text
/opt/mira/models/
```

Default llama-server logs:

```text
/opt/mira/logs/llama-main.log
/opt/mira/logs/llama-small.log
```

## Model Expectations

The automatic offline profile is designed around two local llama-server instances:

- Main chat model for primary conversation work
- Smaller model for lower-cost internal analysis and maintenance tasks

If you bring your own GGUF models, keep the endpoint URLs and model names aligned with the values selected during `deploy/deploy.sh` configuration (see [Model Name Configuration](#model-name-configuration) below). MIRA treats those endpoints as OpenAI-compatible dialects.

## Model Name Configuration

`model_configs.model` records the name each llama-server instance serves, and MIRA sends it with every request — llama-server ignores it, but servers such as Ollama and vLLM select the served model by it. The name reaches the database in one of three ways:

- **Interactive install** (`./deploy.sh`): the offline interview prompts for the main and small model names (defaults `local-main` / `local-small`).
- **Config-file install** (`./deploy.sh --config deploy-config.yml`): the `llama_main_model` and `llama_small_model` keys — see `deploy/deploy-config.example.yml`.
- **Environment override**: export `MIRA_LLAMA_MAIN_MODEL` / `MIRA_LLAMA_SMALL_MODEL` before running the installer; these take precedence over the config-file values.

To change the names after an install, re-run the installer with one of the non-interactive methods above.

## Startup Checklist

Before starting MIRA in offline mode:

1. Place the GGUF model files under `/opt/mira/models/` or the path you configured.
2. Start the main llama-server on port `3090`.
3. Start the small llama-server on port `3092`.
4. Confirm both endpoints respond before launching MIRA.

The application will fail loudly if a required local provider is not reachable.
