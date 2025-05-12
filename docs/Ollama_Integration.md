# Ollama Integration for Multi-User Support

This document explains how to use local Ollama models with the application, including support for multiple concurrent users.

## Overview

The application now supports both Anthropic Claude models (via API) and local Ollama models. The Ollama integration includes a queue system that allows multiple users to access a single Ollama instance concurrently, even though Ollama itself processes requests sequentially.

## Setup

### 1. Install Ollama

First, install Ollama on your system:

```bash
# MacOS
curl -fsSL https://ollama.com/install.sh | sh

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the Model You Want to Use

Pull the model you want to use (e.g., Qwen, Llama, Mistral):

```bash
ollama pull qwen
# or
ollama pull llama3
# or
ollama pull mixtral
```

### 3. Configure the Application

Update your configuration to use Ollama by modifying the `config.yml` file or creating a `config.local.yml` with the following settings:

```yaml
api:
  provider: "ollama"  # Switch from "anthropic" to "ollama"
  ollama_url: "http://localhost:11434"  # URL to your Ollama instance
  ollama_model: "qwen"  # Model name that you pulled from Ollama
```

You can also modify other parameters such as `temperature` or `max_tokens` which will be passed through to the model.

## Usage

Once configured, start the application normally:

```bash
python main.py
```

The application will use the Ollama model through the queue system, allowing multiple users to interact with the application concurrently.

## Tool Compatibility

When using Ollama, only tools that have an `openai_schema` defined will be available. This ensures proper compatibility with Ollama's function calling interface. Tools that don't have this schema defined will be automatically skipped when using Ollama.

## Queue Management

The multi-user capability is implemented through a request queue that:

1. Accepts requests from multiple users
2. Processes them one at a time through Ollama
3. Returns results back to the appropriate user

Each request is assigned a unique ID and users will automatically wait for their turn when the queue is busy.

### Queue Configuration

The queue system handles:
- Regular requests
- Streaming requests (with incremental response chunks)
- Priority levels (lower number = higher priority)
- Timeouts and cancellations
- Automatic cleanup of completed requests

## Streaming Support

Streaming responses are supported with Ollama, allowing incremental output to be displayed to users as it's generated, similar to how the Anthropic API works in streaming mode.

## Advanced Usage

### Queue Statistics

For monitoring and debugging, you can view queue statistics:

```python
from api.request_queue import RequestQueue

# Get the queue singleton
queue = RequestQueue.get_instance()

# Get current queue stats
stats = queue.get_queue_stats()
print(f"Queue size: {stats['queue_size']}")
print(f"Active requests: {stats['active_requests']}")
print(f"Avg processing time: {stats['avg_processing_time']} seconds")
```

### Custom Model Settings

You can customize settings for each model by updating the configuration:

```yaml
api:
  provider: "ollama"
  ollama_model: "llama3"
  temperature: 0.7  # More creative
  max_tokens: 2000  # Longer responses
```

## Scaling Considerations

This implementation is designed for small to medium deployments (1-50 concurrent users). For larger deployments, consider:

1. Upgrading your hardware (more RAM, faster GPU)
2. Setting up multiple Ollama instances with a load balancer
3. Using cloud API fallback for peak load times

## Troubleshooting

### Common Issues

1. **Timeout Errors**: If requests are timing out, increase the timeout setting:
   ```yaml
   api:
     timeout: 120  # Increase from default 60 seconds
   ```

2. **Out of Memory**: If Ollama crashes with out of memory errors:
   - Use a smaller model
   - Add more RAM or swap space
   - Decrease the batch size in Ollama

3. **Queue Building Up**: If the queue is growing too large:
   - Consider adding a maximum queue size limit
   - Implement user request throttling
   - Use a smaller/faster model

### Logs

Check the logs for details about queue operations and Ollama interactions:

```bash
python main.py --log-level=DEBUG
```

This will show detailed information about queue processing, request status, and Ollama API interactions.

## Additional Resources

- [Ollama Documentation](https://ollama.com/docs)
- [Ollama Models Library](https://ollama.com/library)