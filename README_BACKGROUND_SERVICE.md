# Background Task Service

This background service handles asynchronous task execution for the main conversation system.

## Overview

The system consists of two components:
1. **Main Application** - The conversation system that interacts with the user
2. **Background Service** - A standalone process that executes tasks asynchronously

## Running the System

### Step 1: Start the Background Service

In one terminal window, start the background service:

```bash
python background_service.py
```

The background service will:
- Monitor for task requests
- Execute them using its own LLM instances and tools
- Save results and send notifications back to the main process

### Step 2: Run the Main Application

In another terminal window, run the main application:

```bash
python main.py
```

The main application will:
- Interact with the user
- Schedule background tasks when requested
- Check for task notifications from the background service

## Communication Architecture

The two processes communicate via file-based messaging:

```
/persistent/
  ├── tasks/
  │   ├── pending/       # Pending tasks waiting to be executed
  │   ├── running/       # Tasks currently being processed 
  │   ├── completed/     # Successfully completed tasks with results
  │   └── failed/        # Failed tasks with error information
  └── notifications/     # Notifications back to main process
      └── {conversation_id}/
          └── {notification_id}.json
```

## Example Workflow

1. User asks the main application to "check the weather in the background"
2. Main application creates a task request file in `pending/`
3. Background service picks up the task, moves it to `running/`
4. Background service executes the task, using tools as needed
5. Background service saves results and moves the task to `completed/`
6. Background service creates a notification in `notifications/{conversation_id}/`
7. Main application checks for and displays notifications

## Benefits

- Complete separation of concerns
- Each process can run with its own resources and error handling
- Background tasks survive if the main process crashes
- Easy monitoring and debugging
- Task state is preserved if either process restarts