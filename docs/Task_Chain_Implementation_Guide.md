# Task Chain Implementation Guide

This guide outlines the strategy and phases for implementing the Task Chain System, a powerful yet maintainable approach to creating chainable scheduled tasks with data flow, conditional logic, and error handling.

## System Overview

The Task Chain System allows users to:
1. Create sequences of connected tool operations
2. Pass data between steps using template substitution
3. Implement conditional logic and branching
4. Handle errors at multiple levels
5. Schedule chain execution with various frequencies

## Implementation Phases

### Phase 1: Foundation (Core Database Models and Basic Execution)

#### 1.1 Database Models
- [ ] Create `task_chains` table and ORM model
- [ ] Create `chain_steps` table and ORM model
- [ ] Create `chain_executions` table and ORM model
- [ ] Create `step_executions` table and ORM model
- [ ] Add indexes for performance optimization
- [ ] Establish foreign key relationships for data integrity

#### 1.2 Basic Chain Execution
- [ ] Implement `ChainExecutor` class for running chains
- [ ] Add linear step execution (one after another)
- [ ] Implement direct tool operation execution
- [ ] Create basic context management for passing data
- [ ] Add simple error propagation

#### 1.3 Integration with Existing Scheduler
- [ ] Extend scheduler to handle chain execution
- [ ] Add chain due-time calculation
- [ ] Implement chain status tracking
- [ ] Create scheduler service integration for chains

### Phase 2: Enhanced Execution (Templates, Error Handling, Notifications)

#### 2.1 Template Substitution System
- [ ] Create template parser for parameter substitution
- [ ] Implement dot notation for accessing nested properties
- [ ] Add support for basic formatting functions
- [ ] Create evaluation engine for simple expressions
- [ ] Add value type handling and conversion

#### 2.2 Comprehensive Error Handling
- [ ] Implement chain-level error policies
- [ ] Add step-level error policy overrides
- [ ] Create retry mechanism with backoff
- [ ] Add execution status tracking
- [ ] Implement timeout handling

#### 2.3 Notification System
- [ ] Create chain execution notifications
- [ ] Add step failure notifications
- [ ] Implement custom notification points
- [ ] Add notification persistence
- [ ] Create notification display in conversations

### Phase 3: Advanced Features (Conditions, Loops, Orchestration)

#### 3.1 Conditional Logic
- [ ] Implement condition evaluation engine
- [ ] Add conditional next step determination
- [ ] Create branching logic for multiple paths
- [ ] Implement step skipping based on conditions
- [ ] Add conditional chain termination

#### 3.2 Repetition and Loops
- [ ] Create step repetition with intervals
- [ ] Implement repetition termination conditions
- [ ] Add maximum iteration limits
- [ ] Create loop context management
- [ ] Implement dynamic interval adjustment

#### 3.3 LLM Orchestration
- [ ] Add orchestrated step type
- [ ] Implement LLM prompt construction
- [ ] Create context passing to LLM
- [ ] Add result extraction and parsing
- [ ] Implement tool access controls for LLM

### Phase 4: Management Interface and Utilities

#### 4.1 Chain Creation and Management Tool
- [ ] Create `chain_tool` for creating and managing chains
- [ ] Implement chain listing and filtering
- [ ] Add chain versioning
- [ ] Create chain duplication
- [ ] Implement chain status management

#### 4.2 History and Monitoring
- [ ] Add execution history retrieval
- [ ] Create execution debugging helpers
- [ ] Implement performance monitoring
- [ ] Add chain statistics
- [ ] Create execution visualization

#### 4.3 Testing Utilities
- [ ] Implement chain testing mode
- [ ] Create step simulation
- [ ] Add context manipulation utilities
- [ ] Implement validation tools
- [ ] Create troubleshooting helpers

## Component Design

### 1. Task Chain Model
```python
class TaskChain(Base):
    __tablename__ = "task_chains"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Scheduling
    frequency = Column(SQLAEnum(TaskFrequency))
    scheduled_time = Column(DateTime)
    timezone = Column(String)
    
    # Status
    status = Column(SQLAEnum(ChainStatus), default=ChainStatus.ACTIVE)
    last_execution_id = Column(String, ForeignKey("chain_executions.id"))
    last_execution_time = Column(DateTime)
    next_execution_time = Column(DateTime)
    
    # Configuration
    error_policy = Column(SQLAEnum(ErrorPolicy), default=ErrorPolicy.CONTINUE)
    timeout = Column(Integer, default=3600)  # seconds
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, onupdate=datetime.now)
    created_by = Column(String)
    version = Column(Integer, default=1)
    
    # Relationships
    steps = relationship("ChainStep", back_populates="chain", 
                        order_by="ChainStep.position", cascade="all, delete-orphan")
    executions = relationship("ChainExecution", back_populates="chain")
    last_execution = relationship("ChainExecution", foreign_keys=[last_execution_id])
```

### 2. Chain Step Model
```python
class ChainStep(Base):
    __tablename__ = "chain_steps"
    
    id = Column(String, primary_key=True)
    chain_id = Column(String, ForeignKey("task_chains.id"), nullable=False)
    
    name = Column(String, nullable=False)
    position = Column(Integer, nullable=False)
    
    # Execution configuration
    tool_name = Column(String)
    operation = Column(String)
    execution_mode = Column(SQLAEnum(ExecutionMode))
    task_description = Column(Text)  # For orchestrated steps
    
    # Parameters and results
    parameters = Column(MutableDict.as_mutable(JSON), default={})
    output_key = Column(String)
    
    # Flow control
    next_step_logic = Column(MutableDict.as_mutable(JSON), default={})
    timeout = Column(Integer)  # seconds
    error_policy = Column(SQLAEnum(ErrorPolicy))
    retry_config = Column(MutableDict.as_mutable(JSON), default={})
    repeat_config = Column(MutableDict.as_mutable(JSON), default={})
    
    # Relationships
    chain = relationship("TaskChain", back_populates="steps")
    executions = relationship("StepExecution", back_populates="step")
```

### 3. Chain Executor
The central component that manages the execution of chains:

```python
class ChainExecutor:
    def __init__(self, tool_repo, llm_bridge):
        self.tool_repo = tool_repo
        self.llm_bridge = llm_bridge
        self.db = Database()
        self.template_engine = TemplateEngine()
    
    def execute_chain(self, chain_id):
        """Execute a complete chain by its ID"""
        # Implementation details...
    
    def _execute_step(self, step, context):
        """Execute a single chain step"""
        # Implementation details...
    
    def _resolve_next_step(self, step, result, context):
        """Determine the next step based on logic and results"""
        # Implementation details...
    
    def _handle_error(self, step, error, context):
        """Handle errors based on policy"""
        # Implementation details...
```

### 4. Template Engine
Handles parameter substitution and expression evaluation:

```python
class TemplateEngine:
    def resolve_template(self, template, context):
        """Resolve a template string against the given context"""
        # Implementation details...
    
    def evaluate_condition(self, condition, context):
        """Evaluate a conditional expression against context"""
        # Implementation details...
    
    def _parse_expression(self, expr):
        """Parse a template expression"""
        # Implementation details...
```

## Data Structures

### 1. Chain Definition (JSON Format)
```json
{
  "name": "Morning Brief Generator",
  "description": "Creates a daily morning briefing",
  "schedule": {
    "frequency": "daily",
    "time": "05:00:00",
    "timezone": "America/New_York"
  },
  "error_policy": "continue",
  "timeout": 1800,
  "steps": [
    {
      "name": "Get Weather",
      "tool": "weather_tool",
      "operation": "get_forecast",
      "parameters": { 
        "location": "home"
      },
      "output_key": "weather"
    },
    {
      "name": "Generate Report",
      "execution_mode": "orchestrated",
      "task_description": "Create a weather summary",
      "parameters": {
        "weather_data": "{weather}"
      },
      "output_key": "report"
    }
  ]
}
```

### 2. Execution Context
```python
{
  "chain_id": "chain123",
  "execution_id": "exec456",
  "start_time": "2025-05-04T05:00:00Z",
  "variables": {
    "date": "2025-05-04",
    "user_id": "user123"
  },
  "results": {
    "weather": {
      "temperature": 72,
      "condition": "sunny"
    },
    "report": "Today will be sunny with a high of 72°F."
  },
  "current_step": "Generate Report",
  "completed_steps": ["Get Weather"],
  "errors": {}
}
```

## Error Handling

### Error Policies
- `STOP`: Stop chain execution on error
- `CONTINUE`: Continue to next step on error
- `RETRY`: Retry the failed step
- `ALTERNATIVE`: Execute an alternative step
- `ROLLBACK`: Execute compensation steps

### Retry Configuration
```json
{
  "max_attempts": 3,
  "backoff_factor": 2,
  "initial_delay": 5,
  "jitter": true
}
```

## Integration Points

1. **Scheduler Integration**: The Task Chain system extends the existing scheduler
2. **Tool Integration**: Chains execute operations through the existing tool repository
3. **Notification Integration**: Chain results create notifications via the notification manager
4. **Conversation Integration**: Notifications appear in conversations

## Testing Strategy

1. **Unit Tests**: Test individual components (template engine, executor, etc.)
2. **Integration Tests**: Test chains with mock tools
3. **System Tests**: Test complete chains with real tools
4. **Performance Tests**: Test chain execution with large contexts
5. **Error Handling Tests**: Verify error policies work correctly

## Best Practices

1. **Chain Design**
   - Keep chains focused on a single purpose
   - Use clear, descriptive step names
   - Structure steps in a logical sequence
   - Use output keys that reflect content

2. **Error Handling**
   - Define appropriate error policies for each step
   - Consider retry for transient failures
   - Provide useful error messages
   - Test failure scenarios

3. **Performance**
   - Avoid excessive data in the context
   - Use efficient tools for steps
   - Consider timeouts for long-running steps
   - Monitor chain execution times

4. **Maintenance**
   - Document chains and their purpose
   - Version chains appropriately
   - Test chains after tool updates
   - Monitor chain execution statistics

## Future Enhancements

1. **Chain Templates**: Reusable chain templates with customizable parameters
2. **Chain Versioning**: Formal versioning system with change tracking
3. **Chain Visualization**: Interactive visualization of chain execution
4. **Chain Monitoring**: Real-time monitoring and alerting
5. **Chain Marketplace**: Shareable pre-built chains for common tasks

## Implementation Checklist

- [ ] Database schema design and creation
- [ ] Core models implementation
- [ ] Basic chain execution
- [ ] Template engine development
- [ ] Error handling system
- [ ] Scheduler integration
- [ ] Conditional logic
- [ ] Repetition and loops
- [ ] LLM orchestration
- [ ] Management tool
- [ ] Testing utilities
- [ ] Documentation

This implementation guide will evolve as we progress through the development phases.