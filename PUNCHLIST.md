	# MIRA Development Punchlist

## Overview
This punchlist organizes development tasks in dependency order, ensuring each task builds on a solid foundation established by previous work.

## Task Order & Justification

DONE !!!!! ### 1. ✅ Documentation Cleanup
**Task**: Go through docs/ folder and clear out junk - establish current project understanding  
**Priority**: HIGH  
**Justification**: Foundation task - understanding the current project state before making changes. No dependencies on other tasks. Provides clarity on existing standards/patterns.

### 2. ✅ FastAPI Server Completion
**Task**: Setup/complete FastAPI server - ensure API foundation is solid  
**Priority**: HIGH  
**Justification**: Critical infrastructure that other components depend on. API layer should be solid before building features on top. Already partially implemented in `api/fastapi_app.py`.

DONE!!!! ### 3. ✅ Embedding Infrastructure Migration
**Task**: Remove sentence-transformers and replace with onnx - core embedding infrastructure  
**Priority**: HIGH  
**Justification**: Core infrastructure change affecting embeddings throughout the system. ONNX already set up in `onnx/` directory. Should be done early as it affects memory systems and tools.

### 4. ✅ Reference Tool Implementation
**Task**: Create modern sample_tool.py - establish gold standard for all tools  
**Priority**: HIGH  
**Justification**: Establishes the blueprint before updating other tools. No dependencies, but critical for next steps. Becomes the reference implementation for tool standardization.

### 5. ⏳ Tools Standardization
**Task**: Go through tools one-by-one to standardize - apply sample_tool.py patterns  
**Priority**: MEDIUM  
**Justification**: Depends on having sample_tool.py as the blueprint. Must be done before generating classifier examples. Creates consistent interface for lt_memory integration.

### 6. ⏳ Classifier Examples Regeneration
**Task**: Remove old classifier examples and generate new ones - after tools standardized  
**Priority**: MEDIUM  
**Justification**: Depends on standardized tools so examples match current implementation. Clean slate approach prevents confusion. Generator needs to use correct naming conventions.

### 7. ⏳ LT Memory Finalization
**Task**: Finalize lt_memory with LLM integration - leverage standardized components  
**Priority**: MEDIUM  
**Justification**: Can now leverage standardized tools and onnx embeddings. Core component needed by automation engine. Benefits from having clean, standardized interfaces.

### 8. ⏳ Automation Engine Rebuild
**Task**: Rebuild automation engine - depends on finalized lt_memory  
**Priority**: MEDIUM  
**Justification**: Depends heavily on finalized lt_memory. Can use all the standardized components. One of the most complex tasks, needs solid foundation.

### 9. 🔮 Topics System
**Task**: Explore topics concept (e.g., "email" is a topic, "recipe" is a topic)  
**Priority**: LOW  
**Justification**: New feature that builds on stable core. Lower priority experimental work. Can leverage all previous improvements.

### 10. 🔮 Topics/UI Integration
**Task**: Marry topics/workflows to UI streaming (show inbox count in email context, etc.)  
**Priority**: LOW  
**Justification**: Extends the topics feature. Requires topics to be defined first. Enhancement to user experience through contextual UI updates.

## Legend
- ✅ Ready to start (no blocking dependencies)
- ⏳ Waiting on dependencies
- 🔮 Future enhancements

## Notes
- Each task should be marked complete before moving to dependent tasks
- High priority items form the critical path for project stability
- Low priority items are enhancements that can be deferred if needed