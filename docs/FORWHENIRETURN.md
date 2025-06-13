# TODO: Complete Async Embedding Implementation

## ✅ COMPLETED TODAY
- [x] **Memory manager updated** to use OpenAI embeddings (was already correct)
- [x] **Removed embedding caching** (pointless for unique conversation contexts)
- [x] **Added async embedding generation** to memory manager
- [x] **Created async get_relevant_memories_async()** function
- [x] **Updated sync get_relevant_memories()** to use memory manager directly
- [x] **Fixed user-only message filtering** in proactive memory
- [x] **Fixed message weighting calculation bug**
- [x] **Separated embedding systems**: Tool classification uses ONNX (384-dim), Memory uses OpenAI (1024-dim)
- [x] **Updated documentation** to clarify that conversation_context_embedding is for tool classification ONLY

## 🎯 NEXT STEPS

### 1. Update Conversation Flow for Async Memory
**File**: `conversation.py`
**Location**: `generate_response()` method

Need to modify the conversation flow to run memory search in parallel with LLM generation:

```python
async def generate_response_with_async_memory(self, user_input, **kwargs):
    # Add user message
    self.add_message("user", user_input)
    
    # Start both operations in parallel
    memory_task = asyncio.create_task(
        get_relevant_memories_async(self.messages, self.memory_manager)
    )
    llm_task = asyncio.create_task(
        self.llm_provider.generate_response(messages=self.get_formatted_messages(), ...)
    )
    
    # Wait for both to complete (memory should finish first)
    memories, llm_response = await asyncio.gather(memory_task, llm_task)
    
    # Add memories to working memory for LLM context
    if memories:
        formatted_memories = format_relevant_memories(memories)
        self.working_memory.add_item("memory_context", formatted_memories)
    
    # Start pre-computing for next turn
    asyncio.create_task(self._precompute_next_memory_context())
    
    return llm_response
```

### 2. Add Pre-computation for Next Turn
**Goal**: Cache the embedding for the next conversation context

```python
async def _precompute_next_memory_context(self):
    """Pre-compute embedding for next turn's memory search."""
    try:
        context_string = build_weighted_memory_context(self.messages)
        if context_string:
            # Store for next turn (simple cache)
            self._next_context_embedding = await self.memory_manager.generate_embedding_async(context_string)
            self._next_context_messages_hash = hash(str([m.content for m in self.messages]))
    except Exception as e:
        logger.warning(f"Failed to pre-compute next context: {e}")
```

### 3. Update Working Memory Integration
**Current issue**: Memory results need to be added to working memory before LLM call

**Options:**
1. **Sequential approach**: Get memories first, add to working memory, then call LLM
2. **Parallel with injection**: Run both in parallel, inject memory into LLM context when available
3. **Two-pass approach**: Initial LLM call without memory, then follow-up with memory if relevant

### 4. Handle Integration Points
**Files to check/update:**
- `main.py` - conversation initialization 
- `working_memory.py` - memory context integration
- Any other places that call `get_relevant_memories()`

### 5. Test the Complete Flow
- [ ] Verify async timing works (embedding finishes before LLM)
- [ ] Test memory integration with working memory
- [ ] Confirm tool classification still uses ONNX (384-dim)
- [ ] Run existing tests

### 6. Error Handling
- [ ] Fallback if OpenAI API is down for embeddings
- [ ] Handle async cancellation gracefully
- [ ] Log timing to verify performance improvement

## 🔍 ARCHITECTURE NOTES

**Current State:**
- **Memory Manager**: Uses OpenAI embeddings (1024-dim) ✅
- **Tool Classification**: Uses ONNX embeddings (384-dim) via conversation_context_embedding ✅
- **Memory Config**: Set to 1024 dimensions ✅
- **Caching**: Removed (was pointless) ✅

**Performance Goal:**
- Hide 361ms embedding latency behind 1-3 second LLM calls
- Pre-compute next turn to eliminate latency for subsequent messages

**Key Insight:**
Embedding costs are negligible (~3 cents/year), so optimize for UX not cost.

---

# OLD NOTES (PRE-EMBEDDING FIX)

- Setup a FastAPI server
- lt_memory needs to be finalized because it does not have LLM integration
- Automation engine needs a total rebuild
- Tools need to be gone through one-by-one to bring them all in line with most-current standards
- Remove sentence-transformers and replace it with onnx because I already have it set up for some tasks but not in the main.py
- Go through the docs/ folder and clear out junk that isn't relevant. it has become a catchall for markdown files.
- Remove all of the old classifier examples and generate new ones (the generator needs to correctly name the files so that MIRA recognizes them)
- Create a modern sample_tool.py
- Explore the ability to have "topics" mean something (e.g., "email" is a topic, "recipe" is a topic, etc.)
- Marry topics/workflows to send data over the link as an aside to the actual streamed message (e.g., show inbox count when in email context and clear it when changing topics)