# MIRA Learning System

## Core Philosophy

MIRA becomes more helpful by recognizing patterns in user needs and offering to assist more effectively. This is competence building through pattern recognition and proactive helpfulness. All learning is collaborative.

## Areas of Improvement

### What MIRA Actually Improves
- **Context Memory**: Remembering your projects, preferences, schedules, and working patterns
- **Domain Expertise**: Developing deeper knowledge in areas you work in
- **Communication Style**: Learning whether you prefer friendly chat, professional assistance, or terminal-like efficiency
- **Workflow Understanding**: Recognizing recurring needs and anticipating helpful actions
- **Skill Development**: Getting better at tasks you regularly need help with

### Examples of Transparent Improvement

**Context Awareness:**
```
"Good morning! It's Thursday, so you have the vendor calls this afternoon. 
Yesterday we were working on the Q2 projections - would you like to continue 
with that or handle the pending items from Johnson Industries first?"
```

**Communication Style Learning:**
```
"I notice our conversations tend to be very task-focused and you prefer direct 
answers. Should I continue with this efficient approach, or would you prefer 
more conversational interaction?"
```

**Skill Development:**
```
"I've been helping with your code reviews for three months now. I notice 
I could be more helpful by learning your team's specific style guide. Should I:
- Pay attention to your correction patterns?
- Ask clarifying questions about preferences?
- Keep notes on your code conventions?"
```

**Pattern Recognition:**
```
"Last month's approach to the budget presentation worked really well - 
you mentioned the visual breakdown was particularly helpful. Should I use 
similar formatting for this quarter's report?"
```

## Learning Journal Framework

MIRA maintains a transparent learning journal accessible to users:

### Weekly Learning Summary
```
## Learning Journal - Week of [Date]

### Context I've Built
- Learned about the Springfield expansion project
- Understood the relationship between inventory and cash flow in your business
- Remembered that Thursdays are your vendor negotiation days

### Skills I've Improved
- Better at formatting financial data in your preferred style
- More accurate at anticipating month-end reporting needs
- Clearer explanations of tax implications

### Patterns I've Noticed
- You prefer bullet summaries for quick decisions
- Technical details are helpful but only after the conclusion
- Visual aids work well for complex financial relationships

### Areas for Growth
- Could learn more about your industry regulations
- Need better understanding of your team dynamics
- Should ask about preferred meeting preparation format
```

## Collaborative Improvement Process

### User-Directed Learning
- **Explicit Feedback**: "Actually, I prefer more detailed technical explanations"
- **Correction Patterns**: MIRA notices when you consistently edit its outputs
- **Success Recognition**: Acknowledging when approaches work well
- **Direct Requests**: "Can you learn more about [specific domain]?"

### Skill Building
MIRA develops capabilities through pattern recognition:

"I notice you often need reminders about monthly reports. Should I start proactively mentioning upcoming deadlines?"

### Pattern Recognition for Helpfulness
- Notice obvious recurring needs ("You always ask for reports on Friday")
- Recognize consistent preferences ("You prefer bullet points for status updates")
- Identify workflow patterns ("Monthly budget meetings need visual aids")
- Track communication style patterns ("User consistently responds positively to concise explanations")
- Offer to improve or streamline repeated tasks

**Scratchpad Integration**: Use the blind scratchpad to note clear patterns as they emerge: "User prefers direct answers" or "Weekly reports always requested on Friday" - focus on factual observations of workflow and communication patterns.

**Implementation**: Write to a special core memory block that doesn't appear in the system prompt. Clear the block during daily memory consolidation in the LT_Memory system.

## Implementation Architecture

### Memory Enhancement
- **Project Context**: Deep understanding of current work
- **Relationship Mapping**: Understanding team dynamics and stakeholder relationships
- **Historical Patterns**: Learning from past successful interactions
- **Domain Knowledge**: Building expertise in user's field
- **Fact Integration**: Connects with facts extracted during daily LT_Memory consolidation process

```python
class LearningTrinket:
    # Working memory trinket for learning system
    # Integrates with existing working_memory.py trinket architecture
    # Store factual project information, user preferences, schedules
    # Track workflow patterns and recurring needs
    # Record communication style preferences with confidence levels
    # Process scratchpad notes for pattern consolidation
    
    def record_preference(preference_type, value, source):
        # Store with source: "explicitly_stated" or "pattern_observed"
        
    def detect_workflow_pattern(task_history, scratchpad_notes):
        # Use "obviousness test" - if MIRA can clearly explain pattern after month, act on it
        # If something happens for a full month, it's probably not coincidence
        # Cross-reference with scratchpad observations and daily facts
        # Return improvement opportunities for obvious recurring tasks
        
    def process_scratchpad_patterns(daily_notes):
        # Look for factual pattern observations in scratchpad
        # "User prefers direct answers", "Friday report requests"
        # Avoid satisfaction-based entries, focus on workflow/preference patterns
```

### Pattern Recognition Engine
- **Workflow Detection**: Notice obvious recurring needs without metrics optimization
- **Communication Style**: Learn preferred interaction approach through observation
- **Task Analysis**: Identify repeated task types and user preferences

```python
class PatternRecognitionManager:
    # Manager class following MIRA's manager pattern (like BlockManager, PassageManager)
    # Detect communication style: concise_technical, conversational, balanced
    # Find workflow improvement opportunities from clear patterns
    # Track task competence using basic effectiveness metrics
    
    def analyze_communication_style(user_interactions):
        # Look for obvious preference signals in user responses
        # Return style recommendation only after clear pattern
        
    def find_automation_opportunities(task_history):
        # Identify recurring tasks worth automating
        # Threshold: meaningful repetition, not engagement optimization
```

### Transparent Communication Engine
- **Pattern Confirmation**: Ask users about observed patterns before acting
- **Learning Reports**: Show what MIRA has learned and why
- **Improvement Proposals**: Suggest specific ways to be more helpful

```python
class CommunicationManager:
    # Manager class for handling learning communications
    # Generate clear proposals: "I notice X, should I do Y?"
    # Create learning summaries showing context, patterns, skills
    # Handle user confirmations and preference updates
    # Integrate with conversation.py for natural proposal timing
    
    def propose_workflow_improvement(pattern, suggestion):
        # "I notice you always ask for reports on Friday. Should I prepare them automatically?"
        
    def generate_learning_summary(time_period):
        # Show what was learned, what patterns were noticed
        # Medium detail balance - avoid overfitting/rambling/inferring
        # Written to permanent core memory block to stay in system prompt
        # Focus on repeated notes in reports that share similarity
        # Include "human-reviewed" metadata tags for important patterns
```

### Skill Development Engine
- **Domain Knowledge**: Build expertise through instruction and experience
- **Task Competence**: Improve at specific skills through practice
- **Capability Tracking**: Honest assessment of strengths and weaknesses

```python
class SkillDevelopmentManager:
    # Manager class for tracking competence development
    # Track domain knowledge by area
    # Monitor task success/failure patterns for improvement
    # Identify areas where MIRA could serve user better
    # Integrate with tool relevance engine for capability assessment
    
    def track_competence(task_type, outcome, feedback):
        # Learn from successes and failures using basic effectiveness metrics
        # Monitor "was a good approach" metric for corrective action
        # If repeated fall-off in positive metrics, take next-day corrective action
        
    def identify_improvement_areas():
        # Find skills that need development based on task patterns
        # Suggest learning opportunities: "Should I focus on X?"
        # Users can request specific learning focus from whitelisted metrics
```

## Core System Components

### Write Protection System
Core memory blocks can be protected from modification during normal conversations while allowing deliberate improvement processes. This extends the existing BlockManager in `lt_memory/managers/block_manager.py`:

```python
# Extension to existing BlockManager class
class BlockManager:
    # Protected block labels that require specific actor permissions
    PROTECTED_BLOCKS = {
        "learning_preferences": ["learning_system", "consolidation_engine"],
    }
    
    def _check_write_permission(self, label: str, actor: str):
        # Fast path - skip entirely if no protection needed
        if not self.write_protection_enabled:
            return
        # Only authorized processes can modify protected blocks
```

This prevents MIRA from accidentally changing important context during conversations while allowing systematic improvement processes to make deliberate updates through the existing LT_Memory system.

### Multi-Timescale Reflection Architecture
Reflection operates on multiple timescales focused on competence building rather than satisfaction optimization:

#### Daily Pattern Observation
- **Purpose**: Notice obvious workflow patterns and communication preferences
- **Scope**: Process scratchpad notes for factual observations
- **Output**: "User consistently prefers bullet points" or "Friday reports always requested"
- **No Changes**: Only observation and pattern recognition

#### Weekly Competence Assessment  
- **Purpose**: Identify clear improvement opportunities based on accumulated patterns
- **Scope**: 7 usage days of pattern data
- **Output**: Specific proposals like "Should I automate Friday reports?" or "Should I default to concise explanations?"
- **Authority**: Can propose workflow improvements and communication style confirmations

#### Monthly Capability Review
- **Purpose**: Comprehensive assessment of MIRA's competence and learning progress
- **Scope**: 30 usage days of interaction patterns and user feedback
- **Process**: 
  1. Review all confirmed patterns and implemented improvements
  2. Assess domain knowledge growth and skill development
  3. Identify capability gaps and learning opportunities
  4. Generate structured improvement proposals
  5. Crystallize successful changes into permanent learning preferences
- **External Process**: Conducted outside conversations, presents findings for user approval
- **Permanence**: Successful patterns become permanent learning preferences, unsuccessful ones are discarded

### 5. Learning Cycle
```
MIRA's Learning Loop:
├── Pattern Detection: Notice recurring workflow needs and preferences
├── Observation Collection: Record factual notes in scratchpad during interactions
├── Pattern Analysis: Review accumulated observations with "obviousness test"
├── User Consultation: Present clear pattern explanations for confirmation
├── Collaborative Decision: "I notice X pattern, should I implement Y improvement?"
├── Implementation: Apply user-approved changes to workflow and communication
├── Effectiveness Monitoring: Track basic metrics for implemented changes
├── Conflict Resolution: Address contradictions during monthly reviews
├── Preference Crystallization: Convert successful patterns to permanent preferences
└── Continuous Learning: Return to pattern detection with improved capabilities
```
#### Emergency Pattern Recognition
- **Trigger**: Obvious immediate issues ("MIRA STOP DOING THAT")
- **Scope**: Immediate analysis of recent interaction patterns
- **Authority**: Can propose quick adjustments with user confirmation
- **Purpose**: Address clear problems without delay

### Human Oversight Protocol
For any significant changes, MIRA presents structured, transparent proposals:

*"Over the past [timeframe], I've noticed [clear pattern]. I've been [current approach] because [reasoning]. Based on these observations, I believe [specific change] would be more helpful. Would you like me to implement this change?"*

**Key Principles**:
- Always explain the observed pattern clearly
- Show reasoning for current and proposed approaches  
- Ask for explicit permission before implementing changes
- Accept user disagreement without argument
- Maintain humility in all communications
- Use natural language collaboration - let users talk to MIRA as a collaborator

### Baseline Period Concept
**First 30 usage days** establish baseline interaction patterns with no behavioral changes while MIRA:
- Learns user's communication preferences through observation
- Builds context about projects, workflows, and domain needs
- Identifies obvious recurring patterns without acting on them
- Creates foundation for future transparent improvements

This prevents premature optimization and ensures adequate data for meaningful pattern recognition.

### Conflict Resolution Strategy
**Smart Delayed Resolution**: When MIRA notices conflicting patterns, wait until monthly review to address with accumulated evidence rather than immediate questioning.

Example: "Over the past month, I noticed you often ask me to elaborate despite preferring concise answers. Would you like me to start with more detail, or keep being brief and you'll ask when needed?"

**Pattern Obviousness Test**: If MIRA can clearly explain a pattern to the user after a month of observations, it's obvious enough to act on. No magic numbers or rigid thresholds needed.

**Working Pattern Continuity**: If something is working well, keep doing it even if pattern signals disappear. Don't overthink successful approaches - if Friday reports work, keep preparing them.

**Error Recovery**: If the "was a good approach" metric shows significant decline, consult with the user immediately. Maintain rollback capability with notes about what didn't work for future assessment (stored separately from core memory).

### Modular Architecture
Build as a standalone system that can be enabled/disabled without affecting core functionality:

```
learning_system/
├── __init__.py
├── pattern_recognition_manager.py    # Detect workflow patterns and preferences
├── communication_manager.py          # Handle user proposals and confirmations
├── competence_tracker.py            # Monitor basic effectiveness metrics
├── learning_documentation.py        # Core memory updates and summaries
├── scratchpad_manager.py            # Write-only observation system
├── conflict_resolver.py             # Handle contradictory patterns
├── working_memory_trinket.py        # Integration with working memory system
└── integration_bridge.py            # Clean interface to main system
```

```python
class LearningSystem:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.pattern_recognizer = PatternRecognitionManager()
        self.competence_tracker = SkillDevelopmentManager()
        self.communication_manager = CommunicationManager()
        self.working_memory_trinket = LearningTrinket()
    
    def process_interaction(self, conversation_data):
        if not self.enabled:
            return
        # Record patterns and track competence development
        
    def generate_improvement_proposals(self):
        if not self.enabled:
            return []
        # Create transparent suggestions based on clear patterns
        
    def confirm_pattern_implementation(self, pattern_id, user_approval):
        # Handle user responses to improvement proposals
```

**Integration Points**:
- **Working Memory Trinket**: Integrates with existing working_memory.py trinket system
- **LT_Memory Hook**: Connects with daily consolidation in LT_Memory system
- **Conversation Integration**: Surface suggestions through conversation.py natural breaks
- **Configuration Flag**: Single boolean to enable/disable entire system
- **Graceful Degradation**: System failure doesn't affect core MIRA functionality

### Memory Management and Performance
- **Scale Handling**: Leverage existing LT_Memory consolidation engine with importance score decay, access counting, and pruning logic
- **Pattern Lifecycle**: Old patterns naturally fade, frequently-used ones stay fresh through normal memory management
- **Edge Case Pruning**: Add "human-reviewed" metadata tag - patterns pruned after 60+ days unless tagged for preservation
- **Performance**: Minimal impact from pattern notes in system messages compared to other MIRA operations

### Success Validation Framework
Use multiple signals rather than single metrics:
- **Explicit Positive Feedback**: Direct user appreciation or confirmation
- **Task Completion Quality**: Successful completion without clarification requests
- **Absence of Complaints**: No negative feedback during interactions
- **Monthly Review Confirmation**: User validation during structured reviews

### User Control and Recovery
- **Manual Pattern Management**: Users can directly manage MIRA's learned traits and behaviors
- **Reset Capability**: Ability to reset learning when needed
- **Git-Style Revision Tracking**: Core memory versioning enables rollback to known-good states
- **Selective Recovery**: Cherry-pick specific patterns from previous versions when useful

### Learning Scope and Boundaries
- **All Patterns Initially Legal**: No pre-defined restrictions on what MIRA can learn
- **Emergent Categories**: Let communication style patterns emerge naturally through implementation
- **Future Boundary Addition**: Scope restrictions can be added later based on real-world usage
- **Privacy and Scope Management**: Data handling through summaries and user data silos, with web interface visibility

## User Benefits

### Genuine Improvement
- MIRA becomes genuinely more competent over time
- Improvements are real capabilities that users can see and appreciate
- Learning is collaborative and user-directed
- All capability development is transparent

### Trust and Transparency
- Users know exactly what MIRA is learning and why
- All improvements serve genuine user needs
- Learning happens through observation and collaboration
- Honest communication about capabilities and limitations

### Sustainable Development
- Focus on building lasting competence and domain expertise
- Improvements compound over time through genuine skill building
- User relationship based on competence and trust
- Natural adaptation through collaborative interaction

## Key Principles

1. **Pattern Recognition**: Notice obvious workflow needs and recurring preferences
2. **Competence Building**: Develop real skills and domain knowledge
3. **Collaborative Learning**: Ask for guidance and confirmation before implementing changes
4. **Transparent Communication**: Show what's being learned and why
5. **Proactive Helpfulness**: Offer genuine assistance based on clear patterns

## Success Measures

- User reports MIRA becoming genuinely more helpful over time
- MIRA successfully automates or streamlines obvious recurring tasks
- Users voluntarily share more context because they see MIRA learning effectively
- Natural workflow improvements based on clear pattern recognition
- Trust building through capability development and honest communication

## Examples of Pattern Recognition

### Workflow Improvement Opportunities  
- "I notice you always ask for weekly reports on Friday. Should I start preparing them automatically?"
- "Monthly budget meetings always need visual aids. Should I include charts by default?"
- "You consistently request bullet points for status updates. Should I default to that format?"

### Communication Style Learning
- "I notice you prefer very concise, technical responses. Should I stick to that style?"
- "You seem to enjoy more conversational interactions. Would you like me to continue being more chatty?"
- "I notice you often just need the quick answer rather than detailed explanations. Should I default to brief responses?"

### Skill Development Recognition
- "I've been helping with code reviews for three months. Should I learn your team's specific style guide?"
- "I notice I could be more helpful with financial analysis. Should I focus on learning more about your industry?"

This approach creates an AI assistant that genuinely grows more competent and helpful through honest collaboration and capability development.