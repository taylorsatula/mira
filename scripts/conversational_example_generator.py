#!/usr/bin/env python3
"""
Capability-Based Conversational Example Generator

Generates natural conversation examples for AI tool classification training.
Optimized for BGE embeddings with 512 token limit through focused, concise examples.

This generator:
1. Analyzes tools to extract distinct capabilities (turn on, turn off, dim, etc.)
2. Generates focused examples per capability in separate API calls
3. Validates examples with diagnostic feedback
4. Regenerates failed clusters with specific guidance
5. Biases toward short, typed examples for maximum semantic density

Usage:
    python new_gen.py <tool_file_path> [--examples-per-capability 15] [--output output.json]
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for MIRA imports
sys.path.append(str(Path(__file__).parent.parent))

from api.llm_provider import LLMProvider
from errors import APIError, ErrorCode, error_context

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ToolCapability:
    """A specific capability/function of a tool."""
    name: str
    action_verb: str
    description: str
    example_targets: List[str]


@dataclass
class ToolAnalysis:
    """Complete analysis of a tool's functionality broken down by capabilities."""
    tool_name: str
    primary_purpose: str
    capabilities: List[ToolCapability]
    target_users: List[str]
    common_scenarios: List[str]


class CapabilityExampleGenerator:
    """
    Generates focused conversational examples for tool classification.
    
    Each capability is processed independently to create high-quality,
    token-efficient training data optimized for BGE embeddings.
    """
    
    def __init__(self):
        """Initialize with separate LLMs for analysis and generation."""
        # Sonnet for fast, accurate code analysis
        self.analyzer = LLMProvider(
            provider_type="remote",
            api_endpoint="https://api.anthropic.com/v1/chat/completions",
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=300
        )
        
        # Opus for superior creative generation with extended thinking
        self.generator = LLMProvider(
            provider_type="remote",
            api_endpoint="https://api.anthropic.com/v1/chat/completions",
            model="claude-opus-4-20250514",
            temperature=1.0,
            max_tokens=8000,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=300
        )
        
        logger.info("Initialized with Sonnet analyzer and Opus generator")
    
    def analyze_tool(self, tool_path: str) -> ToolAnalysis:
        """
        Analyze a tool file to extract capabilities and metadata.
        
        Args:
            tool_path: Path to the Python tool file
            
        Returns:
            ToolAnalysis with broken-down capabilities
            
        Raises:
            APIError: If analysis fails
        """
        logger.info(f"Analyzing tool: {tool_path}")
        
        with open(tool_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        prompt = """
        Analyze this Python tool and break it down into specific capabilities/functions that users would invoke.
        
        CRITICAL: Respond with ONLY valid JSON. No markdown, no explanations.
        
        Required JSON structure:
        {
            "tool_name": "exact name of the tool from the code",
            "primary_purpose": "concise description of what this tool does",
            "capabilities": [
                {
                    "name": "descriptive name of this capability",
                    "action_verb": "primary action verb (turn on, send, check, etc.)",
                    "description": "what this specific capability does",
                    "example_targets": ["specific", "things", "this", "acts", "on"]
                }
            ],
            "target_users": ["types", "of", "people", "who", "would", "use", "this"],
            "common_scenarios": ["situations", "when", "people", "would", "need", "this", "tool"]
        }
        
        Focus on:
        1. Break tool into distinct user-facing capabilities
        2. Each capability should have a clear action verb and targets
        3. Think about what users actually DO with this tool
        
        Tool source code:
        """
        
        messages = [{"role": "user", "content": f"{prompt}\n\n```python\n{source_code}\n```"}]
        
        try:
            with error_context(
                component_name="capability_generator",
                operation="tool_analysis",
                error_class=APIError,
                error_code=ErrorCode.API_RESPONSE_ERROR,
                logger=logger
            ):
                response = self.analyzer.generate_response(messages)
                response_text = self.analyzer.extract_text_content(response)
                
                logger.debug(f"Response text length: {len(response_text)}")
                logger.debug(f"Response preview: {response_text[:500] if response_text else 'EMPTY'}")
                
                if not response_text.strip():
                    raise APIError("LLM returned empty response")
                
                # Strip markdown code blocks if present
                cleaned_text = response_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]  # Remove ```json
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]  # Remove ```
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]  # Remove trailing ```
                cleaned_text = cleaned_text.strip()
                
                data = json.loads(cleaned_text)
                
                # Convert to dataclass objects
                capabilities = [
                    ToolCapability(
                        name=cap["name"],
                        action_verb=cap["action_verb"],
                        description=cap["description"],
                        example_targets=cap["example_targets"]
                    )
                    for cap in data.get("capabilities", [])
                ]
                
                return ToolAnalysis(
                    tool_name=data["tool_name"],
                    primary_purpose=data["primary_purpose"],
                    capabilities=capabilities,
                    target_users=data.get("target_users", []),
                    common_scenarios=data.get("common_scenarios", [])
                )
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse analysis: {e}")
            raise APIError(f"Tool analysis failed: {e}") from e
    
    def generate_capability_examples(
        self, 
        tool_name: str, 
        capability: ToolCapability, 
        count: int = 15,
        previous_feedback: str = ""
    ) -> List[Dict[str, str]]:
        """
        Generate examples for a single capability with optional feedback.
        
        Args:
            tool_name: Name of the tool
            capability: The capability to generate examples for
            count: Number of examples to generate
            previous_feedback: Optional feedback from failed validation
            
        Returns:
            List of example dictionaries with tool_name and query
        """
        logger.info(f"Generating {count} examples for: {capability.name}")
        
        feedback_section = ""
        if previous_feedback:
            feedback_section = f"""
        
        GUIDANCE FOR IMPROVEMENT:
        {previous_feedback}
        
        Apply this guidance while maintaining natural language variety.
        """
        
        prompt = f"""
        Generate {count} natural conversational examples for ONE SPECIFIC capability.
        
        CAPABILITY DETAILS:
        - Name: {capability.name}
        - Action: {capability.action_verb}
        - Description: {capability.description}
        - Valid Targets: {', '.join(capability.example_targets)}
        - Tool: {tool_name}{feedback_section}
        
        CRITICAL REQUIREMENTS:
        1. DOMAIN-SPECIFIC: Every example must clearly indicate this tool's domain, not generic device management
        2. SPECIFIC ACTIONS: Use concrete action verbs, avoid vague words like "control" or "manage"
        3. ACTIONABLE: Must be complete commands, not observations
        4. BIAS TOWARD SHORT EXAMPLES (5-10 words ideal)
        
        DOMAIN SPECIFICITY EXAMPLES:
        BAD (too generic): "find all devices", "get device status", "control outlet"
        GOOD (domain-specific): "find all smart devices", "get smart plug status", "turn on outlet"
        
        ACTION VERB SPECIFICITY:
        - Instead of "control" → use "turn on/off", "switch", "toggle"
        - Instead of "manage" → use "operate", "adjust", "set"
        - Instead of "handle" → use specific action for this capability
        
        DISTRIBUTION (roughly even):
        - CONCISE (3-5 words): "turn on smart lights"
        - SHORT TYPED (5-10 words): "please turn on the smart bulbs"  
        - NATURAL (10-20 words): "heading to bed, could you turn on the smart lights?"
        
        VARIETY:
        - Casual: "hey turn off smart lights"
        - Urgent: "quick! smart lights off"
        - Polite: "could you please turn off the smart lamp"
        - Uncertain: "I think the smart lights are on?"
        - Shorthand: "smart lights off"
        - Voice-to-text: "turn off smart living room light bulb"
        
        Every example should pass this test: "Could this apply to any device management tool?" 
        If YES, make it more domain-specific.
        
        Return ONLY a JSON array:
        [
            {{"tool_name": "{tool_name}", "query": "domain-specific example"}},
            {{"tool_name": "{tool_name}", "query": "another specific example"}}
        ]
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            with error_context(
                component_name="capability_generator",
                operation="example_generation",
                error_class=APIError,
                error_code=ErrorCode.API_RESPONSE_ERROR,
                logger=logger
            ):
                response = self.generator.generate_response(
                    messages,
                    extra_body={"thinking": {"type": "enabled", "budget_tokens": 1500}}
                )
                response_text = self.generator.extract_text_content(response)
                
                # Strip markdown code blocks if present
                cleaned_text = response_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                
                examples = json.loads(cleaned_text)
                
                if not isinstance(examples, list):
                    raise APIError("Generator returned non-list response")
                
                # Validate structure
                valid_examples = [
                    ex for ex in examples 
                    if isinstance(ex, dict) and "tool_name" in ex and "query" in ex
                ]
                
                logger.info(f"Generated {len(valid_examples)} valid examples")
                return valid_examples
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse examples: {e}")
            raise APIError(f"Example generation failed: {e}") from e
    
    def validate_examples(
        self, 
        examples: List[Dict[str, str]], 
        capability: ToolCapability
    ) -> Dict[str, Any]:
        """
        Validate examples and provide diagnostic feedback.
        
        Args:
            examples: List of generated examples
            capability: The capability being validated
            
        Returns:
            Dict with passed (bool), feedback (str), and other diagnostics
        """
        if not examples:
            return {"passed": True, "feedback": "", "examples": examples}
        
        logger.info(f"Validating {len(examples)} examples for: {capability.name}")
        
        prompt = f"""
        Validate these examples for a specific capability with contextual understanding.
        
        CAPABILITY CONTEXT:
        - Name: {capability.name}
        - Core Action: {capability.action_verb}
        - Description: {capability.description}
        - Valid Targets: {', '.join(capability.example_targets)}
        
        EXAMPLES TO VALIDATE:
        {json.dumps([ex['query'] for ex in examples], indent=2)}
        
        VALIDATION APPROACH:
        Judge if each example would reasonably accomplish this capability's purpose.
        Be generous with natural language - humans express actions in many ways.
        
        ACCEPT examples that:
        1. Express the capability's intent (even with different verbs)
        2. Reference relevant targets (flexible: "lights"="smart bulbs", "device"=any target)
        3. Are actionable commands/requests (not just observations)
        4. Use natural human language
        5. Are DOMAIN-SPECIFIC (not generic device management)
        
        EXAMPLES OF FLEXIBLE ACCEPTANCE:
        - For "turn" capability: "switch off", "power down", "shut off"
        - For "set" capability: "change", "make", "adjust", "put"
        - For "check" capability: "see", "monitor", "get", "show"
        - For brightness: "dim", "brighten", "set brightness"
        - For discovery: "find", "locate", "scan"
        
        REJECT examples that are too generic:
        - "find all devices" (could be any device tool)
        - "get device status" (too generic)
        - "control outlet" (vague action verb)
        
        Focus on INTENT, ACTIONABILITY, and DOMAIN SPECIFICITY.
        
        Return ONLY JSON:
        {{
            "passed": true/false,
            "issues_found": ["only fundamental problems"],
            "feedback": "guidance on missing elements, not word substitutions",
            "good_count": number_of_acceptable_examples,
            "total_count": {len(examples)}
        }}
        
        Set passed=false ONLY if >10% have fundamental issues (no clear action, no target, not actionable).
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.analyzer.generate_response(messages)
            response_text = self.analyzer.extract_text_content(response)
            
            # Strip markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            result = json.loads(cleaned_text)
            result["examples"] = examples
            
            if result["passed"]:
                logger.info(f"✓ Validation passed: {result['good_count']}/{result['total_count']} good")
            else:
                logger.warning(f"✗ Validation failed: {result['feedback']}")
            
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Validation error: {e}")
            return {
                "passed": False, 
                "feedback": f"Validation error: {e}",
                "examples": examples
            }
    
    def generate_with_validation(
        self,
        tool_name: str,
        capability: ToolCapability,
        count: int = 15
    ) -> List[Dict[str, str]]:
        """
        Generate examples with validation and smart fallback.
        
        Args:
            tool_name: Name of the tool
            capability: The capability to generate for
            count: Number of examples to generate
            
        Returns:
            List of validated examples (best available)
        """
        # First attempt
        examples_v1 = self.generate_capability_examples(tool_name, capability, count)
        validation_v1 = self.validate_examples(examples_v1, capability)
        
        if validation_v1["passed"]:
            return examples_v1
        
        # Second attempt with improved feedback
        logger.info(f"Regenerating {capability.name} with guidance")
        
        # Create principle-based feedback instead of prescriptive
        guided_feedback = self._create_guided_feedback(validation_v1["feedback"], capability)
        
        examples_v2 = self.generate_capability_examples(
            tool_name, capability, count, guided_feedback
        )
        
        validation_v2 = self.validate_examples(examples_v2, capability)
        
        if validation_v2["passed"]:
            logger.info(f"✓ Improved examples passed validation")
            return examples_v2
        
        # Both failed - pick the better set
        score_v1 = validation_v1["good_count"]
        score_v2 = validation_v2["good_count"]
        
        if score_v2 >= score_v1:
            logger.warning(f"⚠ Using improved attempt for {capability.name} ({score_v2}/{count} good)")
            return examples_v2
        else:
            logger.warning(f"⚠ Using original attempt for {capability.name} ({score_v1}/{count} good)")
            return examples_v1
    
    def _create_guided_feedback(self, raw_feedback: str, capability: ToolCapability) -> str:
        """
        Convert prescriptive feedback to principle-based guidance.
        
        Args:
            raw_feedback: Original validation feedback
            capability: The capability being improved
            
        Returns:
            Improved guidance that preserves natural variety
        """
        # Focus on principles rather than specific word replacements
        guidance = f"""
        Based on validation feedback, improve examples while maintaining natural language variety:
        
        PRESERVE:
        - Natural synonyms and varied phrasing
        - Different length categories (concise, short, conversational)
        - Casual/formal tone variety
        
        ENSURE EACH EXAMPLE:
        - Expresses the INTENT of "{capability.action_verb}" (natural language welcome)
        - References relevant TARGETS from: {', '.join(capability.example_targets)}
        - Is completely ACTIONABLE (not just observations or questions)
        - Uses natural human language
        - Is DOMAIN-SPECIFIC (clearly indicates this tool's domain)
        - Uses SPECIFIC action verbs (avoid "control", "manage", "handle")
        
        Original feedback context: {raw_feedback}
        
        Focus on fundamental completeness rather than specific word choices.
        """
        
        return guidance.strip()
    
    def process_capability_parallel(
        self,
        tool_name: str,
        capability: ToolCapability,
        examples_per_capability: int
    ) -> List[Dict[str, str]]:
        """
        Process a single capability with generation, validation, and regeneration.
        
        Args:
            tool_name: Name of the tool
            capability: The capability to process
            examples_per_capability: Number of examples to generate
            
        Returns:
            List of validated examples for this capability
        """
        logger.info(f"Processing: {capability.name}")
        
        try:
            examples = self.generate_with_validation(
                tool_name,
                capability,
                examples_per_capability
            )
            logger.info(f"✓ Completed {capability.name}: {len(examples)} examples")
            return examples
            
        except APIError as e:
            logger.error(f"Failed to process {capability.name}: {e}")
            return []
    
    def _generate_sequential(
        self,
        analysis: ToolAnalysis,
        examples_per_capability: int
    ) -> List[Dict[str, str]]:
        """
        Sequential fallback processing for capabilities.
        
        Args:
            analysis: Tool analysis with capabilities
            examples_per_capability: Number of examples per capability
            
        Returns:
            All generated examples
        """
        all_examples = []
        
        for capability in analysis.capabilities:
            try:
                examples = self.process_capability_parallel(
                    analysis.tool_name,
                    capability,
                    examples_per_capability
                )
                all_examples.extend(examples)
                
            except APIError as e:
                logger.error(f"Failed to process {capability.name}: {e}")
                continue
        
        return all_examples
    
    def _generate_sequential_grouped(
        self,
        analysis: ToolAnalysis,
        examples_per_capability: int
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Sequential fallback processing for capabilities with grouped output.
        
        Args:
            analysis: Tool analysis with capabilities
            examples_per_capability: Number of examples per capability
            
        Returns:
            Dict mapping capability names to their examples
        """
        capability_examples = {}
        
        for capability in analysis.capabilities:
            try:
                examples = self.process_capability_parallel(
                    analysis.tool_name,
                    capability,
                    examples_per_capability
                )
                capability_examples[capability.name] = examples
                
            except APIError as e:
                logger.error(f"Failed to process {capability.name}: {e}")
                capability_examples[capability.name] = []
        
        return capability_examples
    
    def _save_separated_outputs(
        self, 
        output_dir_path: str, 
        capability_examples: Dict[str, List[Dict[str, str]]],
        tool_name: str
    ) -> None:
        """
        Save examples separated by capability into individual files.
        
        Args:
            output_dir_path: Directory path to save capability files
            capability_examples: Dict mapping capability names to examples
            tool_name: Name of the tool for file naming
        """
        from pathlib import Path
        
        # Create the output directory
        output_dir = Path(output_dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for capability_name, examples in capability_examples.items():
            if not examples:
                logger.warning(f"No examples for capability: {capability_name}")
                continue
                
            # Create safe filename from capability name
            safe_capability = "".join(c if c.isalnum() or c in '-_' else '_' for c in capability_name.lower())
            capability_file = output_dir / f"{tool_name}_{safe_capability}.json"
            
            with open(capability_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            
            saved_files.append(str(capability_file))
            logger.info(f"Saved {len(examples)} examples for '{capability_name}' to: {capability_file}")
        
        # Also save a summary file
        summary_file = output_dir / f"{tool_name}_summary.json"
        summary = {
            "tool_name": tool_name,
            "total_examples": sum(len(examples) for examples in capability_examples.values()),
            "capabilities": {
                name: len(examples) 
                for name, examples in capability_examples.items()
            },
            "files": saved_files
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary to: {summary_file}")
        logger.info(f"Created {len(saved_files)} capability-specific files in: {output_dir}")
    
    def generate_all(
        self,
        tool_path: str,
        examples_per_capability: int = 15,
        output_path: Optional[str] = None,
        max_workers: Optional[int] = None
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Main orchestration method with parallel processing and sequential fallback.
        
        Args:
            tool_path: Path to tool file
            examples_per_capability: Examples to generate per capability
            output_path: Optional output file path
            max_workers: Maximum number of parallel workers (defaults to number of capabilities)
            
        Returns:
            Dict with capability names as keys and their examples as values
        """
        # Analyze tool
        analysis = self.analyze_tool(tool_path)
        logger.info(f"Tool: {analysis.tool_name}")
        logger.info(f"Purpose: {analysis.primary_purpose}")
        logger.info(f"Capabilities: {len(analysis.capabilities)}")
        
        # Calculate optimal worker count
        num_capabilities = len(analysis.capabilities)
        if max_workers is None:
            # Default to number of capabilities, with reasonable bounds
            optimal_workers = min(num_capabilities, 12)  # Cap at 12 to avoid overwhelming APIs
        else:
            # Use provided max_workers but don't exceed capability count
            optimal_workers = min(max_workers, num_capabilities)
        
        # Try parallel processing first
        capability_examples = {}
        try:
            logger.info(f"Attempting parallel processing with {optimal_workers} workers for {num_capabilities} capabilities")
            
            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # Submit all capability processing tasks
                future_to_capability = {
                    executor.submit(
                        self.process_capability_parallel,
                        analysis.tool_name,
                        capability,
                        examples_per_capability
                    ): capability
                    for capability in analysis.capabilities
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_capability):
                    capability = future_to_capability[future]
                    try:
                        examples = future.result()
                        capability_examples[capability.name] = examples
                    except Exception as e:
                        logger.error(f"Exception processing {capability.name}: {e}")
                        capability_examples[capability.name] = []
            
            logger.info("✓ Parallel processing completed successfully")
            
        except Exception as e:
            logger.warning(f"Parallel processing failed ({e}), falling back to sequential")
            capability_examples = self._generate_sequential_grouped(analysis, examples_per_capability)
        
        total_examples = sum(len(examples) for examples in capability_examples.values())
        logger.info(f"\nTotal: {total_examples} examples generated across {len(capability_examples)} capabilities")
        
        # Save if requested
        if output_path:
            self._save_separated_outputs(output_path, capability_examples, analysis.tool_name)
        
        return capability_examples


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate capability-based conversational examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python new_gen.py tools/kasa_tool.py
  python new_gen.py tools/email_tool.py --examples-per-capability 15
  python new_gen.py tools/calendar.py --output calendar_examples.json
        """
    )
    
    parser.add_argument("tool_path", help="Path to Python tool file")
    parser.add_argument(
        "--examples-per-capability", 
        type=int, 
        default=15,
        help="Examples to generate per capability (default: 15)"
    )
    parser.add_argument("--output", help="Output directory path for capability-separated files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.tool_path):
        logger.error(f"Tool file not found: {args.tool_path}")
        sys.exit(1)
    
    try:
        generator = CapabilityExampleGenerator()
        capability_examples = generator.generate_all(
            args.tool_path,
            args.examples_per_capability,
            args.output
        )
        
        # Show sample results
        total_examples = sum(len(examples) for examples in capability_examples.values())
        print(f"\n✓ Generated {total_examples} examples across {len(capability_examples)} capabilities")
        
        print("\nSamples by capability:")
        for capability_name, examples in capability_examples.items():
            if examples:
                print(f"\n{capability_name} ({len(examples)} examples):")
                for i, ex in enumerate(examples[:2], 1):
                    print(f"  {i}. {ex['query']}")
                if len(examples) > 2:
                    print(f"  ... and {len(examples) - 2} more")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()