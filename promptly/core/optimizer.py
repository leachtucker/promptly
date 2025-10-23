"""
LLM-powered prompt optimization module
"""

import asyncio
import json
import random
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from .clients import BaseLLMClient, GoogleAIClient, LLMResponse, OpenAIClient
from .runner import PromptRunner
from .templates import PromptTemplate
from .tracer import Tracer
from .utils.ai import simple_schema


class ProgressCallback(ABC):
    """Abstract base class for optimization progress callbacks"""

    @abstractmethod
    async def on_population_initialized(self, population_size: int) -> None:
        """Called when initial population is created"""
        pass

    @abstractmethod
    async def on_generation_start(self, generation: int, total_generations: int) -> None:
        """Called at the start of each generation"""
        pass

    @abstractmethod
    async def on_generation_complete(self, stats: dict[str, Any]) -> None:
        """Called when a generation completes with statistics"""
        pass

    @abstractmethod
    async def on_optimization_complete(self, result: "OptimizationResult") -> None:
        """Called when optimization completes"""
        pass


class NoOpProgressCallback(ProgressCallback):
    """No-op implementation for when no callback is provided"""

    async def on_population_initialized(self, population_size: int) -> None:
        pass

    async def on_generation_start(self, generation: int, total_generations: int) -> None:
        pass

    async def on_generation_complete(self, stats: dict[str, Any]) -> None:
        pass

    async def on_optimization_complete(self, result: "OptimizationResult") -> None:
        pass


class OptimizationResult(BaseModel):
    """Result of an optimization run"""

    model_config = {"arbitrary_types_allowed": True}

    best_prompt: PromptTemplate
    fitness_score: float
    generation: int
    population_size: int
    total_evaluations: int
    optimization_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptTestCase(BaseModel):
    """A test case for prompt evaluation"""

    input_variables: dict[str, Any]
    expected_output: Any  # Can be string or dict for structured outputs
    metadata: dict[str, Any] = Field(default_factory=dict)


class FitnessEvaluation(BaseModel):
    """Result of a fitness evaluation"""

    model_config = {"arbitrary_types_allowed": True}

    prompt: PromptTemplate
    score: float
    test_results: list[dict[str, Any]]
    evaluation_reasoning: str  # LLM's reasoning for the score
    metadata: dict[str, Any] = Field(default_factory=dict)


# Structured output models for LLM responses
class EvaluationResponse(BaseModel):
    """Structured response for prompt evaluation"""

    score: float = Field(description="Score from 0.0 to 1.0 where 1.0 is perfect")
    reasoning: str = Field(description="Detailed explanation of the evaluation")


class QualityEvaluationResponse(BaseModel):
    """Structured response for prompt quality evaluation"""

    score: float = Field(description="Score from 0.0 to 1.0 where 1.0 is excellent")
    reasoning: str = Field(
        description="Detailed explanation of the quality evaluation",
    )


class MutationResponse(BaseModel):
    """Structured response for prompt mutation"""

    mutated_prompt: str = Field(
        description="The improved prompt template. Valid JSON string value."
    )


class PopulationGenerationResponse(BaseModel):
    """Structured response for population generation"""

    variations: list[str] = Field(
        description="List of prompt template variations. Valid JSON string value."
    )


class CrossoverResponse(BaseModel):
    """Structured response for prompt crossover"""

    offspring1: str = Field(description="First offspring prompt template. Valid JSON string value")
    offspring2: str = Field(description="Second offspring prompt template. Valid JSON string value")


class LLMFitnessFunction(ABC):
    """Abstract base class for LLM-powered fitness functions"""

    def __init__(self, evaluation_client: BaseLLMClient, evaluation_model: str):
        self.evaluation_client = evaluation_client
        self.evaluation_model = evaluation_model

    @abstractmethod
    async def evaluate(
        self,
        *,
        runner: PromptRunner,
        prompt: PromptTemplate,
        model: str = "gpt-4",
        test_cases: Optional[list[PromptTestCase]] = None,
        variables: Optional[dict[str, Any]] = None,
    ) -> FitnessEvaluation:
        """Evaluate fitness of a prompt using LLM"""
        pass


class LLMComprehensiveFitnessFunction(LLMFitnessFunction):
    """Comprehensive LLM-powered fitness function combining accuracy and semantic evaluation"""

    def __init__(
        self,
        evaluation_client: BaseLLMClient,
        evaluation_model: str,
        test_case_concurrency: int = 10,
    ):
        super().__init__(evaluation_client, evaluation_model)
        self.accuracy_weight = 0.6  # Weight for accuracy evaluation
        self.semantic_weight = 0.4  # Weight for semantic evaluation
        self.test_case_concurrency = test_case_concurrency  # Max concurrent test case executions

    async def evaluate(
        self,
        *,
        runner: PromptRunner,
        prompt: PromptTemplate,
        model: str = "gpt-4",
        test_cases: Optional[list[PromptTestCase]] = None,
        variables: Optional[dict[str, Any]] = None,
    ) -> FitnessEvaluation:
        """Evaluate prompt using comprehensive accuracy and semantic analysis"""

        if not test_cases:
            output = await runner.run(
                model=model,
                prompt=prompt,
                variables=variables,
            )
            return await self._evaluate_prompt_quality(prompt, output.content or "")

        # Run test cases in parallel
        test_results = await self._run_test_cases_parallel(
            runner=runner, model=model, prompt=prompt, test_cases=test_cases
        )

        # Run comprehensive evaluation with both accuracy and semantic analysis
        evaluation_prompt = self._create_comprehensive_evaluation_prompt(prompt, test_results)

        try:
            response = await self.evaluation_client.generate(
                prompt=evaluation_prompt,
                model=self.evaluation_model,
                # temperature=0.3,  # Low temperature for consistent evaluation
                options=_get_llm_config_for_client(self.evaluation_client),
            )

            evaluation_response = EvaluationResponse.model_validate_json(response.content or "")

            # Extract score and reasoning from structured response
            score = evaluation_response.score
            reasoning = evaluation_response.reasoning

        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse evaluation response: {e}") from e

        return FitnessEvaluation(
            prompt=prompt,
            score=score,
            test_results=test_results,
            evaluation_reasoning=reasoning,
            metadata={"evaluation_method": "comprehensive_llm_powered"},
        )

    async def _run_test_cases_parallel(
        self,
        runner: PromptRunner,
        model: str,
        prompt: PromptTemplate,
        test_cases: list[PromptTestCase],
    ) -> list[dict[str, Any]]:
        """Run test cases in parallel with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.test_case_concurrency)

        async def run_single_test(test_case: PromptTestCase) -> dict[str, Any]:
            async with semaphore:
                try:
                    response = await runner.run(
                        model=model,
                        prompt=prompt,
                        variables=test_case.input_variables,
                    )
                    return {
                        "input": test_case.input_variables,
                        "expected": test_case.expected_output,
                        "actual": response.content,
                        "metadata": test_case.metadata,
                    }
                except Exception as e:
                    return {
                        "input": test_case.input_variables,
                        "expected": test_case.expected_output,
                        "actual": f"Error: {str(e)}",
                        "error": str(e),
                        "metadata": test_case.metadata,
                    }

        tasks = [run_single_test(test_case) for test_case in test_cases]
        results = await asyncio.gather(*tasks)
        return results

    def _create_comprehensive_evaluation_prompt(
        self, prompt: PromptTemplate, test_results: list[dict[str, Any]]
    ) -> str:
        """Create comprehensive evaluation prompt combining accuracy and semantic analysis"""
        return f"""
You are an expert LLM prompt evaluator specializing in comprehensive prompt assessment.
Your task is to evaluate the effectiveness of a prompt using both accuracy and semantic analysis.

COMPREHENSIVE EVALUATION FRAMEWORK:

1. ACCURACY ANALYSIS (Weight: 40%)
   - Exact Match: Calculate percentage of test cases with exact expected output matches
   - Functional Correctness: Assess if outputs achieve the intended purpose
   - Completeness: Evaluate if all required information is provided
   - Precision: Check for unnecessary or irrelevant information

2. SEMANTIC SIMILARITY (Weight: 30%)
   - Meaning Preservation: Do actual outputs convey the same meaning as expected?
   - Key Concept Retention: Are important concepts and information preserved?
   - Intent Alignment: Does the output serve the same purpose as expected?
   - Contextual Appropriateness: Is the response appropriate for the given context?

3. CONSISTENCY & ROBUSTNESS (Weight: 20%)
   - Output Stability: Evaluate consistency across similar inputs
   - Format Adherence: Check for consistent formatting and structure
   - Error Handling: Assess graceful handling of edge cases and errors
   - Predictability: Do outputs follow predictable patterns?

4. PROMPT QUALITY (Weight: 10%)
   - Clarity: Is the prompt clear and unambiguous?
   - Completeness: Does it provide sufficient context and instructions?
   - Structure: Is it well-organized and easy to follow?
   - Effectiveness: Would this prompt likely produce high-quality responses?

SCORING METHODOLOGY:
- Calculate individual scores for each criterion (0.0 to 1.0)
- Apply weights to compute final comprehensive score
- Consider both exact matches and semantic equivalence
- Use this scale:
  * 0.9-1.0: Excellent - Production ready with high accuracy and semantic fidelity
  * 0.7-0.89: Good - Minor improvements needed, mostly accurate and semantically correct
  * 0.5-0.69: Adequate - Notable issues with accuracy or semantic alignment
  * 0.3-0.49: Poor - Significant accuracy or semantic problems
  * 0.0-0.29: Critical - Major accuracy and semantic failures

EVALUATION INSTRUCTIONS:
- For each test case, assess both exact accuracy and semantic similarity
- Consider that semantically equivalent responses should score highly even if not exact matches
- Weight exact matches more heavily when precision is critical
- Weight semantic similarity more heavily when meaning preservation is key
- Provide detailed reasoning covering both accuracy and semantic aspects

Please provide:
1. A comprehensive score from 0.0 to 1.0 (where 1.0 is perfect)
2. Detailed reasoning covering accuracy, semantic similarity, and overall assessment

RESPONSE:
- ONLY VALID JSON matching the JSON SCHEMA
- NO OTHER TEXT

==========
JSON SCHEMA:
{simple_schema(EvaluationResponse)}
==========

==========
PROMPT TEMPLATE TO EVALUATE:
{prompt.template}
==========

==========
TEST RESULTS:
{json.dumps(test_results, indent=2)}
==========
"""

    async def _evaluate_prompt_quality(
        self, prompt: PromptTemplate, output: str
    ) -> FitnessEvaluation:
        """Evaluate prompt quality without test cases"""

        quality_prompt = self._create_quality_evaluation_prompt(prompt, output)

        try:
            response = await self.evaluation_client.generate(
                prompt=quality_prompt,
                model=self.evaluation_model,
                # temperature=0.3,
                options=_get_llm_config_for_client(self.evaluation_client),
            )

            evaluation_response = QualityEvaluationResponse.model_validate_json(
                response.content or ""
            )

            score = evaluation_response.score
            reasoning = evaluation_response.reasoning

        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError("Failed to parse quality evaluation response") from e

        return FitnessEvaluation(
            prompt=prompt,
            score=score,
            test_results=[],  # No test results for quality evaluation
            evaluation_reasoning=reasoning,
            metadata={"evaluation_method": "comprehensive_prompt_quality"},
        )

    def _create_quality_evaluation_prompt(self, prompt: PromptTemplate, output: str) -> str:
        """Create quality evaluation prompt for the LLM"""

        return f"""
You are an expert prompt engineer. Your task is to evaluate the quality of this prompt template using comprehensive criteria.

COMPREHENSIVE EVALUATION CRITERIA:
1. CLARITY & PRECISION (Weight: 25%)
   - Is the prompt clear and unambiguous?
   - Are instructions specific and actionable?
   - Is the language appropriate for the target model?

2. COMPLETENESS & CONTEXT (Weight: 25%)
   - Does it provide sufficient context and background?
   - Are all necessary instructions included?
   - Is the scope and purpose clearly defined?

3. STRUCTURE & ORGANIZATION (Weight: 20%)
   - Is it well-structured and easy to follow?
   - Are sections logically organized?
   - Is the flow natural and intuitive?

4. EFFECTIVENESS & USABILITY (Weight: 20%)
   - Would this prompt likely produce high-quality responses?
   - Is it optimized for the target use case?
   - Are examples or guidance provided when helpful?

5. TEMPLATE DESIGN (Weight: 10%)
   - Are template variables appropriately used?
   - Is the template flexible and reusable?
   - Are variable names clear and descriptive?

SCORING SCALE:
- 0.9-1.0: Excellent - Production ready with outstanding quality
- 0.7-0.89: Good - High quality with minor improvements needed
- 0.5-0.69: Adequate - Decent quality but notable issues to address
- 0.3-0.49: Poor - Significant quality problems requiring revision
- 0.0-0.29: Critical - Major quality issues requiring complete redesign

Please provide:
1. A comprehensive quality score from 0.0 to 1.0 (where 1.0 is excellent)
2. Detailed reasoning covering all evaluation criteria

==========
RESPONSE:
- ONLY VALID JSON matching the JSON SCHEMA
- NO OTHER TEXT

JSON SCHEMA:
{simple_schema(QualityEvaluationResponse)}
==========

==========
PROMPT TEMPLATE TO EVALUATE:
{prompt.template}
==========

==========
OUTPUT TO EVALUATE:
{output}
==========
"""


class LLMPromptMutator:
    """LLM-powered prompt mutation"""

    def __init__(self, mutation_client: BaseLLMClient, mutation_model: str):
        self.mutation_client = mutation_client
        self.mutation_model = mutation_model

    async def mutate(
        self, prompt: PromptTemplate, mutation_type: str = "random", mutation_strength: float = 0.5
    ) -> PromptTemplate:
        """Use LLM to intelligently mutate a prompt"""

        mutation_prompt = self._create_mutation_prompt(prompt, mutation_type, mutation_strength)

        try:
            response = await self.mutation_client.generate(
                prompt=mutation_prompt,
                model=self.mutation_model,
                # temperature=0.7 + mutation_strength * 0.3,  # Higher temp for more creativity
                options=_get_llm_config_for_client(self.mutation_client),
            )

            mutation_response = MutationResponse.model_validate_json(response.content or "")

            mutated_template = mutation_response.mutated_prompt

            return PromptTemplate(
                template=mutated_template, name=f"{prompt.name}_mutated", metadata=prompt.metadata
            )

        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse mutation response from LLM: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"LLM mutation failed: {str(e)}") from e

    def _create_mutation_prompt(
        self, prompt: PromptTemplate, mutation_type: str, strength: float
    ) -> str:
        """Create mutation instruction prompt"""

        mutation_instructions = {
            "random": "Make random improvements to this prompt while keeping the core functionality.",
            "improve_clarity": "Improve the clarity and precision of this prompt.",
            "add_examples": "Add helpful examples to make this prompt more effective.",
            "optimize_structure": "Optimize the structure and flow of this prompt.",
            "enhance_instructions": "Enhance the instructions to be more specific and actionable.",
            "reduce_ambiguity": "Reduce ambiguity and make the prompt more unambiguous.",
        }

        instruction = mutation_instructions.get(mutation_type, mutation_instructions["random"])

        return f"""
You are an expert prompt engineer. Your task is to improve the following prompt template.

ORIGINAL PROMPT:
{prompt.template}

MUTATION TYPE: {mutation_type}
MUTATION STRENGTH: {strength} (0.0 = subtle changes, 1.0 = major changes)

INSTRUCTION: {instruction}

IMPORTANT CONSTRAINTS:
1. Keep the same template variables ({{variable_name}})
2. Maintain the core purpose and functionality
3. Only improve the prompt, don't change its fundamental nature
4. The output should be a complete, usable prompt template

CRITICAL JSON FORMATTING REQUIREMENTS:
- The mutated_prompt MUST be a single valid JSON string value
- Properly escape ALL newlines as \\n (backslash-n) in the JSON
- Escape ALL double quotes as \\" in the JSON
- Escape ALL backslashes as \\\\ in the JSON
- Do NOT include literal line breaks within string values
- The entire response must be parseable as valid JSON
- Test your JSON mentally before responding - it must parse correctly

RESPONSE FORMAT:
- ONLY VALID JSON matching the JSON SCHEMA below
- NO markdown code blocks (no ```json)
- NO explanatory text before or after
- Ensure every string is properly escaped for JSON

JSON SCHEMA:
{simple_schema(MutationResponse)}
"""


class LLMPopulationGenerator:
    """LLM-powered initial population generator"""

    def __init__(self, generation_client: BaseLLMClient, generation_model: str = "gpt-4"):
        self.generation_client = generation_client
        self.generation_model = generation_model

    async def generate_initial_population(
        self, base_prompt: PromptTemplate, population_size: int, diversity_level: float = 0.7
    ) -> list[PromptTemplate]:
        """Generate diverse initial population using LLM"""

        generation_prompt = self._create_generation_prompt(
            base_prompt, population_size - 1, diversity_level
        )
        try:
            response = await self.generation_client.generate(
                prompt=generation_prompt,
                model=self.generation_model,
                # temperature=0.7 + diversity_level * 0.3,
                options=_get_llm_config_for_client(self.generation_client),
            )

            generation_response = PopulationGenerationResponse.model_validate_json(
                response.content or ""
            )
            variations = self._create_variations_from_structured(
                generation_response.variations, base_prompt
            )

            # Add the original prompt as the first member
            population = [base_prompt] + variations

            return population[:population_size]

        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError("Failed to parse population generation response from LLM") from e
        except Exception as e:
            raise ValueError("LLM population generation failed") from e

    def _create_generation_prompt(
        self, base_prompt: PromptTemplate, num_variations: int, diversity_level: float
    ) -> str:
        """Create prompt for LLM population generation"""
        return f"""
You are an expert prompt engineer. Create {num_variations} diverse variations of the following prompt template.

DIVERSITY LEVEL: {diversity_level} (0.0 = subtle variations, 1.0 = very diverse)

Create variations that explore different approaches:
1. Different structural patterns (question format, instruction format, conversational, etc.)
2. Various instruction styles (direct, polite, detailed, concise)
3. Different levels of detail and specificity
4. Alternative phrasings and word choices
5. Different emphasis on clarity vs. brevity
6. Various ways to handle the same core task
7. Different prompt engineering techniques (few-shot, chain-of-thought, etc.)

CONSTRAINTS:
- Keep the same template variajbles ({{variable_name}}) exactly as they appear
- Maintain the core purpose and functionality of the original prompt
- Each variation should be complete and usable
- Make each variation distinct and valuable
- Ensure all variations can handle the same inputs and produce similar outputs

RESPONSE FORMAT:
- ONLY VALID JSON matching the example response below
- Test your JSON mentally before responding - it must parse correctly
- The entire response must be parseable as valid JSON

JSON SCHEMA:
{simple_schema(PopulationGenerationResponse)}

ORIGINAL PROMPT:
{base_prompt.template}
"""

    def _create_variations_from_structured(
        self, variations: list[str], base_prompt: PromptTemplate
    ) -> list[PromptTemplate]:
        """Create PromptTemplate objects from structured response variations"""
        prompt_templates = []

        for i, variation in enumerate(variations):
            if variation.strip():  # Only process non-empty variations
                prompt_templates.append(
                    PromptTemplate(
                        template=variation.strip(),
                        name=f"{base_prompt.name}_llm_var_{i + 1}",
                        metadata=base_prompt.metadata,
                    )
                )

        return prompt_templates


class LLMPromptCrossover:
    """LLM-powered prompt crossover"""

    def __init__(self, crossover_client: BaseLLMClient, crossover_model: str):
        self.crossover_client = crossover_client
        self.crossover_model = crossover_model

    async def crossover(
        self, parent1: PromptTemplate, parent2: PromptTemplate
    ) -> tuple[PromptTemplate, PromptTemplate]:
        """Use LLM to intelligently combine two prompts"""

        crossover_prompt = self._create_crossover_prompt(parent1, parent2)
        try:
            response = await self.crossover_client.generate(
                prompt=crossover_prompt,
                model=self.crossover_model,
                # temperature=0.6,
                options=_get_llm_config_for_client(self.crossover_client),
            )

            crossover_response = CrossoverResponse.model_validate_json(response.content or "")

            offspring1 = crossover_response.offspring1
            offspring2 = crossover_response.offspring2

            return (
                PromptTemplate(
                    template=offspring1,
                    name=f"{parent1.name}_offspring1",
                    metadata=parent1.metadata,
                ),
                PromptTemplate(
                    template=offspring2,
                    name=f"{parent2.name}_offspring2",
                    metadata=parent2.metadata,
                ),
            )
        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse crossover response from LLM: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"LLM crossover failed: {str(e)}") from e

    def _create_crossover_prompt(self, parent1: PromptTemplate, parent2: PromptTemplate) -> str:
        """Create crossover instruction prompt"""
        return f"""
You are an expert prompt engineer. Your task is to create two new prompt templates by intelligently combining the best elements from two parent prompts.

PARENT PROMPT 1:
{parent1.template}

PARENT PROMPT 2:
{parent2.template}

INSTRUCTIONS:
1. Create two distinct offspring prompts that combine the strengths of both parents
2. Each offspring should be a complete, functional prompt template
3. Preserve all template variables from both parents
4. Make each offspring unique and innovative
5. Ensure both offspring maintain the core functionality

CRITICAL JSON FORMATTING REQUIREMENTS:
- Both offspring1 and offspring2 MUST be valid JSON string values
- Properly escape ALL newlines as \\n (backslash-n) in the JSON
- Escape ALL double quotes as \\" in the JSON
- Escape ALL backslashes as \\\\ in the JSON
- Do NOT include literal line breaks within string values
- The entire response must be parseable as valid JSON
- Test your JSON mentally before responding - it must parse correctly

RESPONSE FORMAT:
- ONLY VALID JSON matching the JSON SCHEMA below
- NO markdown code blocks (no ```json)
- NO explanatory text before or after
- Ensure every string is properly escaped for JSON

JSON SCHEMA:
{simple_schema(CrossoverResponse)}
"""


class OptimizerPromptRunner(PromptRunner):
    """PromptRunner with optimizer context tracking"""

    def __init__(
        self,
        client: BaseLLMClient,
        tracer: Optional[Tracer] = None,
        backup_client: Optional[BaseLLMClient] = None,
        optimizer_context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(client, tracer, backup_client)
        self.optimizer_context = optimizer_context or {}

    async def _run_with_client(
        self,
        client: BaseLLMClient,
        model: str,
        prompt: PromptTemplate,
        variables: Optional[dict[str, Any]] = None,
        **llm_kwargs: Any,
    ) -> LLMResponse:
        """Override to add optimizer context to traces"""
        if variables is None:
            variables = {}
        response = await super()._run_with_client(client, model, prompt, variables, **llm_kwargs)

        # Add optimizer context to the response metadata if we have a tracer and context
        if self.tracer and self.optimizer_context and response.metadata.get("trace_id"):
            # Update the trace record with optimizer context
            trace_record = self.tracer.get_record(str(response.metadata["trace_id"]))
            if trace_record:
                # Merge optimizer context into metadata
                updated_metadata = trace_record.metadata.copy()
                updated_metadata["optimizer_context"] = self.optimizer_context

                # Update the trace record in the database
                with sqlite3.connect(self.tracer.db_path) as conn:
                    conn.execute(
                        "UPDATE traces SET metadata = ? WHERE id = ?",
                        (json.dumps(updated_metadata, default=str), trace_record.id),
                    )
                    conn.commit()

        return response


class LLMGeneticOptimizer:
    """LLM-powered genetic algorithm for prompt optimization"""

    def __init__(
        self,
        eval_model: str,
        population_size: int = 20,
        generations: int = 10,
        fitness_function: Optional[LLMFitnessFunction] = None,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        eval_client: Optional[BaseLLMClient] = None,
        tracer: Optional[Tracer] = None,
        population_diversity_level: float = 0.7,
        progress_callback: Optional[ProgressCallback] = None,
        max_concurrent_evaluations: int = 10,
        test_case_concurrency: int = 10,
    ):
        self.eval_client = eval_client or OpenAIClient()

        self.population_size = population_size
        self.generations = generations
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.test_case_concurrency = test_case_concurrency
        self.fitness_function = fitness_function or LLMComprehensiveFitnessFunction(
            self.eval_client, eval_model, test_case_concurrency=test_case_concurrency
        )
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.elite_size = max(1, int(population_size * elite_ratio))
        self.tracer = tracer or Tracer()
        self.progress_callback = progress_callback or NoOpProgressCallback()

        # Population generation settings
        self.population_diversity_level = population_diversity_level

        # Initialize mutators, crossovers, and population generator (all LLM-driven)
        self.mutator = LLMPromptMutator(self.eval_client, eval_model)
        self.crossover = LLMPromptCrossover(self.eval_client, eval_model)
        self.population_generator = LLMPopulationGenerator(self.eval_client, eval_model)

        # Internal state
        self.current_generation = 0
        self.population: list[PromptTemplate] = []
        self.fitness_scores: list[float] = []
        self._current_generation_stats: Optional[dict[str, Any]] = None
        self.optimization_id: str = str(uuid.uuid4())

    async def optimize(
        self,
        *,
        runner: PromptRunner,
        base_prompt: PromptTemplate,
        model: str,
        test_cases: Optional[list[PromptTestCase]] = None,
        variables: Optional[dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Run the LLM-powered genetic optimization process"""

        start_time = datetime.now()
        total_evaluations = 0

        # Create optimizer-aware runner if tracing is enabled
        if self.tracer and self.tracer.is_tracing_enabled and runner is not None:
            # Create context for this optimization run
            optimizer_context = {
                "optimization_id": self.optimization_id,
                "population_size": self.population_size,
                "generations": self.generations,
                "base_prompt_name": base_prompt.name,
                "start_time": start_time.isoformat(),
            }

            # Wrap the runner with optimizer context
            runner = OptimizerPromptRunner(
                client=runner.client,
                tracer=runner.tracer,
                backup_client=runner.backup_client,
                optimizer_context=optimizer_context,
            )

        # Initialize population
        await self._initialize_population(base_prompt)
        await self.progress_callback.on_population_initialized(len(self.population))

        # Evolution loop
        for generation in range(self.generations):
            self.current_generation = generation
            await self.progress_callback.on_generation_start(generation, self.generations)

            # Update optimizer context with current generation if we have an optimizer runner
            if isinstance(runner, OptimizerPromptRunner):
                runner.optimizer_context["generation"] = generation

            # Evaluate fitness for all individuals using LLM
            evaluations = await self._evaluate_population(
                test_cases=test_cases,
                runner=runner,
                model=model,
                variables=variables,
            )
            total_evaluations += len(evaluations)

            errors = [eval for eval in evaluations if isinstance(eval, BaseException)]
            if errors:
                print(f"Failed to evaluate {len(errors)} individuals: {errors}")

            valid_evaluations: list[FitnessEvaluation] = [
                eval for eval in evaluations if isinstance(eval, FitnessEvaluation)
            ]

            # Check if we got any valid evaluations
            if not valid_evaluations:
                raise ValueError(
                    "All fitness evaluations failed. No valid individuals in population."
                )

            # Find best individual
            best_idx = max(range(len(valid_evaluations)), key=lambda i: valid_evaluations[i].score)
            best_evaluation = valid_evaluations[best_idx]

            # Log progress
            await self._log_generation_progress(generation, best_evaluation, errors)
            if self._current_generation_stats is not None:
                await self.progress_callback.on_generation_complete(self._current_generation_stats)

            # Create next generation (except for last generation)
            if generation < self.generations - 1:
                await self._create_next_generation_llm(valid_evaluations)

        # Calculate final results
        optimization_time = (datetime.now() - start_time).total_seconds()
        best_prompt = best_evaluation.prompt

        result = OptimizationResult(
            best_prompt=best_prompt,
            fitness_score=best_evaluation.score,
            generation=self.current_generation,
            population_size=self.population_size,
            total_evaluations=total_evaluations,
            optimization_time=optimization_time,
            metadata={
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elite_ratio": self.elite_ratio,
                "elite_size": self.elite_size,
                "llm_powered": True,
                "population_diversity_level": self.population_diversity_level,
            },
        )

        await self.progress_callback.on_optimization_complete(result)
        return result

    async def _initialize_population(self, base_prompt: PromptTemplate) -> None:
        """Initialize population with LLM-generated variations of the base prompt"""
        self.population = await self.population_generator.generate_initial_population(
            base_prompt, self.population_size, self.population_diversity_level
        )

    async def _evaluate_population(
        self,
        *,
        test_cases: Optional[list[PromptTestCase]],
        runner: PromptRunner,
        variables: Optional[dict[str, Any]] = None,
        model: str,
    ) -> list[Union[FitnessEvaluation, BaseException]]:
        """Evaluate fitness for entire population"""
        if not self.fitness_function:
            raise ValueError("No fitness function provided")

        # Run evaluations in parallel with configurable concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)

        async def evaluate_single(prompt: PromptTemplate) -> FitnessEvaluation:
            async with semaphore:
                return await self.fitness_function.evaluate(
                    runner=runner,
                    prompt=prompt,
                    test_cases=test_cases,
                    variables=variables,
                    model=model,
                )

        tasks = [evaluate_single(prompt) for prompt in self.population]
        evaluations = await asyncio.gather(*tasks, return_exceptions=True)

        return evaluations

    def _tournament_selection(
        self, evaluations: list[FitnessEvaluation], tournament_size: int = 3
    ) -> PromptTemplate:
        """Select a parent using tournament selection"""
        tournament = random.sample(evaluations, min(tournament_size, len(evaluations)))
        winner = max(tournament, key=lambda x: x.score)
        return winner.prompt

    async def _create_next_generation_llm(self, evaluations: list[FitnessEvaluation]) -> None:
        """Create next generation using parallelized LLM-powered operations"""
        new_population = []

        # Sort by fitness (descending)
        sorted_evaluations = sorted(evaluations, key=lambda x: x.score, reverse=True)

        # Elitism: keep best individuals
        for i in range(min(self.elite_size, len(sorted_evaluations))):
            new_population.append(sorted_evaluations[i].prompt)

        # Calculate how many offspring we need
        offspring_needed = self.population_size - len(new_population)

        # Pre-select all parents and determine operations
        operations = []
        for _ in range(offspring_needed):
            parent1 = self._tournament_selection(sorted_evaluations)
            parent2 = self._tournament_selection(sorted_evaluations)

            # Determine if crossover and mutation will be applied
            apply_crossover = random.random() < self.crossover_rate
            apply_mutation = random.random() < self.mutation_rate

            operations.append(
                {
                    "parent1": parent1,
                    "parent2": parent2,
                    "apply_crossover": apply_crossover,
                    "apply_mutation": apply_mutation,
                    "mutation_type": random.choice(
                        ["random", "improve_clarity", "add_examples", "optimize_structure"]
                    ),
                    "mutation_strength": random.uniform(0.3, 0.8),
                }
            )

        # Execute all operations in parallel with controlled concurrency
        offspring = await self._execute_generation_operations_parallel(operations)

        # Add offspring to new population
        new_population.extend(offspring)

        # Trim to exact population size
        self.population = new_population[: self.population_size]

    async def _execute_generation_operations_parallel(
        self, operations: list[dict[str, Any]]
    ) -> list[PromptTemplate]:
        """Execute all crossover and mutation operations in parallel"""

        # Semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)

        async def process_operation(op: dict[str, Any]) -> PromptTemplate:
            async with semaphore:
                parent1 = op["parent1"]
                parent2 = op["parent2"]

                # Crossover
                if op["apply_crossover"] and self.crossover:
                    try:
                        offspring1, offspring2 = await self.crossover.crossover(parent1, parent2)
                        # Use first offspring by default
                        result = offspring1
                    except Exception:
                        # On crossover failure, fall back to parent
                        result = parent1
                else:
                    # Just use parent1
                    result = parent1

                # Mutation
                if op["apply_mutation"] and self.mutator:
                    try:
                        result = await self.mutator.mutate(
                            result,
                            mutation_type=op["mutation_type"],
                            mutation_strength=op["mutation_strength"],
                        )
                    except Exception:
                        # On mutation failure, keep the unmutated result
                        pass

                return result

        tasks = [process_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid prompts
        valid_results = []
        for result in results:
            if isinstance(result, PromptTemplate):
                valid_results.append(result)
            # Silently skip failed operations - they're logged by the error handlers above

        return valid_results

    async def _log_generation_progress(
        self, generation: int, best_evaluation: FitnessEvaluation, errors: list[BaseException]
    ) -> None:
        """Log progress of optimization"""
        # Calculate metrics for internal tracking
        avg_fitness = (
            sum(self.fitness_scores) / len(self.fitness_scores) if self.fitness_scores else 0
        )
        max_fitness = max(self.fitness_scores) if self.fitness_scores else 0

        self._current_generation_stats = {
            "generation": generation + 1,
            "best_fitness": best_evaluation.score,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "best_prompt": best_evaluation.prompt.template,
            "reasoning": best_evaluation.evaluation_reasoning,
            "errors": errors,
        }


def _get_llm_config_for_client(client: BaseLLMClient) -> dict[str, Any]:
    """
    Get the appropriate LLM configuration based on client type.
    """
    if isinstance(client, GoogleAIClient):
        return {"max_output_tokens": 10_000, "response_mime_type": "application/json"}
    else:
        # Default config for OpenAI and Anthropic clients
        return {"max_completion_tokens": 10_000, "response_format": {"type": "json_object"}}
