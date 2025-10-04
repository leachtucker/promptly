"""
LLM-powered prompt optimization module
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
import random
import asyncio
import json
from datetime import datetime

from .templates import PromptTemplate
from .clients import BaseLLMClient, LLMResponse, OpenAIClient
from .runner import PromptRunner
from .tracer import Tracer, TraceRecord


@dataclass
class OptimizationResult:
    """Result of an optimization run"""
    best_prompt: PromptTemplate
    fitness_score: float
    generation: int
    population_size: int
    total_evaluations: int
    optimization_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    """A test case for prompt evaluation"""
    input_variables: Dict[str, Any]
    expected_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitnessEvaluation:
    """Result of a fitness evaluation"""
    prompt: PromptTemplate
    score: float
    test_results: List[Dict[str, Any]]
    evaluation_reasoning: str  # LLM's reasoning for the score
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMFitnessFunction(ABC):
    """Abstract base class for LLM-powered fitness functions"""
    
    def __init__(self, evaluation_client: BaseLLMClient, evaluation_model: str = "gpt-4"):
        self.evaluation_client = evaluation_client
        self.evaluation_model = evaluation_model
    
    @abstractmethod
    async def evaluate(
        self, 
        prompt: PromptTemplate, 
        test_cases: Optional[List[TestCase]] = None,
        runner: Optional[PromptRunner] = None,
        **kwargs: Any
    ) -> FitnessEvaluation:
        """Evaluate fitness of a prompt using LLM"""
        pass


class LLMAccuracyFitnessFunction(LLMFitnessFunction):
    """LLM-powered fitness function for accuracy evaluation"""
    
    async def evaluate(
        self,
        prompt: PromptTemplate,
        test_cases: Optional[List[TestCase]] = None,
        runner: Optional[PromptRunner] = None,
        model: str = "gpt-3.5-turbo",
        **kwargs: Any
    ) -> FitnessEvaluation:
        """Evaluate prompt accuracy using LLM reasoning"""
        
        if test_cases is None or len(test_cases) == 0:
            # Evaluate prompt quality without test cases
            return await self._evaluate_prompt_quality(prompt)
        
        if runner is None:
            raise ValueError("Runner is required when test cases are provided")
        
        # First, run the prompt on all test cases
        test_results = []
        for test_case in test_cases:
            try:
                response = await runner.run(
                    model=model,
                    prompt=prompt,
                    variables=test_case.input_variables,
                    **kwargs
                )
                test_results.append({
                    "input": test_case.input_variables,
                    "expected": test_case.expected_output,
                    "actual": response.content,
                    "metadata": test_case.metadata
                })
            except Exception as e:
                test_results.append({
                    "input": test_case.input_variables,
                    "expected": test_case.expected_output,
                    "actual": f"Error: {str(e)}",
                    "error": str(e),
                    "metadata": test_case.metadata
                })
        
        # Use LLM to evaluate the results
        evaluation_prompt = self._create_evaluation_prompt(prompt, test_results)
        
        try:
            evaluation_response = await self.evaluation_client.generate(
                prompt=evaluation_prompt,
                model=self.evaluation_model,
                temperature=0.3,  # Low temperature for consistent evaluation
                max_tokens=500
            )
            
            # Parse LLM response for score and reasoning
            score, reasoning = self._parse_evaluation_response(evaluation_response.content)
            
        except Exception as e:
            # Fallback to simple accuracy if LLM evaluation fails
            score = self._calculate_simple_accuracy(test_results)
            reasoning = f"LLM evaluation failed: {str(e)}. Using simple accuracy."
        
        return FitnessEvaluation(
            prompt=prompt,
            score=score,
            test_results=test_results,
            evaluation_reasoning=reasoning,
            metadata={"evaluation_method": "llm_powered"}
        )
    
    def _create_evaluation_prompt(self, prompt: PromptTemplate, test_results: List[Dict[str, Any]]) -> str:
        """Create evaluation prompt for the LLM"""
        return f"""
You are an expert prompt evaluator. Your task is to evaluate how well a prompt template performs on test cases.

PROMPT TEMPLATE TO EVALUATE:
{prompt.template}

TEST RESULTS:
{json.dumps(test_results, indent=2)}

EVALUATION CRITERIA:
1. Accuracy: How often does the prompt produce the expected output?
2. Consistency: Are the responses consistent in format and quality?
3. Robustness: How well does it handle edge cases?
4. Clarity: Are the responses clear and well-structured?

Please provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. Detailed reasoning for your score

Format your response as:
SCORE: [number between 0.0 and 1.0]
REASONING: [detailed explanation of your evaluation]
"""
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, str]:
        """Parse LLM evaluation response"""
        try:
            lines = response.strip().split('\n')
            score_line = None
            reasoning_lines = []
            
            for i, line in enumerate(lines):
                if line.startswith('SCORE:'):
                    score_line = line
                    reasoning_lines = lines[i+1:]
                    break
            
            if score_line:
                score_str = score_line.split('SCORE:')[1].strip()
                score = float(score_str)
                reasoning = '\n'.join(reasoning_lines).replace('REASONING:', '').strip()
                return score, reasoning
            else:
                # Fallback parsing
                return 0.5, "Could not parse LLM evaluation response"
                
        except Exception:
            return 0.5, "Error parsing evaluation response"
    
    def _calculate_simple_accuracy(self, test_results: List[Dict[str, Any]]) -> float:
        """Fallback simple accuracy calculation"""
        if not test_results:
            return 0.0
        
        correct = sum(1 for result in test_results 
                     if result.get('actual', '').strip().lower() == 
                        result.get('expected', '').strip().lower())
        return correct / len(test_results)
    
    async def _evaluate_prompt_quality(self, prompt: PromptTemplate) -> FitnessEvaluation:
        """Evaluate prompt quality without test cases"""
        
        quality_prompt = self._create_quality_evaluation_prompt(prompt)
        
        try:
            evaluation_response = await self.evaluation_client.generate(
                prompt=quality_prompt,
                model=self.evaluation_model,
                temperature=0.3,
                max_tokens=500
            )
            
            score, reasoning = self._parse_evaluation_response(evaluation_response.content)
            
        except Exception as e:
            score = 0.5
            reasoning = f"Quality evaluation failed: {str(e)}"
        
        return FitnessEvaluation(
            prompt=prompt,
            score=score,
            test_results=[],  # No test results for quality evaluation
            evaluation_reasoning=reasoning,
            metadata={"evaluation_method": "prompt_quality"}
        )
    
    def _create_quality_evaluation_prompt(self, prompt: PromptTemplate) -> str:
        """Create quality evaluation prompt for the LLM"""
        return f"""
You are an expert prompt engineer. Your task is to evaluate the quality of this prompt template.

PROMPT TEMPLATE TO EVALUATE:
{prompt.template}

EVALUATION CRITERIA:
1. Clarity: Is the prompt clear and unambiguous?
2. Completeness: Does it provide sufficient context and instructions?
3. Specificity: Is it specific enough to guide good responses?
4. Structure: Is it well-structured and easy to follow?
5. Effectiveness: Would this prompt likely produce high-quality responses?
6. Template Variables: Are template variables appropriately used?

Please provide:
1. A score from 0.0 to 1.0 (where 1.0 is excellent)
2. Detailed reasoning for your score

Format your response as:
SCORE: [number between 0.0 and 1.0]
REASONING: [detailed explanation of your evaluation]
"""


class LLMSemanticFitnessFunction(LLMFitnessFunction):
    """LLM-powered fitness function using semantic similarity"""
    
    async def evaluate(
        self,
        prompt: PromptTemplate,
        test_cases: Optional[List[TestCase]] = None,
        runner: Optional[PromptRunner] = None,
        model: str = "gpt-3.5-turbo",
        **kwargs: Any
    ) -> FitnessEvaluation:
        """Evaluate using semantic similarity via LLM"""
        
        if test_cases is None or len(test_cases) == 0:
            # Evaluate prompt quality without test cases
            return await self._evaluate_prompt_quality(prompt)
        
        if runner is None:
            raise ValueError("Runner is required when test cases are provided")
        
        test_results = []
        for test_case in test_cases:
            try:
                response = await runner.run(
                    model=model,
                    prompt=prompt,
                    variables=test_case.input_variables,
                    **kwargs
                )
                test_results.append({
                    "input": test_case.input_variables,
                    "expected": test_case.expected_output,
                    "actual": response.content,
                    "metadata": test_case.metadata
                })
            except Exception as e:
                test_results.append({
                    "input": test_case.input_variables,
                    "expected": test_case.expected_output,
                    "actual": f"Error: {str(e)}",
                    "error": str(e),
                    "metadata": test_case.metadata
                })
        
        # Use LLM to evaluate semantic similarity
        similarity_prompt = self._create_similarity_prompt(test_results)
        
        try:
            similarity_response = await self.evaluation_client.generate(
                prompt=similarity_prompt,
                model=self.evaluation_model,
                temperature=0.3,
                max_tokens=500
            )
            
            score, reasoning = self._parse_similarity_response(similarity_response.content)
            
        except Exception as e:
            score = 0.5
            reasoning = f"Semantic evaluation failed: {str(e)}"
        
        return FitnessEvaluation(
            prompt=prompt,
            score=score,
            test_results=test_results,
            evaluation_reasoning=reasoning,
            metadata={"evaluation_method": "llm_semantic"}
        )
    
    def _create_similarity_prompt(self, test_results: List[Dict[str, Any]]) -> str:
        """Create semantic similarity evaluation prompt"""
        return f"""
You are an expert at evaluating semantic similarity between texts. Evaluate how semantically similar the actual outputs are to the expected outputs.

TEST RESULTS:
{json.dumps(test_results, indent=2)}

For each test case, consider:
1. Do the actual and expected outputs convey the same meaning?
2. Are key concepts and information preserved?
3. Is the intent and purpose the same?

Provide an overall similarity score from 0.0 to 1.0 and reasoning.

Format:
SCORE: [number between 0.0 and 1.0]
REASONING: [detailed explanation]
"""
    
    def _parse_similarity_response(self, response: str) -> Tuple[float, str]:
        """Parse similarity evaluation response"""
        # Similar parsing logic to accuracy function
        try:
            lines = response.strip().split('\n')
            score_line = None
            reasoning_lines = []
            
            for i, line in enumerate(lines):
                if line.startswith('SCORE:'):
                    score_line = line
                    reasoning_lines = lines[i+1:]
                    break
            
            if score_line:
                score_str = score_line.split('SCORE:')[1].strip()
                score = float(score_str)
                reasoning = '\n'.join(reasoning_lines).replace('REASONING:', '').strip()
                return score, reasoning
            else:
                # Fallback parsing
                return 0.5, "Could not parse LLM evaluation response"
                
        except Exception:
            return 0.5, "Error parsing evaluation response"
    
    async def _evaluate_prompt_quality(self, prompt: PromptTemplate) -> FitnessEvaluation:
        """Evaluate prompt quality without test cases"""
        
        quality_prompt = self._create_quality_evaluation_prompt(prompt)
        
        try:
            evaluation_response = await self.evaluation_client.generate(
                prompt=quality_prompt,
                model=self.evaluation_model,
                temperature=0.3,
                max_tokens=500
            )
            
            score, reasoning = self._parse_similarity_response(evaluation_response.content)
            
        except Exception as e:
            score = 0.5
            reasoning = f"Quality evaluation failed: {str(e)}"
        
        return FitnessEvaluation(
            prompt=prompt,
            score=score,
            test_results=[],  # No test results for quality evaluation
            evaluation_reasoning=reasoning,
            metadata={"evaluation_method": "prompt_quality"}
        )
    
    def _create_quality_evaluation_prompt(self, prompt: PromptTemplate) -> str:
        """Create quality evaluation prompt for the LLM"""
        return f"""
You are an expert prompt engineer. Your task is to evaluate the quality of this prompt template.

PROMPT TEMPLATE TO EVALUATE:
{prompt.template}

EVALUATION CRITERIA:
1. Clarity: Is the prompt clear and unambiguous?
2. Completeness: Does it provide sufficient context and instructions?
3. Specificity: Is it specific enough to guide good responses?
4. Structure: Is it well-structured and easy to follow?
5. Effectiveness: Would this prompt likely produce high-quality responses?
6. Template Variables: Are template variables appropriately used?

Please provide:
1. A score from 0.0 to 1.0 (where 1.0 is excellent)
2. Detailed reasoning for your score

Format your response as:
SCORE: [number between 0.0 and 1.0]
REASONING: [detailed explanation of your evaluation]
"""


class LLMPromptMutator:
    """LLM-powered prompt mutation"""
    
    def __init__(self, mutation_client: BaseLLMClient, mutation_model: str = "gpt-4"):
        self.mutation_client = mutation_client
        self.mutation_model = mutation_model
    
    async def mutate(
        self, 
        prompt: PromptTemplate, 
        mutation_type: str = "random",
        mutation_strength: float = 0.5
    ) -> PromptTemplate:
        """Use LLM to intelligently mutate a prompt"""
        
        mutation_prompt = self._create_mutation_prompt(prompt, mutation_type, mutation_strength)
        
        try:
            mutation_response = await self.mutation_client.generate(
                prompt=mutation_prompt,
                model=self.mutation_model,
                temperature=0.7 + mutation_strength * 0.3,  # Higher temp for more creativity
                max_tokens=1000
            )
            
            mutated_template = self._extract_mutated_prompt(mutation_response.content)
            
            return PromptTemplate(
                template=mutated_template,
                name=f"{prompt.name}_mutated",
                metadata=prompt.metadata
            )
            
        except Exception as e:
            # Fallback to simple mutation
            return self._simple_mutation(prompt)
    
    def _create_mutation_prompt(self, prompt: PromptTemplate, mutation_type: str, strength: float) -> str:
        """Create mutation instruction prompt"""
        
        mutation_instructions = {
            "random": "Make random improvements to this prompt while keeping the core functionality.",
            "improve_clarity": "Improve the clarity and precision of this prompt.",
            "add_examples": "Add helpful examples to make this prompt more effective.",
            "optimize_structure": "Optimize the structure and flow of this prompt.",
            "enhance_instructions": "Enhance the instructions to be more specific and actionable.",
            "reduce_ambiguity": "Reduce ambiguity and make the prompt more unambiguous."
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

Provide only the improved prompt template, no explanations or additional text.
"""
    
    def _extract_mutated_prompt(self, response: str) -> str:
        """Extract the mutated prompt from LLM response"""
        # Clean up the response to extract just the prompt
        lines = response.strip().split('\n')
        
        # Remove common prefixes/suffixes
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Here') and not line.startswith('The'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _simple_mutation(self, prompt: PromptTemplate) -> PromptTemplate:
        """Fallback simple mutation"""
        template = prompt.template
        
        # Simple text-based mutations as fallback
        mutations = [
            f"{template}\n\nBe precise and clear in your response.",
            f"Please {template.lower()}",
            f"{template}\n\nProvide specific details.",
            f"Task: {template}",
        ]
        
        mutated_template = random.choice(mutations)
        
        return PromptTemplate(
            template=mutated_template,
            name=f"{prompt.name}_simple_mutated",
            metadata=prompt.metadata
        )


class LLMPromptCrossover:
    """LLM-powered prompt crossover"""
    
    def __init__(self, crossover_client: BaseLLMClient, crossover_model: str = "gpt-4"):
        self.crossover_client = crossover_client
        self.crossover_model = crossover_model
    
    async def crossover(
        self, 
        parent1: PromptTemplate, 
        parent2: PromptTemplate
    ) -> Tuple[PromptTemplate, PromptTemplate]:
        """Use LLM to intelligently combine two prompts"""
        
        crossover_prompt = self._create_crossover_prompt(parent1, parent2)
        
        try:
            crossover_response = await self.crossover_client.generate(
                prompt=crossover_prompt,
                model=self.crossover_model,
                temperature=0.6,
                max_tokens=1500
            )
            
            offspring1, offspring2 = self._extract_offspring(crossover_response.content)
            
            return (
                PromptTemplate(
                    template=offspring1,
                    name=f"{parent1.name}_offspring1",
                    metadata=parent1.metadata
                ),
                PromptTemplate(
                    template=offspring2,
                    name=f"{parent2.name}_offspring2",
                    metadata=parent2.metadata
                )
            )
            
        except Exception as e:
            # Fallback to simple crossover
            return self._simple_crossover(parent1, parent2)
    
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

Provide the two offspring prompts in this exact format:
OFFSPRING 1:
[first prompt template]

OFFSPRING 2:
[second prompt template]
"""
    
    def _extract_offspring(self, response: str) -> Tuple[str, str]:
        """Extract offspring prompts from LLM response"""
        try:
            parts = response.split('OFFSPRING 1:')
            if len(parts) < 2:
                raise ValueError("Invalid format")
            
            offspring_section = parts[1]
            offspring_parts = offspring_section.split('OFFSPRING 2:')
            
            if len(offspring_parts) < 2:
                raise ValueError("Invalid format")
            
            offspring1 = offspring_parts[0].strip()
            offspring2 = offspring_parts[1].strip()
            
            return offspring1, offspring2
            
        except Exception:
            # Fallback parsing
            lines = response.strip().split('\n')
            mid_point = len(lines) // 2
            offspring1 = '\n'.join(lines[:mid_point])
            offspring2 = '\n'.join(lines[mid_point:])
            return offspring1, offspring2
    
    def _simple_crossover(self, parent1: PromptTemplate, parent2: PromptTemplate) -> Tuple[PromptTemplate, PromptTemplate]:
        """Fallback simple crossover"""
        # Simple concatenation-based crossover
        offspring1_template = f"{parent1.template}\n\n{parent2.template}"
        offspring2_template = f"{parent2.template}\n\n{parent1.template}"
        
        return (
            PromptTemplate(template=offspring1_template, name=f"{parent1.name}_simple_offspring1", metadata=parent1.metadata),
            PromptTemplate(template=offspring2_template, name=f"{parent2.name}_simple_offspring2", metadata=parent2.metadata)
        )


class LLMGeneticOptimizer:
    """LLM-powered genetic algorithm for prompt optimization"""
    
    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        fitness_function: Optional[LLMFitnessFunction] = None,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        mutation_client: Optional[BaseLLMClient] = None,
        crossover_client: Optional[BaseLLMClient] = None,
        eval_client: Optional[BaseLLMClient] = None,
        tracer: Optional[Tracer] = None,
        **kwargs: Any
    ):
        self.eval_client = eval_client or OpenAIClient()

        self.population_size = population_size
        self.generations = generations
        self.fitness_function = fitness_function or LLMAccuracyFitnessFunction(self.eval_client)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tracer = tracer or Tracer()
        
        # LLM clients for mutation and crossover
        self.mutation_client = mutation_client
        self.crossover_client = crossover_client
        
        # Initialize mutators and crossovers
        if self.mutation_client:
            self.mutator = LLMPromptMutator(self.mutation_client)
        else:
            self.mutator = None
            
        if self.crossover_client:
            self.crossover = LLMPromptCrossover(self.crossover_client)
        else:
            self.crossover = None
        
        # Internal state
        self.current_generation = 0
        self.population: List[PromptTemplate] = []
        self.fitness_scores: List[float] = []
        
    async def optimize(
        self,
        base_prompt: PromptTemplate,
        test_cases: Optional[List[TestCase]] = None,
        runner: Optional[PromptRunner] = None,
        **kwargs: Any
    ) -> OptimizationResult:
        """Run the LLM-powered genetic optimization process"""
        
        start_time = datetime.now()
        total_evaluations = 0
        
        # Initialize population
        self._initialize_population(base_prompt)
        
        # Evolution loop
        for generation in range(self.generations):
            self.current_generation = generation
            
            # Evaluate fitness for all individuals using LLM
            evaluations = await self._evaluate_population(
                test_cases, runner, **kwargs
            )
            total_evaluations += len(evaluations)
            
            # Update fitness scores
            self.fitness_scores = [eval.score for eval in evaluations]
            
            # Find best individual
            best_idx = max(range(len(evaluations)), key=lambda i: evaluations[i].score)
            best_evaluation = evaluations[best_idx]
            
            # Log progress
            await self._log_generation_progress(generation, best_evaluation)
            
            # Create next generation (except for last generation)
            if generation < self.generations - 1:
                await self._create_next_generation_llm(evaluations)
        
        # Calculate final results
        optimization_time = (datetime.now() - start_time).total_seconds()
        best_prompt = best_evaluation.prompt
        
        return OptimizationResult(
            best_prompt=best_prompt,
            fitness_score=best_evaluation.score,
            generation=self.current_generation,
            population_size=self.population_size,
            total_evaluations=total_evaluations,
            optimization_time=optimization_time,
            metadata={
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elite_size": self.elite_size,
                "llm_powered": True
            }
        )
    
    def _initialize_population(self, base_prompt: PromptTemplate) -> None:
        """Initialize population with variations of the base prompt"""
        self.population = []
        
        # Add the original prompt
        self.population.append(base_prompt)
        
        # Create variations for the rest of the population
        for i in range(self.population_size - 1):
            # Simple variations as initial population
            variation = self._create_simple_variation(base_prompt, i)
            self.population.append(variation)
    
    def _create_simple_variation(self, base_prompt: PromptTemplate, index: int) -> PromptTemplate:
        """Create simple variations of the base prompt"""
        template = base_prompt.template
        
        # Add different prefixes/suffixes
        variations = [
            f"Please {template.lower()}",
            f"Task: {template}",
            f"{template}\n\nBe precise and clear.",
            f"{template}\n\nProvide specific details.",
            f"{template}\n\nThink step by step.",
            f"Answer the following: {template}",
            f"{template}\n\nUse examples when helpful.",
        ]
        
        variation_template = variations[index % len(variations)]
        
        return PromptTemplate(
            template=variation_template,
            name=f"{base_prompt.name}_var_{index}",
            metadata=base_prompt.metadata
        )
    
    async def _evaluate_population(
        self, 
        test_cases: Optional[List[TestCase]], 
        runner: Optional[PromptRunner], 
        **kwargs: Any
    ) -> List[FitnessEvaluation]:
        """Evaluate fitness for entire population"""
        if not self.fitness_function:
            raise ValueError("No fitness function provided")
        
        # Run evaluations in parallel for efficiency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent evaluations
        
        async def evaluate_single(prompt: PromptTemplate) -> FitnessEvaluation:
            async with semaphore:
                return await self.fitness_function.evaluate(prompt, test_cases, runner, **kwargs)
        
        tasks = [evaluate_single(prompt) for prompt in self.population]
        evaluations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return only FitnessEvaluation objects
        return [eval for eval in evaluations if isinstance(eval, FitnessEvaluation)]
    
    def _tournament_selection(self, evaluations: List[FitnessEvaluation], tournament_size: int = 3) -> PromptTemplate:
        """Select a parent using tournament selection"""
        tournament = random.sample(evaluations, min(tournament_size, len(evaluations)))
        winner = max(tournament, key=lambda x: x.score)
        return winner.prompt
    
    async def _create_next_generation_llm(self, evaluations: List[FitnessEvaluation]) -> None:
        """Create next generation using LLM-powered operations"""
        new_population = []
        
        # Sort by fitness (descending)
        sorted_evaluations = sorted(evaluations, key=lambda x: x.score, reverse=True)
        
        # Elitism: keep best individuals
        for i in range(min(self.elite_size, len(sorted_evaluations))):
            new_population.append(sorted_evaluations[i].prompt)
        
        # Generate offspring using LLM crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents (tournament selection)
            parent1 = self._tournament_selection(sorted_evaluations)
            parent2 = self._tournament_selection(sorted_evaluations)
            
            # Crossover
            if random.random() < self.crossover_rate and self.crossover:
                try:
                    offspring1, offspring2 = await self.crossover.crossover(parent1, parent2)
                    new_population.extend([offspring1, offspring2])
                except Exception:
                    # Fallback to simple crossover
                    offspring1, offspring2 = self._simple_crossover(parent1, parent2)
                    new_population.extend([offspring1, offspring2])
            else:
                # Just copy parents
                new_population.extend([parent1, parent2])
            
            # Mutation
            if random.random() < self.mutation_rate and self.mutator:
                try:
                    mutation_types = ["random", "improve_clarity", "add_examples", "optimize_structure"]
                    mutation_type = random.choice(mutation_types)
                    mutation_strength = random.uniform(0.3, 0.8)
                    
                    mutated = await self.mutator.mutate(
                        new_population[-1], 
                        mutation_type=mutation_type,
                        mutation_strength=mutation_strength
                    )
                    new_population[-1] = mutated
                except Exception:
                    # Fallback to simple mutation
                    pass
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
    
    def _simple_crossover(self, parent1: PromptTemplate, parent2: PromptTemplate) -> Tuple[PromptTemplate, PromptTemplate]:
        """Simple crossover fallback"""
        # Simple concatenation-based crossover
        offspring1_template = f"{parent1.template}\n\n{parent2.template}"
        offspring2_template = f"{parent2.template}\n\n{parent1.template}"
        
        return (
            PromptTemplate(template=offspring1_template, name=f"{parent1.name}_simple_offspring1", metadata=parent1.metadata),
            PromptTemplate(template=offspring2_template, name=f"{parent2.name}_simple_offspring2", metadata=parent2.metadata)
        )
    
    async def _log_generation_progress(self, generation: int, best_evaluation: FitnessEvaluation) -> None:
        """Log progress of optimization"""
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores) if self.fitness_scores else 0
        max_fitness = max(self.fitness_scores) if self.fitness_scores else 0
        
        print(f"Generation {generation + 1}/{self.generations}")
        print(f"  Best fitness: {best_evaluation.score:.3f}")
        print(f"  Average fitness: {avg_fitness:.3f}")
        print(f"  Max fitness: {max_fitness:.3f}")
        print(f"  Best prompt: {best_evaluation.prompt.template[:100]}...")
        print(f"  Reasoning: {best_evaluation.evaluation_reasoning[:100]}...")
        print()
