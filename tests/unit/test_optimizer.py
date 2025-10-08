"""
Tests for optimizer module
"""

import pytest
from typing import Optional, List
from promptly.core.optimizer import (
    LLMGeneticOptimizer,
    LLMComprehensiveFitnessFunction,
    PromptTestCase,
    OptimizationResult,
    FitnessEvaluation,
    LLMPromptMutator,
    LLMPromptCrossover,
    LLMPopulationGenerator,
)
from promptly.core.templates import PromptTemplate
from promptly.core.clients import BaseLLMClient, LLMResponse
from promptly.core.tracer import UsageData


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, responses=None, structured_responses=None):
        self.responses = responses or []
        self.structured_responses = structured_responses or []
        self.call_count = 0
        self.structured_call_count = 0
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        if self.structured_responses:
            # Return JSON response for structured calls
            import json
            response_data = self.structured_responses[self.call_count % len(self.structured_responses)]
            response_text = json.dumps(response_data)
        else:
            response_text = self.responses[self.call_count % len(self.responses)] if self.responses else "Mock response"
        
        self.call_count += 1
        
        return LLMResponse(
            content=response_text,
            model=model or "mock-model",
            usage=UsageData(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            metadata={"mock": True}
        )
    
    async def generate_structured(self, prompt: str, response_model, model: Optional[str] = None, **kwargs):
        """Mock structured generation"""
        if self.structured_responses:
            response_data = self.structured_responses[self.structured_call_count % len(self.structured_responses)]
            self.structured_call_count += 1
            return response_model(**response_data)
        else:
            # Return a default instance
            return response_model()
    
    def get_available_models(self) -> List[str]:
        return ["mock-model"]


class TestPromptTestCase:
    """Test PromptTestCase Pydantic model"""
    
    def test_test_case_creation(self):
        """Test PromptTestCase creation"""
        test_case = PromptTestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4",
            metadata={"difficulty": "easy"}
        )
        
        assert test_case.input_variables == {"question": "What is 2+2?"}
        assert test_case.expected_output == "4"
        assert test_case.metadata == {"difficulty": "easy"}
    
    def test_test_case_defaults(self):
        """Test PromptTestCase with default values"""
        test_case = PromptTestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4"
        )
        
        assert test_case.metadata == {}


class TestFitnessEvaluation:
    """Test FitnessEvaluation dataclass"""
    
    def test_fitness_evaluation_creation(self):
        """Test FitnessEvaluation creation"""
        prompt = PromptTemplate(template="Test prompt", name="test")
        evaluation = FitnessEvaluation(
            prompt=prompt,
            score=0.85,
            test_results=[{"correct": True}],
            evaluation_reasoning="Good performance",
            metadata={"method": "accuracy"}
        )
        
        assert evaluation.score == 0.85
        assert evaluation.evaluation_reasoning == "Good performance"
        assert evaluation.metadata == {"method": "accuracy"}


class TestLLMComprehensiveFitnessFunction:
    """Test LLM comprehensive fitness function"""
    
    @pytest.mark.asyncio
    async def test_evaluate_with_mock_client(self, tracer_with_temp_db):
        """Test fitness evaluation with mock client"""
        mock_client = MockLLMClient(structured_responses=[
            {"score": 0.8, "reasoning": "The prompt performs well on most test cases."}
        ])
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_client, "gpt-4")
        
        prompt = PromptTemplate(template="Answer: {{question}}", name="test")
        test_cases = [
            PromptTestCase(input_variables={"question": "2+2"}, expected_output="4"),
            PromptTestCase(input_variables={"question": "3+3"}, expected_output="6"),
        ]
        
        # Create proper mock runner with tracer
        from promptly.core.runner import PromptRunner
        mock_runner = PromptRunner(client=mock_client, tracer=tracer_with_temp_db)
        
        result = await fitness_function.evaluate(runner=mock_runner, prompt=prompt, model="gpt-4", test_cases=test_cases)
        
        assert isinstance(result, FitnessEvaluation)
        assert result.score == 0.8
        assert result.evaluation_reasoning == "The prompt performs well on most test cases."
        assert len(result.test_results) == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_without_test_cases(self, tracer_with_temp_db):
        """Test fitness evaluation without test cases - evaluates prompt quality directly"""
        # Mock client returns quality-based evaluation
        mock_client = MockLLMClient(structured_responses=[
            {"score": 0.75, "reasoning": "The prompt is clear and well-structured."}
        ])
        
        # Create proper mock runner with tracer
        from promptly.core.runner import PromptRunner
        mock_runner = PromptRunner(client=mock_client, tracer=tracer_with_temp_db)
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_client, "gpt-4")
        prompt = PromptTemplate(template="Write a helpful and informative response.", name="test")
        
        # When no test cases provided, evaluates based on prompt quality
        result = await fitness_function.evaluate(runner=mock_runner, prompt=prompt, model="gpt-4", test_cases=None)
        
        assert isinstance(result, FitnessEvaluation)
        assert result.score == 0.75
        assert result.evaluation_reasoning == "The prompt is clear and well-structured."
        assert len(result.test_results) == 0
        assert result.metadata["evaluation_method"] == "comprehensive_prompt_quality"


class TestLLMPromptMutator:
    """Test LLM prompt mutator"""
    
    @pytest.mark.asyncio
    async def test_mutate_with_mock_client(self):
        """Test prompt mutation with mock client"""
        mock_client = MockLLMClient(structured_responses=[
            {"mutated_prompt": "Improved prompt template with better instructions."}
        ])
        
        mutator = LLMPromptMutator(mock_client, "gpt-4")
        prompt = PromptTemplate(template="Original prompt", name="test")
        
        result = await mutator.mutate(prompt, mutation_type="improve_clarity")
        
        assert isinstance(result, PromptTemplate)
        assert result.template == "Improved prompt template with better instructions."
        assert result.name == "test_mutated"


class TestLLMPromptCrossover:
    """Test LLM prompt crossover"""
    
    @pytest.mark.asyncio
    async def test_crossover_with_mock_client(self):
        """Test prompt crossover with mock client"""
        mock_client = MockLLMClient(structured_responses=[
            {"offspring1": "Combined prompt 1", "offspring2": "Combined prompt 2"}
        ])
        
        crossover = LLMPromptCrossover(mock_client, "gpt-4")
        parent1 = PromptTemplate(template="Parent 1", name="p1")
        parent2 = PromptTemplate(template="Parent 2", name="p2")
        
        result1, result2 = await crossover.crossover(parent1, parent2)
        
        assert isinstance(result1, PromptTemplate)
        assert isinstance(result2, PromptTemplate)
        assert result1.template == "Combined prompt 1"
        assert result2.template == "Combined prompt 2"


class TestLLMGeneticOptimizer:
    """Test LLM genetic optimizer"""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client, "gpt-4")
        
        optimizer = LLMGeneticOptimizer(
            eval_model="gpt-4",
            population_size=5,
            generations=3,
            fitness_function=fitness_function,
            eval_client=mock_client,
        )
        
        assert optimizer.population_size == 5
        assert optimizer.generations == 3
        assert optimizer.fitness_function == fitness_function
    
    @pytest.mark.asyncio
    async def test_initialize_population(self):
        """Test population initialization (LLM-driven)"""
        # Mock successful structured response
        mock_client = MockLLMClient(structured_responses=[
            {
                "variations": [
                    "Variation 1",
                    "Variation 2",
                    "Variation 3",
                    "Variation 4"
                ]
            }
        ])
        fitness_function = LLMComprehensiveFitnessFunction(mock_client, "gpt-4")
        
        optimizer = LLMGeneticOptimizer(
            eval_model="gpt-4",
            population_size=5,
            generations=3,
            fitness_function=fitness_function,
            eval_client=mock_client
        )
        
        base_prompt = PromptTemplate(template="Base prompt", name="base")
        await optimizer._initialize_population(base_prompt)
        
        assert len(optimizer.population) == 5
        assert optimizer.population[0].template == "Base prompt"
    
    def test_tournament_selection(self):
        """Test tournament selection"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client, "gpt-4")
        
        optimizer = LLMGeneticOptimizer(
            eval_model="gpt-4",
            population_size=5,
            generations=3,
            fitness_function=fitness_function,
            eval_client=mock_client,
        )
        
        # Create mock evaluations with different scores
        evaluations = [
            FitnessEvaluation(
                prompt=PromptTemplate(template="Prompt 1", name="p1"),
                score=0.5,
                test_results=[],
                evaluation_reasoning=""
            ),
            FitnessEvaluation(
                prompt=PromptTemplate(template="Prompt 2", name="p2"),
                score=0.9,
                test_results=[],
                evaluation_reasoning=""
            ),
            FitnessEvaluation(
                prompt=PromptTemplate(template="Prompt 3", name="p3"),
                score=0.3,
                test_results=[],
                evaluation_reasoning=""
            ),
        ]
        
        # Run tournament selection multiple times
        selected_scores = []
        for _ in range(10):
            selected = optimizer._tournament_selection(evaluations)
            # Find the evaluation for the selected prompt
            selected_eval = next(e for e in evaluations if e.prompt == selected)
            selected_scores.append(selected_eval.score)
        
        # Should tend to select higher-scoring prompts
        avg_score = sum(selected_scores) / len(selected_scores)
        assert avg_score > 0.5  # Should be higher than the average of all scores
    
    @pytest.mark.asyncio
    async def test_optimize_mock_run(self, tracer_with_temp_db):
        """Test optimization with mocked components"""
        # Mock clients with predictable responses
        # Need to provide:
        # 1. Population generation (1 call)
        # 2. Fitness evaluations (population_size * generations = 3 * 2 = 6 calls)
        mock_eval_client = MockLLMClient(structured_responses=[
            # 1. Population generation - create 2 variations (plus base = 3 total)
            {
                "variations": [
                    "Please answer: {{question}}",
                    "Response to {{question}}"
                ]
            },
            # 2. Generation 1 evaluations (3 individuals)
            {"score": 0.7, "reasoning": "Decent performance."},
            {"score": 0.8, "reasoning": "Good performance."},
            {"score": 0.75, "reasoning": "Fair performance."},
            # 3. Generation 2 evaluations (3 individuals)
            {"score": 0.85, "reasoning": "Very good performance."},
            {"score": 0.9, "reasoning": "Excellent performance."},
            {"score": 0.88, "reasoning": "Strong performance."},
        ])
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_eval_client, "gpt-4")
        
        optimizer = LLMGeneticOptimizer(
            eval_model="gpt-4",
            population_size=3,
            generations=2,
            fitness_function=fitness_function,
            eval_client=mock_eval_client,
            tracer=tracer_with_temp_db,
            mutation_rate=0.0,  # Disable for predictable test
            crossover_rate=0.0,  # Disable for predictable test
        )
        
        base_prompt = PromptTemplate(template="Answer: {{question}}", name="base")
        test_cases = [
            PromptTestCase(input_variables={"question": "2+2"}, expected_output="4"),
        ]
        
        # Create proper mock runner with tracer
        from promptly.core.runner import PromptRunner
        mock_runner = PromptRunner(client=mock_eval_client, tracer=tracer_with_temp_db)
        
        result = await optimizer.optimize(mock_runner, base_prompt, test_cases, model="gpt-4")
        
        assert isinstance(result, OptimizationResult)
        assert result.best_prompt is not None
        assert result.fitness_score > 0
        assert result.generation == 1  # 0-indexed, so 2 generations = generation 1
        assert result.total_evaluations > 0
        assert result.optimization_time > 0
    
    @pytest.mark.asyncio
    async def test_optimize_without_test_cases(self):
        """Test optimization without test cases - evaluates based on prompt quality"""
        # When no test cases provided, quality-based evaluation is used
        # Need to provide responses for:
        # 1. Population generation (1 call)
        # 2. Fitness evaluations (population_size * generations = 3 * 2 = 6 calls)
        # 3. Crossover operations (may happen in generation 2)
        # 4. Mutation operations (may happen in generation 2)
        
        mock_eval_client = MockLLMClient(structured_responses=[
            # 1. Population generation - create 2 variations (plus base = 3 total)
            {
                "variations": [
                    "Write a helpful and clear response.",
                    "Provide an informative and useful answer."
                ]
            },
            # 2. Generation 1 evaluations (3 individuals)
            {"score": 0.6, "reasoning": "Decent prompt quality."},
            {"score": 0.75, "reasoning": "Good prompt quality."},
            {"score": 0.65, "reasoning": "Acceptable prompt quality."},
            # 3. Crossover for generation 2
            {"offspring1": "Write helpful responses clearly.", "offspring2": "Provide clear and useful information."},
            # 4. Mutation for generation 2 (may or may not happen depending on rate)
            {"mutated_prompt": "Write exceptionally helpful and informative responses with clarity."},
            # 5. Generation 2 evaluations (3 individuals)
            {"score": 0.85, "reasoning": "Very good prompt quality."},
            {"score": 0.9, "reasoning": "Excellent prompt quality."},
            {"score": 0.88, "reasoning": "High quality prompt."},
        ])
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_eval_client, "gpt-4")
        
        # Create optimizer with controlled rates to ensure predictable behavior
        optimizer = LLMGeneticOptimizer(
            eval_model="gpt-4",
            population_size=3,
            generations=2,
            fitness_function=fitness_function,
            eval_client=mock_eval_client,
            mutation_rate=0.0,  # Disable mutation for predictable test
            crossover_rate=0.0,  # Disable crossover for predictable test
        )
        
        # Use a template without variables since quality evaluation doesn't provide test inputs
        base_prompt = PromptTemplate(template="Write a helpful and informative response.", name="base")
        
        # Create a proper mock runner
        from promptly.core.runner import PromptRunner
        mock_runner = PromptRunner(client=mock_eval_client)
        
        result = await optimizer.optimize(mock_runner, base_prompt, test_cases=None, model="gpt-4")
        
        assert isinstance(result, OptimizationResult)
        assert result.best_prompt is not None
        assert result.fitness_score > 0
        assert result.generation == 1  # 0-indexed, so 2 generations = generation 1
        assert result.total_evaluations > 0
        assert result.optimization_time > 0


class TestOptimizationResult:
    """Test OptimizationResult dataclass"""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation"""
        prompt = PromptTemplate(template="Optimized prompt", name="optimized")
        
        result = OptimizationResult(
            best_prompt=prompt,
            fitness_score=0.85,
            generation=5,
            population_size=10,
            total_evaluations=50,
            optimization_time=120.5,
            metadata={"method": "genetic"}
        )
        
        assert result.best_prompt == prompt
        assert result.fitness_score == 0.85
        assert result.generation == 5
        assert result.population_size == 10
        assert result.total_evaluations == 50
        assert result.optimization_time == 120.5
        assert result.metadata == {"method": "genetic"}


# Integration tests
class TestOptimizerIntegration:
    """Integration tests for optimizer components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test complete optimization workflow"""
        # This would be a more comprehensive test with real (but mocked) LLM calls
        pass
    
    def test_cli_compatibility(self):
        """Test that optimizer works with CLI interface"""
        # This would test the CLI integration
        pass


class TestLLMPopulationGenerator:
    """Test LLM population generator"""
    
    def test_population_generator_initialization(self):
        """Test population generator initialization"""
        mock_client = MockLLMClient()
        generator = LLMPopulationGenerator(mock_client, "gpt-4")
        
        assert generator.generation_client == mock_client
        assert generator.generation_model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_generate_initial_population_success(self):
        """Test successful population generation"""
        # Mock structured response
        mock_client = MockLLMClient(structured_responses=[
            {
                "variations": [
                    "Please answer the following question: {{question}}",
                    "Task: {{question}}",
                    "{{question}}\n\nPlease provide a detailed response."
                ]
            }
        ])
        generator = LLMPopulationGenerator(mock_client)
        
        base_prompt = PromptTemplate(
            template="Answer this: {{question}}",
            name="test_prompt"
        )
        
        population = await generator.generate_initial_population(base_prompt, 4, 0.7)
        
        assert len(population) == 4
        assert population[0].template == "Answer this: {{question}}"  # Original prompt
        assert "{{question}}" in population[1].template
        assert "{{question}}" in population[2].template
        assert "{{question}}" in population[3].template
        
        # Check naming
        assert population[1].name.startswith("test_prompt_llm_var_")
        assert population[2].name.startswith("test_prompt_llm_var_")
        assert population[3].name.startswith("test_prompt_llm_var_")
    
    
    def test_create_generation_prompt(self):
        """Test prompt generation for LLM"""
        mock_client = MockLLMClient()
        generator = LLMPopulationGenerator(mock_client)
        
        base_prompt = PromptTemplate(
            template="Answer this: {{question}}",
            name="test_prompt"
        )
        
        prompt = generator._create_generation_prompt(base_prompt, 3, 0.8)
        
        assert "Answer this: {{question}}" in prompt
        assert "3" in prompt
        assert "0.8" in prompt
        assert "variations" in prompt.lower()
    
    def test_create_variations_from_structured(self):
        """Test creating variations from structured response"""
        mock_client = MockLLMClient()
        generator = LLMPopulationGenerator(mock_client)
        
        base_prompt = PromptTemplate(
            template="Answer this: {{question}}",
            name="test_prompt"
        )
        
        variations_data = [
            "Please answer: {{question}}",
            "{{question}}\n\nPlease provide details."
        ]
        
        variations = generator._create_variations_from_structured(variations_data, base_prompt)
        
        assert len(variations) == 2
        assert "Please answer: {{question}}" in variations[0].template
        assert "{{question}}" in variations[1].template
        assert "Please provide details." in variations[1].template
        assert variations[0].name.startswith("test_prompt_llm_var_")
        assert variations[1].name.startswith("test_prompt_llm_var_")
    


class TestLLMGeneticOptimizerWithPopulationGeneration:
    """Test LLMGeneticOptimizer with LLM population generation"""
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """Test optimizer initialization (strictly LLM-driven)"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client, "gpt-4")
        
        optimizer = LLMGeneticOptimizer(
            eval_model="gpt-4",
            population_size=5,
            generations=2,
            fitness_function=fitness_function,
            eval_client=mock_client,
            population_diversity_level=0.8
        )
        
        assert optimizer.population_diversity_level == 0.8
        assert optimizer.population_generator is not None
    
    @pytest.mark.asyncio
    async def test_initialize_population_with_llm(self):
        """Test population initialization using LLM (strictly LLM-driven)"""
        # Mock successful structured response
        mock_client = MockLLMClient(structured_responses=[
            {
                "variations": [
                    "Please answer: {{question}}",
                    "{{question}}\n\nProvide details.",
                    "Answer the following question: {{question}}\n\nBe specific and clear."
                ]
            }
        ])
        fitness_function = LLMComprehensiveFitnessFunction(mock_client, "gpt-4")
        
        optimizer = LLMGeneticOptimizer(
            eval_model="gpt-4",
            population_size=4,
            generations=2,
            fitness_function=fitness_function,
            eval_client=mock_client 
        )
        
        base_prompt = PromptTemplate(
            template="Answer this: {{question}}",
            name="test_prompt"
        )
        
        await optimizer._initialize_population(base_prompt)
        
        assert len(optimizer.population) == 4
        assert optimizer.population[0].template == "Answer this: {{question}}"
        assert optimizer.population[1].name.startswith("test_prompt_llm_var_")
