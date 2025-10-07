"""
Tests for optimizer module
"""

import pytest
from unittest.mock import AsyncMock
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
    async def test_evaluate_with_mock_client(self):
        """Test fitness evaluation with mock client"""
        mock_client = MockLLMClient(structured_responses=[
            {"score": 0.8, "reasoning": "The prompt performs well on most test cases."}
        ])
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        prompt = PromptTemplate(template="Answer: {{question}}", name="test")
        test_cases = [
            PromptTestCase(input_variables={"question": "2+2"}, expected_output="4"),
            PromptTestCase(input_variables={"question": "3+3"}, expected_output="6"),
        ]
        
        # Mock the runner
        mock_runner = AsyncMock()
        mock_runner.run.return_value = LLMResponse(
            content="4",
            model="test-model",
            usage=UsageData()
        )
        
        result = await fitness_function.evaluate(mock_runner, prompt, test_cases)
        
        assert isinstance(result, FitnessEvaluation)
        assert result.score == 0.8
        assert result.evaluation_reasoning == "The prompt performs well on most test cases."
        assert len(result.test_results) == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_without_test_cases(self):
        """Test fitness evaluation without test cases - evaluates prompt quality directly"""
        # Mock client returns quality-based evaluation
        mock_client = MockLLMClient(structured_responses=[
            {"score": 0.75, "reasoning": "The prompt is clear and well-structured."}
        ])
        
        # Mock runner needed for quality evaluation
        mock_runner = AsyncMock()
        mock_runner.run.return_value = LLMResponse(
            content="Sample output from the prompt",
            model="test-model",
            usage=UsageData()
        )
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        prompt = PromptTemplate(template="Answer: {{question}}", name="test")
        
        # When no test cases provided, evaluates based on prompt quality
        result = await fitness_function.evaluate(mock_runner, prompt, test_cases=None)
        
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
        
        mutator = LLMPromptMutator(mock_client)
        prompt = PromptTemplate(template="Original prompt", name="test")
        
        result = await mutator.mutate(prompt, mutation_type="improve_clarity")
        
        assert isinstance(result, PromptTemplate)
        assert result.template == "Improved prompt template with better instructions."
        assert result.name == "test_mutated"
    
    @pytest.mark.asyncio
    async def test_mutate_fallback(self):
        """Test mutation fallback when LLM fails"""
        # Mock client that raises an exception
        mock_client = AsyncMock()
        mock_client.generate_structured.side_effect = Exception("API error")
        
        mutator = LLMPromptMutator(mock_client)
        prompt = PromptTemplate(template="Original prompt", name="test")
        
        result = await mutator.mutate(prompt)
        
        assert isinstance(result, PromptTemplate)
        assert result.name == "test_simple_mutated"
        # Should contain some variation of the original (case insensitive)
        assert "original prompt" in result.template.lower()
    
    def test_simple_mutation(self):
        """Test simple mutation fallback"""
        mutator = LLMPromptMutator(MockLLMClient())
        prompt = PromptTemplate(template="Original prompt", name="test")
        
        result = mutator._simple_mutation(prompt)
        
        assert isinstance(result, PromptTemplate)
        assert result.name == "test_simple_mutated"
        assert len(result.template) > len(prompt.template)


class TestLLMPromptCrossover:
    """Test LLM prompt crossover"""
    
    @pytest.mark.asyncio
    async def test_crossover_with_mock_client(self):
        """Test prompt crossover with mock client"""
        mock_client = MockLLMClient(structured_responses=[
            {"offspring1": "Combined prompt 1", "offspring2": "Combined prompt 2"}
        ])
        
        crossover = LLMPromptCrossover(mock_client)
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
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=5,
            generations=3,
            fitness_function=fitness_function,
            mutation_client=mock_client,
            crossover_client=mock_client
        )
        
        assert optimizer.population_size == 5
        assert optimizer.generations == 3
        assert optimizer.fitness_function == fitness_function
        assert optimizer.mutation_client == mock_client
        assert optimizer.crossover_client == mock_client
    
    def test_initialize_population(self):
        """Test population initialization"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=5,
            generations=3,
            fitness_function=fitness_function
        )
        
        base_prompt = PromptTemplate(template="Base prompt", name="base")
        # Use the simple initialization method for this test
        optimizer._initialize_population_simple(base_prompt)
        
        assert len(optimizer.population) == 5
        assert optimizer.population[0].template == "Base prompt"
        # Other prompts should be variations
        for i in range(1, 5):
            assert optimizer.population[i].name.startswith("base_var_")
    
    def test_tournament_selection(self):
        """Test tournament selection"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=5,
            generations=3,
            fitness_function=fitness_function
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
    async def test_optimize_mock_run(self):
        """Test optimization with mocked components"""
        # Mock clients with predictable responses
        mock_eval_client = MockLLMClient(structured_responses=[
            {"score": 0.8, "reasoning": "Good performance."},
            {"score": 0.9, "reasoning": "Excellent performance."},
        ])
        mock_mutation_client = MockLLMClient(structured_responses=[
            {"mutated_prompt": "Improved prompt"}
        ])
        mock_crossover_client = MockLLMClient(structured_responses=[
            {"offspring1": "Combined 1", "offspring2": "Combined 2"}
        ])
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_eval_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=3,
            generations=2,
            fitness_function=fitness_function,
            mutation_client=mock_mutation_client,
            crossover_client=mock_crossover_client
        )
        
        base_prompt = PromptTemplate(template="Answer: {{question}}", name="base")
        test_cases = [
            PromptTestCase(input_variables={"question": "2+2"}, expected_output="4"),
        ]
        
        # Mock runner
        mock_runner = AsyncMock()
        mock_runner.run.return_value = LLMResponse(
            content="4",
            model="test-model",
            usage=UsageData()
        )
        
        result = await optimizer.optimize(mock_runner, base_prompt, test_cases)
        
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
        # Provide enough responses for multiple generations of evaluations
        mock_eval_client = MockLLMClient(structured_responses=[
            {"score": 0.6, "reasoning": "Decent prompt quality."},
            {"score": 0.75, "reasoning": "Good prompt quality."},
            {"score": 0.85, "reasoning": "Very good prompt quality."},
            {"score": 0.8, "reasoning": "Good quality with clear structure."},
            {"score": 0.9, "reasoning": "Excellent prompt quality."},
            {"score": 0.88, "reasoning": "High quality prompt."},
            {"score": 0.82, "reasoning": "Strong prompt design."},
            {"score": 0.78, "reasoning": "Well-structured prompt."},
        ])
        mock_mutation_client = MockLLMClient(structured_responses=[
            {"mutated_prompt": "Write a clear and comprehensive response with helpful details."},
            {"mutated_prompt": "Provide a well-structured and informative answer."},
            {"mutated_prompt": "Create an engaging and detailed response."},
        ])
        mock_crossover_client = MockLLMClient(structured_responses=[
            {"offspring1": "Write a thorough and helpful response.", "offspring2": "Provide clear and detailed information."},
            {"offspring1": "Create a well-organized answer.", "offspring2": "Write an informative explanation."},
        ])
        
        fitness_function = LLMComprehensiveFitnessFunction(mock_eval_client)
        
        # Create optimizer
        optimizer = LLMGeneticOptimizer(
            population_size=3,
            generations=2,
            fitness_function=fitness_function,
            mutation_client=mock_mutation_client,
            crossover_client=mock_crossover_client,
        )
        
        # Use a template without variables since quality evaluation doesn't provide test inputs
        base_prompt = PromptTemplate(template="Write a helpful and informative response.", name="base")
        
        # Create a proper mock runner
        from promptly.core.runner import PromptRunner
        mock_runner = PromptRunner(client=mock_eval_client)
        
        result = await optimizer.optimize(mock_runner, base_prompt, test_cases=None)
        
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
    async def test_optimizer_with_llm_population_generation(self):
        """Test optimizer initialization with LLM population generation"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=5,
            generations=2,
            fitness_function=fitness_function,
            population_generator_client=mock_client,
            use_llm_population_generation=True,
            population_diversity_level=0.8
        )
        
        assert optimizer.use_llm_population_generation is True
        assert optimizer.population_diversity_level == 0.8
        assert optimizer.population_generator is not None
        assert optimizer.population_generator_client == mock_client
    
    @pytest.mark.asyncio
    async def test_optimizer_without_llm_population_generation(self):
        """Test optimizer initialization without LLM population generation"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=5,
            generations=2,
            fitness_function=fitness_function,
            use_llm_population_generation=False
        )
        
        assert optimizer.use_llm_population_generation is False
        assert optimizer.population_generator is None
        assert optimizer.population_generator_client is None
    
    @pytest.mark.asyncio
    async def test_initialize_population_with_llm(self):
        """Test population initialization using LLM"""
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
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=4,
            generations=2,
            fitness_function=fitness_function,
            population_generator_client=mock_client,
            use_llm_population_generation=True
        )
        
        base_prompt = PromptTemplate(
            template="Answer this: {{question}}",
            name="test_prompt"
        )
        
        await optimizer._initialize_population(base_prompt)
        
        assert len(optimizer.population) == 4
        assert optimizer.population[0].template == "Answer this: {{question}}"
        assert optimizer.population[1].name.startswith("test_prompt_llm_var_")
    
    @pytest.mark.asyncio
    async def test_initialize_population_simple_when_disabled(self):
        """Test simple population initialization when LLM generation is disabled"""
        mock_client = MockLLMClient()
        fitness_function = LLMComprehensiveFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=4,
            generations=2,
            fitness_function=fitness_function,
            use_llm_population_generation=False
        )
        
        base_prompt = PromptTemplate(
            template="Answer this: {{question}}",
            name="test_prompt"
        )
        
        await optimizer._initialize_population(base_prompt)
        
        assert len(optimizer.population) == 4
        assert optimizer.population[0].template == "Answer this: {{question}}"
        assert optimizer.population[1].name.startswith("test_prompt_var_")
