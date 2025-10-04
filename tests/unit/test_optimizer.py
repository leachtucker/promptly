"""
Tests for optimizer module
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from typing import Optional, List
from promptly.core.optimizer import (
    LLMGeneticOptimizer,
    LLMAccuracyFitnessFunction,
    LLMSemanticFitnessFunction,
    PromptTestCase,
    OptimizationResult,
    FitnessEvaluation,
    LLMPromptMutator,
    LLMPromptCrossover,
)
from promptly.core.templates import PromptTemplate
from promptly.core.clients import BaseLLMClient, LLMResponse
from promptly.core.tracer import Tracer, UsageData


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        response_text = self.responses[self.call_count % len(self.responses)] if self.responses else "Mock response"
        self.call_count += 1
        
        return LLMResponse(
            content=response_text,
            model=model or "mock-model",
            usage=UsageData(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            metadata={"mock": True}
        )
    
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


class TestLLMAccuracyFitnessFunction:
    """Test LLM accuracy fitness function"""
    
    @pytest.mark.asyncio
    async def test_evaluate_with_mock_client(self):
        """Test fitness evaluation with mock client"""
        mock_client = MockLLMClient([
            "SCORE: 0.8\nREASONING: The prompt performs well on most test cases."
        ])
        
        fitness_function = LLMAccuracyFitnessFunction(mock_client)
        
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
        
        result = await fitness_function.evaluate(prompt, test_cases, mock_runner)
        
        assert isinstance(result, FitnessEvaluation)
        assert result.score == 0.8
        assert result.evaluation_reasoning == "The prompt performs well on most test cases."
        assert len(result.test_results) == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_without_test_cases(self):
        """Test fitness evaluation without test cases (quality-based)"""
        mock_client = MockLLMClient([
            "SCORE: 0.75\nREASONING: The prompt is clear and well-structured."
        ])
        
        fitness_function = LLMAccuracyFitnessFunction(mock_client)
        
        prompt = PromptTemplate(template="Answer: {{question}}", name="test")
        
        result = await fitness_function.evaluate(prompt, test_cases=None, runner=None)
        
        assert isinstance(result, FitnessEvaluation)
        assert result.score == 0.75
        assert result.evaluation_reasoning == "The prompt is clear and well-structured."
        assert len(result.test_results) == 0
        assert result.metadata["evaluation_method"] == "prompt_quality"
    
    def test_parse_evaluation_response(self):
        """Test parsing of LLM evaluation response"""
        fitness_function = LLMAccuracyFitnessFunction(MockLLMClient())
        
        response = "SCORE: 0.75\nREASONING: Good performance overall."
        score, reasoning = fitness_function._parse_evaluation_response(response)
        
        assert score == 0.75
        assert reasoning == "Good performance overall."
    
    def test_parse_evaluation_response_fallback(self):
        """Test parsing fallback for malformed response"""
        fitness_function = LLMAccuracyFitnessFunction(MockLLMClient())
        
        response = "Invalid response format"
        score, reasoning = fitness_function._parse_evaluation_response(response)
        
        assert score == 0.5
        assert "Could not parse" in reasoning
    
    def test_calculate_simple_accuracy(self):
        """Test simple accuracy calculation"""
        fitness_function = LLMAccuracyFitnessFunction(MockLLMClient())
        
        test_results = [
            {"actual": "4", "expected": "4"},
            {"actual": "6", "expected": "6"},
            {"actual": "5", "expected": "4"},  # Wrong
        ]
        
        accuracy = fitness_function._calculate_simple_accuracy(test_results)
        assert accuracy == 2/3  # 2 out of 3 correct


class TestLLMPromptMutator:
    """Test LLM prompt mutator"""
    
    @pytest.mark.asyncio
    async def test_mutate_with_mock_client(self):
        """Test prompt mutation with mock client"""
        mock_client = MockLLMClient([
            "Improved prompt template with better instructions."
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
        mock_client.generate.side_effect = Exception("API error")
        
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
        mock_client = MockLLMClient([
            "OFFSPRING 1:\nCombined prompt 1\n\nOFFSPRING 2:\nCombined prompt 2"
        ])
        
        crossover = LLMPromptCrossover(mock_client)
        parent1 = PromptTemplate(template="Parent 1", name="p1")
        parent2 = PromptTemplate(template="Parent 2", name="p2")
        
        result1, result2 = await crossover.crossover(parent1, parent2)
        
        assert isinstance(result1, PromptTemplate)
        assert isinstance(result2, PromptTemplate)
        assert result1.template == "Combined prompt 1"
        assert result2.template == "Combined prompt 2"
    
    @pytest.mark.asyncio
    async def test_crossover_fallback(self):
        """Test crossover fallback when LLM fails"""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("API error")
        
        crossover = LLMPromptCrossover(mock_client)
        parent1 = PromptTemplate(template="Parent 1", name="p1")
        parent2 = PromptTemplate(template="Parent 2", name="p2")
        
        result1, result2 = await crossover.crossover(parent1, parent2)
        
        assert isinstance(result1, PromptTemplate)
        assert isinstance(result2, PromptTemplate)
        assert "Parent 1" in result1.template
        assert "Parent 2" in result2.template
    
    def test_simple_crossover(self):
        """Test simple crossover fallback"""
        crossover = LLMPromptCrossover(MockLLMClient())
        parent1 = PromptTemplate(template="Parent 1", name="p1")
        parent2 = PromptTemplate(template="Parent 2", name="p2")
        
        result1, result2 = crossover._simple_crossover(parent1, parent2)
        
        assert isinstance(result1, PromptTemplate)
        assert isinstance(result2, PromptTemplate)
        assert "Parent 1" in result1.template and "Parent 2" in result1.template
        assert "Parent 2" in result2.template and "Parent 1" in result2.template


class TestLLMGeneticOptimizer:
    """Test LLM genetic optimizer"""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        mock_client = MockLLMClient()
        fitness_function = LLMAccuracyFitnessFunction(mock_client)
        
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
        fitness_function = LLMAccuracyFitnessFunction(mock_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=5,
            generations=3,
            fitness_function=fitness_function
        )
        
        base_prompt = PromptTemplate(template="Base prompt", name="base")
        optimizer._initialize_population(base_prompt)
        
        assert len(optimizer.population) == 5
        assert optimizer.population[0].template == "Base prompt"
        # Other prompts should be variations
        for i in range(1, 5):
            assert optimizer.population[i].name.startswith("base_var_")
    
    def test_tournament_selection(self):
        """Test tournament selection"""
        mock_client = MockLLMClient()
        fitness_function = LLMAccuracyFitnessFunction(mock_client)
        
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
        mock_main_client = MockLLMClient(["4", "6", "8"])
        mock_eval_client = MockLLMClient([
            "SCORE: 0.8\nREASONING: Good performance.",
            "SCORE: 0.9\nREASONING: Excellent performance.",
        ])
        mock_mutation_client = MockLLMClient(["Improved prompt"])
        mock_crossover_client = MockLLMClient([
            "OFFSPRING 1:\nCombined 1\n\nOFFSPRING 2:\nCombined 2"
        ])
        
        fitness_function = LLMAccuracyFitnessFunction(mock_eval_client)
        
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
        
        result = await optimizer.optimize(base_prompt, test_cases, mock_runner)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_prompt is not None
        assert result.fitness_score > 0
        assert result.generation == 1  # 0-indexed, so 2 generations = generation 1
        assert result.total_evaluations > 0
        assert result.optimization_time > 0
    
    @pytest.mark.asyncio
    async def test_optimize_without_test_cases(self):
        """Test optimization without test cases (quality-based)"""
        # Mock clients with predictable responses
        mock_eval_client = MockLLMClient([
            "SCORE: 0.8\nREASONING: Good prompt quality.",
            "SCORE: 0.9\nREASONING: Excellent prompt quality.",
        ])
        mock_mutation_client = MockLLMClient(["Improved prompt"])
        mock_crossover_client = MockLLMClient([
            "OFFSPRING 1:\nCombined 1\n\nOFFSPRING 2:\nCombined 2"
        ])
        
        fitness_function = LLMAccuracyFitnessFunction(mock_eval_client)
        
        optimizer = LLMGeneticOptimizer(
            population_size=3,
            generations=2,
            fitness_function=fitness_function,
            mutation_client=mock_mutation_client,
            crossover_client=mock_crossover_client
        )
        
        base_prompt = PromptTemplate(template="Answer: {{question}}", name="base")
        
        result = await optimizer.optimize(base_prompt, test_cases=None, runner=None)
        
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
