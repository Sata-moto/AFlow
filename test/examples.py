"""
工作流测试示例

这个脚本展示了如何使用测试工具来分析 DROP Round 8 的性能问题
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test.workflow_tester import WorkflowTester, load_test_data
from test.detailed_tracer import test_with_detailed_trace, compare_workflows


async def example_1_basic_test():
    """示例 1: 基础测试 - 测试 DROP Round 8"""
    print("\n" + "="*80)
    print("示例 1: 基础测试 DROP Round 8")
    print("="*80 + "\n")
    
    # 创建测试器
    tester = WorkflowTester(dataset='DROP', round_num=8)
    
    # 加载工作流
    tester.load_workflow()
    
    # 加载测试数据（3 个问题）
    test_data = load_test_data('DROP', limit=3)
    
    # 运行测试
    results = await tester.test_dataset(test_data, trace_execution=False)
    
    # 分析结果
    analysis = tester.analyze_results(results)
    
    print("\n分析结果:")
    print(f"  总问题数: {analysis['total_problems']}")
    print(f"  总成本: ${analysis['total_cost']:.6f}")
    print(f"  平均成本: ${analysis['average_cost']:.6f}")
    print(f"  最小成本: ${analysis['min_cost']:.6f}")
    print(f"  最大成本: ${analysis['max_cost']:.6f}")
    
    # 保存结果
    tester.save_results(results, 'test/example_1_results.json')
    print("\n结果已保存到: test/example_1_results.json")


async def example_2_detailed_trace():
    """示例 2: 详细追踪 - 查看每一步的执行"""
    print("\n" + "="*80)
    print("示例 2: 详细追踪 DROP Round 8")
    print("="*80 + "\n")
    
    # 运行详细追踪（只测试 2 个问题）
    results = await test_with_detailed_trace('DROP', round_num=8, num_problems=2)
    
    # 结果已经在函数中打印和保存
    print("\n详细追踪完成！")
    print("查看文件: test/detailed_trace_DROP_round_8.json")


async def example_3_compare_rounds():
    """示例 3: 对比不同轮次"""
    print("\n" + "="*80)
    print("示例 3: 对比 DROP Round 7 和 Round 8")
    print("="*80 + "\n")
    
    # 对比 Round 7 和 Round 8（各测试 5 个问题）
    await compare_workflows('DROP', round_nums=[7, 8], num_problems=5)
    
    print("\n对比完成！")
    print("查看文件: test/comparison_DROP_rounds_7_8.json")


async def example_4_single_problem():
    """示例 4: 测试单个问题（最详细）"""
    print("\n" + "="*80)
    print("示例 4: 测试单个问题")
    print("="*80 + "\n")
    
    # 创建测试器
    tester = WorkflowTester(dataset='DROP', round_num=8)
    tester.load_workflow()
    
    # 自定义问题
    test_problem = """
    Context: The Bears began their 2008 campaign at home against the Indianapolis Colts. 
    In the first quarter, Chicago drew first blood as kicker Robbie Gould got a 
    38-yard field goal. The Colts responded with kicker Adam Vinatieri getting a 
    24-yard field goal. In the second quarter, the Bears regained the lead as QB Kyle Orton 
    completed a 17-yard TD pass to WR Marty Booker. Indianapolis answered with RB Joseph Addai 
    getting a 2-yard TD run. Chicago closed out the half with Gould nailing a 40-yard field goal.
    
    Question: How many field goals did Robbie Gould kick in the first half?
    """
    
    # 测试这个问题
    result = await tester.test_single_problem(
        problem=test_problem.strip(),
        ground_truth="2",
        trace_execution=True
    )
    
    print("\n测试结果:")
    print(f"  问题: {result['problem'][:100]}...")
    print(f"  答案: {result['answer']}")
    print(f"  正确答案: {result['ground_truth']}")
    print(f"  成本: ${result['cost']:.6f}")
    
    if 'execution_trace' in result:
        print(f"  LLM 调用次数: {result['execution_trace'].get('llm_calls', 'N/A')}")


async def main():
    """运行所有示例"""
    print("\n" + "#"*80)
    print("# AFlow 工作流测试示例")
    print("#"*80)
    
    # 运行示例（可以注释掉不需要的）
    
    # 示例 1: 基础测试
    await example_1_basic_test()
    
    # 示例 2: 详细追踪（会打印很多信息）
    # await example_2_detailed_trace()
    
    # 示例 3: 对比轮次
    # await example_3_compare_rounds()
    
    # 示例 4: 单个问题测试
    # await example_4_single_problem()
    
    print("\n" + "#"*80)
    print("# 所有示例完成！")
    print("#"*80 + "\n")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
