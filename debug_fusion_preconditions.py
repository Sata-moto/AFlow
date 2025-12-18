#!/usr/bin/env python3
"""
调试融合前置条件检查
"""
import json
import os
import sys

# 模拟 DataUtils.find_envelope_workflows 的逻辑
def debug_envelope_workflows(root_path, max_workflows=3):
    results_file = os.path.join(root_path, "workflows", "results.json")
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"📊 Loaded {len(results)} result entries")
    
    # Group by round
    rounds_data = {}
    for result in results:
        round_num = result["round"]
        if round_num not in rounds_data:
            rounds_data[round_num] = {
                "round": round_num,
                "scores": [],
                "solved_problems": set()
            }
        
        rounds_data[round_num]["scores"].append(result["score"])
        
        # Union solved problems
        if "solved_problems" in result:
            if isinstance(result["solved_problems"], list):
                rounds_data[round_num]["solved_problems"].update(result["solved_problems"])
            elif isinstance(result["solved_problems"], set):
                rounds_data[round_num]["solved_problems"].update(result["solved_problems"])
    
    print(f"\n📁 Found {len(rounds_data)} rounds:")
    for round_num in sorted(rounds_data.keys()):
        data = rounds_data[round_num]
        avg_score = sum(data["scores"]) / len(data["scores"])
        print(f"  Round {round_num}: score={avg_score:.4f}, solved={len(data['solved_problems'])} problems")
    
    # Calculate average metrics for each round
    round_summaries = []
    for round_num, data in rounds_data.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        
        round_summaries.append({
            "round": round_num,
            "avg_score": avg_score,
            "solved_problems": data["solved_problems"]
        })
    
    # Greedy algorithm to find envelope workflows
    print(f"\n🔍 Running greedy envelope selection (max={max_workflows}):")
    envelope_workflows = []
    covered_problems = set()
    available_workflows = round_summaries.copy()
    
    for iteration in range(min(max_workflows, len(available_workflows))):
        if not available_workflows:
            print(f"  Iteration {iteration+1}: No available workflows left")
            break
        
        # Find workflow that covers the most uncovered problems
        best_workflow = None
        best_new_coverage = 0
        best_index = -1
        
        for i, workflow in enumerate(available_workflows):
            new_problems = workflow["solved_problems"] - covered_problems
            new_coverage = len(new_problems)
            
            print(f"    Round {workflow['round']}: {len(workflow['solved_problems'])} total, {new_coverage} new")
            
            # If tie, prefer workflow with higher score
            if (new_coverage > best_new_coverage or 
                (new_coverage == best_new_coverage and 
                 (best_workflow is None or workflow["avg_score"] > best_workflow["avg_score"]))):
                best_workflow = workflow
                best_new_coverage = new_coverage
                best_index = i
        
        # If no new coverage, break
        if best_new_coverage == 0:
            print(f"  Iteration {iteration+1}: ❌ No new coverage, stopping")
            break
        
        # Add best workflow to envelope
        print(f"  Iteration {iteration+1}: ✅ Selected Round {best_workflow['round']} (+{best_new_coverage} problems)")
        envelope_workflows.append(best_workflow)
        covered_problems.update(best_workflow["solved_problems"])
        available_workflows.pop(best_index)
    
    print(f"\n✅ Envelope workflows: {len(envelope_workflows)}")
    print(f"📈 Total coverage: {len(covered_problems)} problems")
    for workflow in envelope_workflows:
        print(f"  Round {workflow['round']}: score={workflow['avg_score']:.4f}, problems={len(workflow['solved_problems'])}")
    
    return envelope_workflows

if __name__ == "__main__":
    workspace = "/home/wx/AFlow/workspace/MBPP"
    
    print("=" * 80)
    print("融合前置条件调试")
    print("=" * 80)
    
    envelope = debug_envelope_workflows(workspace, max_workflows=3)
    
    print("\n" + "=" * 80)
    print(f"结论: 找到 {len(envelope)} 个 envelope workflows (需要至少 3 个)")
    if len(envelope) < 3:
        print("❌ 不满足融合条件")
    else:
        print("✅ 满足融合条件")
    print("=" * 80)
