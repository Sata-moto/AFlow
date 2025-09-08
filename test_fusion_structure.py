#!/usr/bin/env python3
"""
Test script to verify that fusion workflows follow the standard workflow structure
"""

import json
import os
import tempfile
import shutil
from scripts.enhanced_optimizer import EnhancedOptimizer
from scripts.optimizer_utils.experience_utils import ExperienceUtils

def create_mock_fusion_scenario():
    """Create a mock scenario to test fusion workflow structure"""
    test_root = tempfile.mkdtemp(prefix="aflow_fusion_structure_test_")
    workflows_dir = os.path.join(test_root, "workflows")
    os.makedirs(workflows_dir, exist_ok=True)
    
    # Create mock envelope workflows (rounds 1, 2, 3)
    envelope_workflows = []
    for round_num in [1, 2, 3]:
        round_dir = os.path.join(workflows_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # Create basic structure
        with open(os.path.join(round_dir, "__init__.py"), 'w') as f:
            f.write("")
        
        with open(os.path.join(round_dir, "graph.py"), 'w') as f:
            f.write(f"# Graph for round {round_num}\nclass Graph:\n    pass")
        
        with open(os.path.join(round_dir, "prompt.py"), 'w') as f:
            f.write(f"# Prompt for round {round_num}\nPROMPT = 'Test prompt {round_num}'")
        
        # Create experience.json
        experience_data = {
            "father node": max(1, round_num - 1),
            "modification": f"Test modification for round {round_num}",
            "before": 0.5 + (round_num - 1) * 0.1,
            "after": 0.5 + round_num * 0.1,
            "succeed": True
        }
        
        with open(os.path.join(round_dir, "experience.json"), 'w') as f:
            json.dump(experience_data, f, indent=2)
        
        # Create log.json
        log_data = [
            {"problem_id": f"test_{round_num}_1", "error": None},
            {"problem_id": f"test_{round_num}_2", "error": "sample error"}
        ]
        
        with open(os.path.join(round_dir, "log.json"), 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Create envelope workflow data
        envelope_workflows.append({
            "round": round_num,
            "avg_score": 0.5 + round_num * 0.1,
            "solved_problems": [f"prob_{round_num}_{i}" for i in range(5)]
        })
    
    return test_root, envelope_workflows

def test_fusion_workflow_structure():
    """Test that fusion workflows create proper structure"""
    print("Testing fusion workflow structure...")
    
    test_root, envelope_workflows = create_mock_fusion_scenario()
    
    try:
        # Create a mock enhanced optimizer instance
        class MockEnhancedOptimizer:
            def __init__(self, root_path):
                self.root_path = root_path
                self.round = 3  # Simulate being at round 3
                self.fusion_metadata_counter = 0
            
            def _create_fusion_experience_file(self, target_dir, envelope_workflows, fusion_round):
                """Copy from enhanced_optimizer.py"""
                try:
                    import time
                    
                    best_workflow = max(envelope_workflows, key=lambda w: w["avg_score"])
                    father_node = best_workflow["round"]
                    
                    source_rounds = [w["round"] for w in envelope_workflows]
                    source_scores = [f"{w['avg_score']:.4f}" for w in envelope_workflows]
                    
                    fusion_modification = f"Workflow Fusion: Combined high-performing workflows from rounds {source_rounds} " \
                                        f"(scores: {source_scores}). Integrated the best aspects of {len(envelope_workflows)} " \
                                        f"workflows to achieve improved coverage and performance."
                    
                    experience_data = {
                        "father node": father_node,
                        "modification": fusion_modification,
                        "before": best_workflow["avg_score"],
                        "after": 0.90,  # Simulate successful fusion
                        "succeed": True  # Simulate successful fusion
                    }
                    
                    experience_path = os.path.join(target_dir, "experience.json")
                    with open(experience_path, 'w', encoding='utf-8') as f:
                        json.dump(experience_data, f, indent=4, ensure_ascii=False)
                    
                    print(f"✅ Created fusion experience.json with father node {father_node}")
                    
                except Exception as e:
                    print(f"❌ Error creating fusion experience file: {e}")
            
            def _create_fusion_log_file(self, target_dir, envelope_workflows):
                """Copy from enhanced_optimizer.py"""
                try:
                    import time
                    
                    log_data = {
                        "fusion_metadata": {
                            "timestamp": time.time(),
                            "fusion_type": "envelope_fusion",
                            "source_workflows": [
                                {
                                    "round": w["round"],
                                    "score": w["avg_score"],
                                    "solved_problems": len(w["solved_problems"])
                                }
                                for w in envelope_workflows
                            ],
                            "total_unique_problems": len(set().union(*[set(w["solved_problems"]) for w in envelope_workflows])),
                            "fusion_strategy": "Combined best aspects of envelope workflows using LLM-guided fusion"
                        },
                        "execution_logs": []
                    }
                    
                    log_path = os.path.join(target_dir, "log.json")
                    with open(log_path, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=4, ensure_ascii=False)
                    
                    print(f"✅ Created fusion log.json with metadata for {len(envelope_workflows)} source workflows")
                    
                except Exception as e:
                    print(f"❌ Error creating fusion log file: {e}")
        
        mock_optimizer = MockEnhancedOptimizer(test_root)
        
        # Step 1: Create fusion round directory
        fusion_round = mock_optimizer.round + 1  # Should be round 4
        target_dir = os.path.join(test_root, "workflows", f"round_{fusion_round}")
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Step 1: Created fusion round directory: round_{fusion_round}")
        
        # Step 2: Create standard files
        with open(os.path.join(target_dir, "__init__.py"), 'w') as f:
            f.write("")
        
        with open(os.path.join(target_dir, "graph.py"), 'w') as f:
            f.write("# Fused workflow graph\nclass FusedGraph:\n    pass")
        
        with open(os.path.join(target_dir, "prompt.py"), 'w') as f:
            f.write("# Fused workflow prompt\nPROMPT = 'Fused workflow prompt'")
        
        print("Step 2: Created basic workflow files")
        
        # Step 3: Create fusion-specific experience.json and log.json
        mock_optimizer._create_fusion_experience_file(target_dir, envelope_workflows, fusion_round)
        mock_optimizer._create_fusion_log_file(target_dir, envelope_workflows)
        
        print("Step 3: Created fusion-specific files")
        
        # Step 4: Test that load_experience can process the fusion workflow
        experience_utils = ExperienceUtils(test_root)
        processed_experience = experience_utils.load_experience()
        
        print(f"Step 4: load_experience processed rounds: {list(processed_experience.keys())}")
        
        # Debug: print the processed experience content
        print("Processed experience content:")
        for key, value in processed_experience.items():
            print(f"  Father node {key}:")
            print(f"    Score: {value['score']}")
            print(f"    Success: {list(value['success'].keys()) if value['success'] else []}")
            print(f"    Failure: {list(value['failure'].keys()) if value['failure'] else []}")
            if value['success']:
                for round_num, data in value['success'].items():
                    modification = data.get('modification', '')
                    print(f"      Round {round_num}: {modification[:100]}...")
        
        print(f"Step 4: load_experience processed rounds: {list(processed_experience.keys())}")
        
        # Step 5: Verify fusion workflow is included (check if fusion modification exists in any father node)
        fusion_found = False
        for father_node, node_data in processed_experience.items():
            # Check success entries
            for round_num, round_data in node_data.get('success', {}).items():
                if 'Workflow Fusion' in round_data.get('modification', ''):
                    print(f"✅ Fusion workflow found in father node {father_node}, round {round_num}")
                    fusion_found = True
                    break
            # Check failure entries
            for round_num, round_data in node_data.get('failure', {}).items():
                if 'Workflow Fusion' in round_data.get('modification', ''):
                    print(f"✅ Fusion workflow found in father node {father_node}, round {round_num}")
                    fusion_found = True
                    break
        
        if fusion_found:
            print("✅ Fusion workflow successfully integrated into processed experience!")
            return True
        else:
            print("❌ Fusion workflow not found in processed experience")
            return False
            
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(test_root):
            shutil.rmtree(test_root)
        print(f"Cleaned up test directory: {test_root}")

if __name__ == "__main__":
    print("="*60)
    print("Testing Fusion Workflow Standard Structure")
    print("="*60)
    
    success = test_fusion_workflow_structure()
    
    print("="*60)
    if success:
        print("✅ ALL TESTS PASSED: Fusion workflows follow standard structure!")
    else:
        print("❌ TESTS FAILED: Fusion workflows don't follow standard structure")
    print("="*60)
    
    exit(0 if success else 1)
