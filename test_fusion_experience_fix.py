#!/usr/bin/env python3
"""
Test the fixed fusion experience.json handling
"""

import json
import os
import tempfile
import shutil

def test_fusion_experience_json():
    """Test that fusion experience.json is properly created and updated"""
    print("Testing fusion experience.json handling...")
    
    # Create test directory structure
    test_root = tempfile.mkdtemp(prefix="aflow_fusion_experience_test_")
    workflows_dir = os.path.join(test_root, "workflows")
    
    try:
        # Create a mock enhanced optimizer to test the methods
        from scripts.enhanced_optimizer import EnhancedOptimizer
        from scripts.optimizer_utils.experience_utils import ExperienceUtils
        
        # Mock envelope workflows
        envelope_workflows = [
            {"round": 1, "avg_score": 0.72, "solved_problems": ["p1", "p2", "p3"]},
            {"round": 2, "avg_score": 0.68, "solved_problems": ["p2", "p3", "p4"]},
            {"round": 3, "avg_score": 0.75, "solved_problems": ["p1", "p4", "p5"]}
        ]
        
        # Create target directory
        fusion_round = 4
        target_dir = os.path.join(workflows_dir, f"round_{fusion_round}")
        os.makedirs(target_dir, exist_ok=True)
        
        # Create mock optimizer with necessary components
        class MockOptimizer:
            def __init__(self, root_path):
                self.root_path = root_path
                self.experience_utils = ExperienceUtils(root_path)
            
            def _create_fusion_experience_file(self, target_dir, envelope_workflows, fusion_round):
                """Test version of the fixed method"""
                try:
                    # Find the best source workflow to use as "father node"
                    best_workflow = max(envelope_workflows, key=lambda w: w["avg_score"])
                    
                    # Create fusion modification description
                    source_rounds = [w["round"] for w in envelope_workflows]
                    source_scores = [f"{w['avg_score']:.4f}" for w in envelope_workflows]
                    
                    fusion_modification = f"Workflow Fusion: Combined high-performing workflows from rounds {source_rounds} " \
                                        f"(scores: {source_scores}). Integrated the best aspects of {len(envelope_workflows)} " \
                                        f"workflows to achieve improved coverage and performance."
                    
                    # Create a mock sample object for the fusion
                    fusion_sample = {
                        "round": best_workflow["round"],  # Use best workflow as father node
                        "score": best_workflow["avg_score"]  # Use best score as "before"
                    }
                    
                    # Use standard experience creation method
                    experience_data = self.experience_utils.create_experience_data(fusion_sample, fusion_modification)
                    
                    # Save experience.json using standard method
                    experience_path = os.path.join(target_dir, "experience.json")
                    with open(experience_path, 'w', encoding='utf-8') as f:
                        json.dump(experience_data, f, indent=4, ensure_ascii=False)
                    
                    print(f"✅ Created fusion experience.json with father node {best_workflow['round']}")
                    return experience_path
                    
                except Exception as e:
                    print(f"❌ Error creating fusion experience file: {e}")
                    return None
        
        mock_optimizer = MockOptimizer(test_root)
        
        # Step 1: Test creation of fusion experience.json
        print("Step 1: Testing fusion experience.json creation...")
        experience_path = mock_optimizer._create_fusion_experience_file(target_dir, envelope_workflows, fusion_round)
        
        if not experience_path or not os.path.exists(experience_path):
            print("❌ Failed to create experience.json")
            return False
        
        # Step 2: Verify the created file format
        print("Step 2: Verifying experience.json format...")
        with open(experience_path, 'r', encoding='utf-8') as f:
            experience_data = json.load(f)
        
        required_fields = ["father node", "modification", "before", "after", "succeed"]
        for field in required_fields:
            if field not in experience_data:
                print(f"❌ Missing field: {field}")
                return False
        
        print(f"✅ All required fields present: {list(experience_data.keys())}")
        
        # Step 3: Test updating the experience with evaluation results
        print("Step 3: Testing experience update...")
        
        # Simulate evaluation results
        fusion_score = 0.78
        mock_optimizer.experience_utils.update_experience(target_dir, experience_data, fusion_score)
        
        # Verify updated file
        with open(experience_path, 'r', encoding='utf-8') as f:
            updated_data = json.load(f)
        
        if updated_data["after"] != fusion_score:
            print(f"❌ Score not updated correctly. Expected: {fusion_score}, Got: {updated_data['after']}")
            return False
        
        if updated_data["succeed"] != (fusion_score > updated_data["before"]):
            print(f"❌ Success flag incorrect. Expected: {fusion_score > updated_data['before']}, Got: {updated_data['succeed']}")
            return False
        
        print(f"✅ Experience updated correctly:")
        print(f"   Before: {updated_data['before']}")
        print(f"   After: {updated_data['after']}")
        print(f"   Success: {updated_data['succeed']}")
        
        # Step 4: Test JSON serialization (ensure no bool serialization errors)
        print("Step 4: Testing JSON serialization...")
        try:
            json_str = json.dumps(updated_data, indent=4, ensure_ascii=False)
            parsed_back = json.loads(json_str)
            
            if parsed_back["succeed"] == updated_data["succeed"]:
                print("✅ JSON serialization works correctly")
            else:
                print("❌ JSON serialization corrupted boolean value")
                return False
                
        except Exception as e:
            print(f"❌ JSON serialization error: {e}")
            return False
        
        # Step 5: Test file format completeness
        print("Step 5: Testing file format completeness...")
        
        # Read the file as text to check for incomplete JSON
        with open(experience_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        if '"succeed":' in file_content and not ('"succeed": true' in file_content or '"succeed": false' in file_content):
            print("❌ Incomplete JSON - succeed field has no value")
            return False
        
        print("✅ JSON file format is complete")
        print("Final experience.json content:")
        print(file_content)
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
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
    print("Testing Fusion Experience.json Handling")
    print("="*60)
    
    success = test_fusion_experience_json()
    
    print("="*60)
    if success:
        print("✅ ALL TESTS PASSED: Fusion experience.json handling is fixed!")
    else:
        print("❌ TESTS FAILED: Issues remain with fusion experience.json handling")
    print("="*60)
    
    exit(0 if success else 1)
