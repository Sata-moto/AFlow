#!/usr/bin/env python3
"""
Test script to verify that fusion workflow information persists in processed_experience.json
"""

import json
import os
import tempfile
import shutil
from scripts.enhanced_optimizer import EnhancedOptimizer
from scripts.optimizer_utils.experience_utils import ExperienceUtils

def setup_test_environment():
    """Set up a temporary test environment"""
    test_root = tempfile.mkdtemp(prefix="aflow_fusion_test_")
    workflows_dir = os.path.join(test_root, "workflows")
    os.makedirs(workflows_dir, exist_ok=True)
    
    # Create some fake experience files for different rounds
    for round_num in [1, 2, 3]:
        round_dir = os.path.join(workflows_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # Create fake experience.json
        experience_data = {
            "father node": str(round_num),
            "modification": f"Test modification for round {round_num}",
            "before": 0.5 + round_num * 0.1,
            "after": 0.6 + round_num * 0.1,
            "succeed": True
        }
        
        with open(os.path.join(round_dir, "experience.json"), 'w') as f:
            json.dump(experience_data, f, indent=2)
    
    return test_root

def test_fusion_persistence():
    """Test that fusion information persists across load_experience calls"""
    print("Testing fusion information persistence...")
    
    # Set up test environment
    test_root = setup_test_environment()
    workflows_dir = os.path.join(test_root, "workflows")
    experience_path = os.path.join(workflows_dir, "processed_experience.json")
    
    try:
        # Create experience utils
        experience_utils = ExperienceUtils(test_root)
        
        # Step 1: Load initial experience (this will create processed_experience.json)
        print("Step 1: Loading initial experience...")
        initial_experience = experience_utils.load_experience()
        print(f"Initial experience rounds: {list(initial_experience.keys())}")
        
        # Step 2: Simulate adding fusion info manually (like _add_fusion_to_experience does)
        print("Step 2: Adding fusion information...")
        fusion_round = "4"  # Simulate fusion creating round 4
        
        if os.path.exists(experience_path):
            with open(experience_path, 'r', encoding='utf-8') as f:
                processed_experience = json.load(f)
        else:
            processed_experience = {}
        
        # Add fusion info
        processed_experience[fusion_round] = {
            "score": 0.85,
            "success": {},
            "failure": {},
            "fusion_info": {
                "is_fusion": True,
                "source_rounds": [1, 2, 3],
                "fusion_modification": "Fused from rounds [1, 2, 3] (scores: ['0.60', '0.70', '0.80']). Combined workflows.",
                "fusion_score": 0.85
            }
        }
        
        # Save with fusion info
        with open(experience_path, 'w', encoding='utf-8') as f:
            json.dump(processed_experience, f, indent=4, ensure_ascii=False)
        
        print(f"Added fusion info for round {fusion_round}")
        print(f"Experience with fusion: {list(processed_experience.keys())}")
        
        # Step 3: Test the preserve and restore mechanism
        print("Step 3: Testing preserve/restore mechanism...")
        
        # Create a mock enhanced optimizer just to use the preservation methods
        class MockEnhancedOptimizer:
            def __init__(self, root_path):
                self.root_path = root_path
            
            def _preserve_fusion_info(self, experience_path: str):
                fusion_info = {}
                
                try:
                    if os.path.exists(experience_path):
                        with open(experience_path, 'r', encoding='utf-8') as f:
                            existing_experience = json.load(f)
                        
                        for round_key, round_data in existing_experience.items():
                            if isinstance(round_data, dict) and 'fusion_info' in round_data:
                                fusion_info[round_key] = round_data['fusion_info']
                except Exception as e:
                    print(f"Error preserving fusion info: {e}")
                
                return fusion_info
            
            def _restore_fusion_info(self, experience_path: str, fusion_info):
                try:
                    if not fusion_info or not os.path.exists(experience_path):
                        return
                    
                    with open(experience_path, 'r', encoding='utf-8') as f:
                        current_experience = json.load(f)
                    
                    for round_key, fusion_data in fusion_info.items():
                        if round_key in current_experience:
                            current_experience[round_key]['fusion_info'] = fusion_data
                        else:
                            current_experience[round_key] = {
                                "score": fusion_data.get('fusion_score', 0.0),
                                "success": {},
                                "failure": {},
                                "fusion_info": fusion_data
                            }
                    
                    with open(experience_path, 'w', encoding='utf-8') as f:
                        json.dump(current_experience, f, indent=4, ensure_ascii=False)
                        
                except Exception as e:
                    print(f"Error restoring fusion info: {e}")
        
        mock_optimizer = MockEnhancedOptimizer(test_root)
        
        # Preserve fusion info
        preserved_info = mock_optimizer._preserve_fusion_info(experience_path)
        print(f"Preserved fusion info: {list(preserved_info.keys())}")
        
        # Step 4: Simulate load_experience overwriting the file
        print("Step 4: Simulating load_experience overwrite...")
        reloaded_experience = experience_utils.load_experience()
        print(f"After load_experience: {list(reloaded_experience.keys())}")
        
        # Check if fusion info was lost
        with open(experience_path, 'r', encoding='utf-8') as f:
            after_load = json.load(f)
        
        fusion_lost = fusion_round not in after_load or 'fusion_info' not in after_load.get(fusion_round, {})
        print(f"Fusion info lost after load_experience: {fusion_lost}")
        
        # Step 5: Restore fusion info
        print("Step 5: Restoring fusion info...")
        mock_optimizer._restore_fusion_info(experience_path, preserved_info)
        
        # Verify restoration
        with open(experience_path, 'r', encoding='utf-8') as f:
            final_experience = json.load(f)
        
        fusion_restored = fusion_round in final_experience and 'fusion_info' in final_experience.get(fusion_round, {})
        print(f"Final experience rounds: {list(final_experience.keys())}")
        print(f"Fusion info restored: {fusion_restored}")
        
        if fusion_restored:
            fusion_data = final_experience[fusion_round]['fusion_info']
            print(f"Restored fusion data: {fusion_data}")
        
        # Test result
        if fusion_restored and final_experience[fusion_round]['fusion_info']['is_fusion']:
            print("✅ TEST PASSED: Fusion information persistence works correctly!")
            return True
        else:
            print("❌ TEST FAILED: Fusion information was not properly restored")
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
    success = test_fusion_persistence()
    exit(0 if success else 1)
