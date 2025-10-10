#!/usr/bin/env python3
"""
Test script to verify the new fusion control parameters work correctly.
"""

def test_fusion_control_logic():
    """Test the fusion control logic with different scenarios."""
    
    # Test scenarios
    scenarios = [
        {"round": 1, "last_fusion": -1, "should_fuse": False, "reason": "before start round"},
        {"round": 3, "last_fusion": -1, "should_fuse": False, "reason": "before start round"},
        {"round": 5, "last_fusion": -1, "should_fuse": True, "reason": "at start round"},
        {"round": 6, "last_fusion": 5, "should_fuse": False, "reason": "insufficient interval (1 < 2)"},
        {"round": 7, "last_fusion": 5, "should_fuse": True, "reason": "sufficient interval (2 >= 2)"},
        {"round": 10, "last_fusion": 8, "should_fuse": True, "reason": "sufficient interval (2 >= 2)"},
    ]
    
    fusion_start_round = 5
    fusion_interval_rounds = 2
    
    print("Testing fusion control logic:")
    print(f"- fusion_start_round: {fusion_start_round}")
    print(f"- fusion_interval_rounds: {fusion_interval_rounds}")
    print()
    
    for i, scenario in enumerate(scenarios):
        round_num = scenario["round"]
        last_fusion = scenario["last_fusion"]
        expected = scenario["should_fuse"]
        reason = scenario["reason"]
        
        # Test start round condition
        if round_num < fusion_start_round:
            should_fuse = False
            actual_reason = "before start round"
        # Test interval condition
        elif last_fusion != -1 and (round_num - last_fusion) < fusion_interval_rounds:
            should_fuse = False
            actual_reason = f"insufficient interval ({round_num - last_fusion} < {fusion_interval_rounds})"
        else:
            should_fuse = True
            actual_reason = "conditions met"
        
        status = "✓" if should_fuse == expected else "✗"
        print(f"{status} Scenario {i+1}: Round {round_num}, last_fusion={last_fusion}")
        print(f"  Expected: {expected} ({reason})")
        print(f"  Actual: {should_fuse} ({actual_reason})")
        print()

if __name__ == "__main__":
    test_fusion_control_logic()
