#!/bin/bash
# 快速测试脚本

echo "=========================================="
echo "AFlow Workflow Testing"
echo "=========================================="
echo ""

# 检查参数
if [ "$#" -lt 2 ]; then
    echo "Usage: ./test/quick_test.sh <dataset> <round_num> [num_problems]"
    echo ""
    echo "Examples:"
    echo "  ./test/quick_test.sh DROP 8 5"
    echo "  ./test/quick_test.sh MATH 3 10"
    echo ""
    exit 1
fi

DATASET=$1
ROUND=$2
NUM_PROBLEMS=${3:-5}

echo "Dataset: $DATASET"
echo "Round: $ROUND"
echo "Number of problems: $NUM_PROBLEMS"
echo ""
echo "Running detailed trace..."
echo ""

# 运行详细追踪
python test/detailed_tracer.py $DATASET $ROUND $NUM_PROBLEMS

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  test/detailed_trace_${DATASET}_round_${ROUND}.json"
echo ""
echo "To compare with other rounds:"
echo "  python test/detailed_tracer.py compare $DATASET <round1> <round2> $NUM_PROBLEMS"
