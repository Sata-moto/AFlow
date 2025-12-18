import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
from scripts.utils.sanitize import sanitize




class MBPPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str, solved_threshold: float = 0.5):
        super().__init__(name, file_path, log_path, solved_threshold)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func())
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def check_solution(self, solution, test, entry_point):
        solution = sanitize(code=solution, entrypoint=entry_point)
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            exec(test, global_dict)

            check = global_dict["check"]

            result = self.run_with_timeout(check, 15)

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as e:
            error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
            result = (self.FAIL, error_message)

            with open("error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

        return result

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, prompt, entry_point):
        return await graph(prompt, entry_point)

    async def evaluate_problem(self, data: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = data["prompt"]
        
        # 获取problem_id: 优先使用task_id/id字段,否则使用_index生成problem_{idx}格式
        if "task_id" in data:
            problem_id = data["task_id"]
        elif "id" in data:
            problem_id = data["id"]
        elif "_index" in data:
            problem_id = f"problem_{data['_index']}"
        else:
            problem_id = "unknown"
        
        category = self._get_problem_category(problem_id)
        expected_output = "\nCorrect Solution:\ndef " + data["code"]

        try:
            # Generate prediction using the graph function
            prediction, cost = await self._generate_output(graph, input_text, data["entry_point"])

            # Check the solution
            ret = self.check_solution(prediction, data["test"], data["entry_point"])
            test_case_details = ret[1]
            expected_output = test_case_details + "\nCorrect Solution:" + data["code"]

            # Calculate score based on the check result
            score = 1.0 if ret[0] == self.PASS else 0.0
            judge_explanation = test_case_details

            # 所有问题都记录日志（包含类别信息）
            self.log_mismatch(input_text, expected_output, prediction, judge_explanation, score, problem_id, category)

            return input_text, prediction, expected_output, score, cost

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error in evaluate_problem:\n{error_traceback}")
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            self.log_mismatch(input_text, expected_output, str(e), f"Error: {e}", 0.0, problem_id, category)
            return input_text, str(e), expected_output, 0.0, 0.0

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        # The scoring logic for MBPP is already implemented in evaluate_problem, this is just to conform to the interface
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
