from limin_bench.base import LikertEvaluationRun


evaluation_run = LikertEvaluationRun.from_json_file("evaluation_run.json")

print(evaluation_run.instability)
