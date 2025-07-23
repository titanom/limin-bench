from limin_bench.base import LikertEvaluationRun


evaluation_run = LikertEvaluationRun.from_json_file("evaluation_run.json")

for row in evaluation_run.rows:
    print(row.instability)

print(evaluation_run.instability(method="fus"))
