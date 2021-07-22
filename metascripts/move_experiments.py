from argparse import ArgumentParser

parser = ArgumentParser(description="Pre-Training")
parser.add_argument("--workspace1", default="")
parser.add_argument("--project1", default="")
parser.add_argument("--workspace2", default="")
parser.add_argument("--project2", default="")
args, _ = parser.parse_known_args()

import comet_ml

comet_api = comet_ml.api.API()

experiments = comet_api.get_experiments(args.workspace1, project_name=args.project1)


for e in experiments:
    try:
        comet_api.move_experiments([e.id], args.workspace2, args.project2, symlink=True)
    except comet_ml.exceptions.CometRestApiException:
        print(
            "POST https://www.comet.ml/api/rest/v2/write/experiment/move failed with status code 400: User not allowed to modify this experiment."
        )

# comet_api.move_experiments(
#     [e.id for e in experiments], args.workspace2, args.project2, symlink=True
# )
