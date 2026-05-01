# WRIVA CVGL Metrics

This repo contains the metrics code utilized by Codabench to run evaluations.

## Installation

Use the `requirements.txt` to install dependencies needed for running locally.

```bash
uv venv --python 3.14
uv pip install -r requirements.txt
```

## Running

Execute the metrics by passing in the folder containing both the `ref/` and `res/` folders, containing the reference dataset and results submission respectively. Additionally, pass in an output folder. This will write the scores that are displayed on the leaderboard to a `scores.txt`:

```bash
uv run python evaluate.py /path/to/res/and/ref/ /path/to/output/
```

## Expected Submission Format

To create your zip file that is uploaded to Codabench, please organize your folders into a specific format, otherwise the scoring program will error. The expected format is:

```
submission/
├─ WRIVA-CVGL-DEV-001/
│  ├─ *.json
│  ├─ ...
├─ WRIVA-CVGL-DEV-002/
│  ├─ *.json
│  ├─ ...
├─ WRIVA-CVGL-DEV-003/
│  ├─ *.json
│  ├─ ...
```

- The different folders within submission (ie, `WRIVA-CVGL-DEV-001/`) represent each dataset you will submit for. 
- These folders will contain your JSON files for each dataset. These JSON files must match the requested names exactly in order to properly correlate reference to submission.
- **When submitting to Codabench, do NOT zip a high level folder containing each dataset. Zip each dataset folder individually. For example:**
  - **CORRECT**: `zip -r submission.zip WRIVA-CVGL-DEV-001/ WRIVA-CVGL-DEV-002/ WRIVA-CVGL-DEV-003/`
  - **INCORRECT**: `zip -r submission.zip submission/` 

## Expected JSON format

This section discusses the expected format for each submission JSON. Your submissions must contain, *at minimum* these keys and values. You are welcome to put other information within the JSONs. Names for the submission JSONs must match the reference names requested exactly.

Structure for the submission JSONs:

```json
{
  "lat": 39.16236957932923,
  "lon":  -76.89239703213026,
  "heading": 0.0,
  "pitch": 0.0
}
```

- `lat`: predicted latitude of the frame 
- `lon`: predicted longitude of the frame
- `heading`: Optional, describes the heading of the image. See the [Evaluation](#evaluation) section for further information.
- `pitch`: Optional, describes the pitch of the image. See the [Evaluation](#evaluation) section for further information.


## Evaluation

At minimum, we evaluate and display on the leaderboard the 90th percentile RMSE for latitude/longitude geolocation error for each dataset. 

You can additionally provide us a predicted heading and pitch for each frame, though this is optional. If specified, we compute the 90th percentile RMSE for heading orientation. This is technically a part of the leaderboard, but hidden and not necessarily evaluated by us.


### Generating required Codabench competition structure

If looking to upload a compeition with this content to Codabench, the files and folders in here need to be formatted very specifically. I use a Makefile to do this. Assuming you have a folder of the reference data named `WRIVA-CVGL-DEV-ONLY-REFERENCE.zip`, simply run

```bash
make
```

Which will create a `wriva-cvgl-codabench-competition.zip` file that can be uploaded right to Codabench competition creation. If you need to re-make the submission zip, you can run

```bash
make clean
make
```

to re-create it.

