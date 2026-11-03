# VLM question loop

The VLM check of the batch pipeline (`scripts/vlm_check.py`) asks a local Qwen2.5-VL model yes/no questions
about a few frames of a generated video; a video with any Yes is marked bad. This folder finds those
questions automatically: a question writer (Claude, through the Claude Code CLI) looks at frame sheets,
the labels of a set of videos and the checker's last results, and writes new questions. The checker runs
them on the set, the score goes back to the writer, and so on, until the score stops going up.

Goal of the questions: only good videos are used to train a robot policy. A video is bad when, over the
whole video, a problem is big enough to hurt robot learning; small problems are fine. The worst mistake is a
bad video that passes as good.

## Files

| file | what it does | where it runs |
|---|---|---|
| `make_set.py` | set folder: links to the chosen demo folders, and `labels.csv` from a verdict CSV | GUI machine |
| `make_sheets.py` | frame sheets for the writer: `frames11/` (11 moments, full frame), `crops11/` (object close-ups) | GUI machine |
| `run_batch.py` | one VLM-only batch through the GUI server (like its Start button); waits while no allowed GPU has room, never runs the checker on the CPU | GUI machine |
| `score.py` | the batch against the labels: `score.json`, `pyes.csv`; with `--calibrate` the thresholds are set from the batch (`--calibrate max_right`: see score.py) | GUI machine |
| `prompts/writer_vN.txt` | the prompt of the writer, filled each round; one file per version (see below) | - |
| `loop.py` | the whole loop; only the Python standard library | machine with the `claude` command |
| `config.example.json` | all settings of one run | - |

The loop runs on the machine with the Claude Code CLI and reaches the GUI machine with the `remote.prefix`
command of the config (for example `ssh user@host docker exec -i container`), or runs there directly with an
empty prefix.

## Run

1. Make the set (on the GUI machine):

       python scripts/vlm_loop/make_set.py --demos_dir DEMOS --demos 0-49 --verdicts VERDICTS.csv \
           --video_col demo --verdict_col verdict --good approved,weak --bad rejected --out SET_DIR

2. Copy `config.example.json`, fill in the paths, and start the loop (on the machine with `claude`):

       python scripts/vlm_loop/loop.py CONFIG.json

   The same command goes on after a stop: rounds with a score are not run again.
3. Stop it with a file `STOP` in the run folder, or with SIGTERM; it ends after the current step.

To test the questions on other videos, make a second set with `make_set.py` (other demos, same verdict CSV)
and add it to the config as `"test": {"set_dir": ..., "labels": ..., "batch": ...}` (see below).
`make_set.py` can pick the demos itself: `--first_good 25 --first_bad 25` takes the first 25 good and 25 bad
demos in order (fewer when there are not enough, with a message), and `--exclude_set TRAIN_SET` leaves out the
demos of the train set when making the test set.

## A round

- Round 0 runs `round0_questions` (default `scripts/vlm_questions.json`).
- Round r >= 1:
  1. `round<r>/prompt.txt`: the writer prompt (`prompt_template` of the config) filled with the earlier rounds
     and their scores.
  2. The writer runs as `claude -p` in the run folder. It may only read (Read, Glob, Grep) inside the run folder; Bash, Edit, Write and web tools are off. Its report goes to `round<r>/writer_report.md`, every tool call to `round<r>/writer_stream.jsonl` and the files it opened to `round<r>/files_read.json`. A file outside the run folder stops the loop.
  3. The questions come from the ```json block of the report: `round<r>/questions.json`.
  4. `run_batch.py` runs them on the set (GUI batch `<batch_prefix><r>`, visible in the Batch dialog) and `score.py` scores them. With `calibrate` (default) every threshold is set from the batch: halfway between the highest P(yes) of a good video and the next higher P(yes) of a bad video, so no good video gets Yes (`questions_calibrated.json`, `score_as_run.json` keeps the score with the thresholds as written). With `"calibrate": "max_right"` the thresholds are then lowered where this gives more right videos (see score.py). Good videos can then get Yes. The writer prompts v1 to v8 describe only the first rule. The Y/N in the GUI table use the thresholds as written.
  5. `score.json`, `pyes.csv` and the frames the checker saw (`qwen_frames/`) are copied into `round<r>/` for the next writer.

## Writer prompt versions

`prompts/writer_vN.txt`; a changed prompt is a new file, an old file is never edited. The runs are the run
folders whose `round1/prompt.txt` is that version filled in.

| version | commit | change | runs |
|---|---|---|---|
| v1 | 0b0eff7 | first prompt | yam_v1, yam_v3 |
| v2 | f12d01e | v1 + thresholds set from the batch, one question that compares the two views | yam_v2 |
| v3 | 3097fc5 | v1 + at least one question that compares the two views | yam_v4 |
| v4 | 48228bb | v3 + the thresholds are set after the run, not by the writer | yam_v5, yam_v6, yam_v7, mix_v1 to mix_v4 |
| v5 | 61e640e | v4 without the general-question rule; the checker is named "Qwen" | yam_v8 |
| v6 | bcd9a46 | v4 with the checker named "Qwen" | - |
| v7 | 9b4bb07 | v5 + the labels do not judge the color of the robot | - |
| v8 | - | v6 + the labels do not judge the color of the robot | - |

## Stop and pick

- Stop: `patience` rounds in a row without more correct videos (a tie is no gain), or round `max_rounds`.
- Pick: the round with the most correct videos (with the set thresholds); tie -> fewer bad videos passed as good; tie -> the larger
  margin (the smallest distance between a threshold and any video's P(yes), see `score.py`); tie -> the earlier
  round.
- `summary.md` and `summary.json` (with the picked questions) in the run folder.

## Test (optional)

With `test` in the config, after the last round the picked questions run once on the test set, with the
thresholds set on the loop's set, and are scored without changing any threshold (`score.py` without
`--calibrate`): `test/questions.json`, `test/score.json`, `test/pyes.csv`, and a "Test:" line in `summary.md`.
The writer never sees the test: it runs after the last round, and no round is written once `test/` exists.
It is not run when the loop stopped on an error; running the same command again runs only the test.

## Status files (for a GUI panel)

- `config.json`: the config of the run, saved at the start. A run folder with another saved config is not run.
- `loop_status.json`: the current step, one row per finished round (right, bad passed, good flagged, margin,
  question ids), the stop reason and the pick when finished, and the test result. Written after every step.
- `loop.log`: the text log.
