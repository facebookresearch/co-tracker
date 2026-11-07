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
| `make_set.py` | one set (train, test, all, ...) as rows of a sets CSV (video, label, set), from the review in a share folder; no folder or link | machine with the videos |
| `make_sheets.py` | frame sheets for the writer: `frames11/` (11 moments, full frame), `crops11/` (object close-ups), from the videos of a set in the share folder | machine with the videos |
| `run_batch.py` | one VLM-only batch on the videos of a set, read straight from the share folder (the VLM step of `scripts/batch_pipeline.py`, used as it is; not through the GUI); waits while no allowed GPU has room, never runs the checker on the CPU | machine with the videos |
| `score.py` | the batch against the labels: `score.json`, `pyes.csv`; with `--calibrate` the thresholds are set from the batch (`--calibrate max_right`: see score.py) | machine with the videos |
| `prompts/writer_vN.txt` | the prompt of the writer, filled each round; one file per version (see below) | - |
| `loop.py` | the whole loop: makes the sets, writes and runs the questions, scores test and other; only the Python standard library | machine with the `claude` command |
| `configs/*.json` | ready settings of a run, one file per kind of run (see Run), used as they are | - |
| `setup_ssh.sh` | ssh client and a user with the uid/gid of the owner of `/workspace`, to run the loop in a container (the key goes in `/workspace/.ssh`; no key is made) | inside that container, as root |
| `results/vlm_v1.0/` | the reported results: review, sets and P(yes) of each task (`<task>_<review, sets or pyes>_vlm_v1.0.csv`; see Results) | - |
| `results/vlm_v0.0/` | the first review list of the Franka share folder (2026-10-12; see its README) | - |

The loop runs on the machine with the Claude Code CLI and reaches the machine with the videos (the share folder)
with the `remote.prefix` command (for example `ssh user@host docker exec -i container`), or runs there directly
with an empty prefix. The prefix and the path of the `claude` command belong to one machine, so the configs do not
have them. Put them in `configs/local.json` (not in git). The loop adds them to the config:

    {"remote": {"prefix": ["ssh", "user@host", "docker", "exec", "-i", "container"]},
     "writer": {"cli": "/path/to/claude"}}

Without `local.json`, the loop runs on the machine with the videos and calls `claude`. The other paths in the
configs (`remote.repo`, `remote.python`, `remote.work_dir`, `share`) are paths on the machine with the videos.
Change them when they are different there.

The share folder is made by `share.py` of the SoftMimicGen repo (not in this repo). It has the videos in
`videos/<O|weak|X|not_reviewed>/demo_NNN/`. The folder of a demo is its verdict: O = approved, weak = weak,
X = rejected. In `lists/`, `videos.csv` names the current video of every demo, and `review.csv` holds its review.
The labels of the sets come from these lists.

## Run

One command, from the repo root on the machine with `claude`, with a config of `configs/` and a name for the run:

    python scripts/vlm_loop/loop.py scripts/vlm_loop/configs/franka_train_test.json --name franka_v2

| config | sets (made at the start from the share folder) | checked after the last round |
|---|---|---|
| `franka_train_test_other.json` | train: first 50 good + 50 bad; test: next 50 + 50; other: all other reviewed videos | test, other |
| `franka_train_test.json` | train: first 50 good + 50 bad; test: next 50 + 50 | test |
| `franka_all.json` | all: every reviewed video | - |
| `yam_train_test_other.json` | train: first 15 good + 15 bad; test: next 15 + 15; other: all other reviewed videos | test, other |
| `yam_train_test.json` | train: first 25 good + 25 bad; test: all other reviewed videos | test |
| `yam_all.json` | all: every reviewed video | - |

- The run folder is `outputs/vlm_loop/<name>`. The same command with the same name goes on after a stop: rounds
  with a score are not run again. A change in `local.json` does not stop this. A new name starts a new run, so
  several runs can go at the same time.
- The numbers of a set are the `make_set.py` options in `sets_make` of the config (`first_good`, `first_bad`,
  `demos`, `reviewed_only`); each set leaves out the videos of the sets before it.
- In place of `sets_make`, `"sets"` can name a sets CSV that is there already, for example
  `scripts/vlm_loop/results/vlm_v1.0/franka_sets_vlm_v1.0.csv` to run the reported sets again.
- Stop a run with a file `STOP` in its run folder, or with SIGTERM; it ends after the current step. Delete `STOP`
  before you run the command again.
- At the end the loop writes the final results to `outputs/vlm_loop/<name>/final/`: `sets.csv`, `questions.json`
  (the picked questions with their thresholds), `pyes.csv` (their P(yes) for every scored video) and `summary.md`.
  `score.py --pyes` scores them again without Qwen. To keep a run in git, copy `final/` to
  `scripts/vlm_loop/results/<name>/`.

## A round

- Round 0 runs `round0_questions` (default `scripts/vlm_questions.json`).
- Round r >= 1:
  1. `round<r>/prompt.txt`: the writer prompt (`prompt_template` of the config) filled with the earlier rounds
     and their scores.
  2. The writer runs as `claude -p` in the run folder. It may only read (Read, Glob, Grep) inside the run folder; Bash, Edit, Write and web tools are off. Its report goes to `round<r>/writer_report.md`, every tool call to `round<r>/writer_stream.jsonl` and the files it opened to `round<r>/files_read.json`. A file outside the run folder stops the loop.
  3. The questions come from the ```json block of the report: `round<r>/questions.json`. When `run_batch.py` refuses them (bad questions file), they move to `round<r>/questions_refused.json`, and the next start runs the writer again.
  4. `run_batch.py` runs them on the videos of the set (batch `<batch_prefix><r>`, results in `outputs/gui/<batch_prefix><r>`, not shown in the GUI) and `score.py` scores them. With `calibrate` (default) every threshold is set from the batch: halfway between the highest P(yes) of a good video and the next higher P(yes) of a bad video, so no good video gets Yes (`questions_calibrated.json`, `score_as_run.json` keeps the score with the thresholds as written). With `"calibrate": "max_right"` the thresholds are then lowered where this gives more right videos (see score.py). Good videos can then get Yes. The writer prompts v1 to v8 describe only the first rule. The Y/N in the GUI table use the thresholds as written.
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
| v5 | 61e640e | v4 without the general-question rule; the checker is named "Qwen" | yam_v8, yam_v9, yam_v10 |
| v6 | bcd9a46 | v4 with the checker named "Qwen" | - |
| v7 | 9b4bb07 | v5 + the labels do not judge the color of the robot | yam_v12 |
| v8 | 4510817 | v6 + the labels do not judge the color of the robot | yam_v11, franka_v1, test_v1 |

## Stop and pick

- Stop: `patience` rounds in a row without more correct videos (a tie is no gain), or round `max_rounds`.
- Pick: the round with the most correct videos (with the set thresholds); tie -> fewer bad videos passed as good; tie -> the larger
  margin (the smallest distance between a threshold and any video's P(yes), see `score.py`); tie -> the earlier
  round.
- `summary.md` and `summary.json` (with the picked questions) in the run folder.

## Test and other (optional)

With `test` (and `other`) in the config, the picked questions run once more after the last round: on the test
set, then on the other set. They keep the thresholds set on the loop's set (`score.py` without `--calibrate`).
The results are in `test/` and `other/` (`questions.json`, `score.json`, `pyes.csv`), with a "Test:" and an
"Other:" line in `summary.md`. The writer never sees them: they run after the last round, and no round is written
once `test/` or `other/` exists. They are not run when the loop stopped on an error. Running the same command
again runs only what is missing.

## Status files (for a GUI panel)

- `config.json`: the config of the run, saved at the start. A run folder with another saved config is not run.
  The settings from `local.json` may change.
- `loop_status.json`: the current step, one row per finished round (right, bad passed, good flagged, margin,
  question ids), the stop reason and the pick when finished, and the test result. Written after every step.
- `loop.log`: the text log.

## Results

The question sets in the repo are the reported results. Checker: Qwen2.5-VL-7B-Instruct.

| task | run | questions | train | test | other videos |
|---|---|---|---|---|---|
| YAM | yam_v11, round 4 | `scripts/vlm_questions_yam.json` | 48/50 (bad 23/25) | 41/50 (bad 14/16) | - |
| Franka | franka_v1, round 2 | `scripts/vlm_questions_franka.json` | 64/100 (bad 14/50) | 55/100 (bad 9/50) | 383/538 (bad 38/165) |

- Good videos flagged: YAM 0/25 (train) and 7/34 (test). Franka 0/50 (train), 4/50 (test) and 28/373 (other videos).
- Good = approved + weak in the human review. Bad = rejected.
- The thresholds are set on the train videos with the default `--calibrate` rule (no `max_right`).
- Franka other videos are the videos of the saved review (`franka_review_vlm_v1.0.csv`, 738 videos) that are not
  in train or test. Their P(yes) come from batch
  franka_vlm_v1_full. That batch also asked one more question (room_smear). It is left out of this score.

Score the saved results in `scripts/vlm_loop/results/vlm_v1.0/` again (here Franka test):

    python scripts/vlm_loop/score.py --pyes scripts/vlm_loop/results/vlm_v1.0/franka_pyes_vlm_v1.0.csv --questions scripts/vlm_questions_franka.json --sets scripts/vlm_loop/results/vlm_v1.0/franka_sets_vlm_v1.0.csv --set test OUT_DIR

Score a CoTracker GUI batch on a folder of a share folder (here `<share>/videos/O`):

    python scripts/vlm_loop/score.py outputs/gui/O
