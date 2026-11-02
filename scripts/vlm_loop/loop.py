"""VLM question loop: a question writer (Claude, through the Claude Code CLI) rewrites the yes/no questions of
the VLM check from the checker's results, round after round, until the score stops going up.

    python scripts/vlm_loop/loop.py CONFIG_JSON

Round 0 runs the starting questions. Every later round:
    1. fills prompt_template.txt with the files of the earlier rounds -> round<r>/prompt.txt
    2. runs the writer in the run folder (`claude -p`), with only read access to the run folder
       -> round<r>/writer_report.md, round<r>/writer_stream.jsonl, round<r>/files_read.json
    3. takes the questions from the ```json block of the report -> round<r>/questions.json
    4. runs the batch on the machine of the GUI server (run_batch.py) and scores it (score.py); with "calibrate"
       (the default) score.py sets every threshold from the batch's P(yes) values (see score.py), so the writer
       does not guess thresholds
    5. copies score.json, pyes.csv (and score_as_run.json, questions_calibrated.json) and the frames the
       checker saw back -> round<r>/
The loop stops after `patience` rounds in a row without more correct videos (a tie is no gain), after
`max_rounds`, when a file STOP appears in the run folder, or on SIGTERM (after the current step). Anything
unexpected (writer error, no ```json block, wrong number of questions, a file read outside the run folder,
a failed batch) stops it with a message in loop_status.json. A round that already has score.json is not run
again, so the same command goes on after a stop.

The final questions: the round with the most correct videos; tie -> fewer bad videos passed as good;
tie -> the larger margin (score.py); tie -> the earlier round. Written to summary.json and summary.md.

Test (optional, config "test"): after the last round, the picked questions with the thresholds set on the
loop's set run once on a test set of other videos and are scored without changing any threshold -> test/ and a
line in summary.md. The writer never sees the test: no round is written once test/ exists. Not run when the loop
stopped on an error.

The loop itself uses only the Python standard library. The batch, the score and the checks of the questions
run where the GUI server and the videos are: the "remote" command prefix of the config (for example
["ssh", "user@host", "docker", "exec", "-i", "container"]), or [] when the loop runs on that machine.

Config (JSON):
    name              run name, used in the batch and question file names
    run_dir           local run folder (prompts, reports, results; the writer reads only here)
    labels            labels.csv of the set (from make_set.py), copied into run_dir/labels.csv
    remote            {"prefix": [...], "repo": repo path there, "python": python there, "work_dir": folder there}
    set_dir           the set folder there (make_set.py)
    sheets_dir        where make_sheets.py writes the sheets there (default <work_dir>/sheets)
    round0_questions  the starting questions, relative to the repo there (default scripts/vlm_questions.json)
    prompt_template   the writer prompt, a file in scripts/vlm_loop/ (default prompt_template.txt)
    calibrate         set the thresholds from each batch's P(yes) values (default true)
    gui, gpus, towel_color, min_free_gb, model     passed to run_batch.py
    wait_minutes, max_wait_hours                   waiting for a free GPU (run_batch.py; default 5 min, 12 h)
    n_questions       questions per round (default 5)
    max_rounds        last round number (default 8)
    patience          rounds without gain before the stop (default 2)
    writer            {"cli": path of the claude command, "model": optional, "effort": optional (low, medium,
                      high, xhigh, max), "timeout_s": default 3600}
    test              optional: {"set_dir": the test set folder there (make_set.py), "labels": its labels.csv
                      there, "batch": GUI batch name (default <name>_test)}
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP_SIGNAL = False


class LoopStop(Exception):
    """Ends the loop with a message (written to loop_status.json)."""


def on_sigterm(signum, frame):
    global STOP_SIGNAL
    STOP_SIGNAL = True


# ---------------------------------------------------------------- remote commands

class Remote:
    """Commands on the machine of the GUI server: prefix + the quoted command, or the command itself."""

    def __init__(self, cfg: dict, log):
        self.prefix, self.repo, self.python = cfg.get("prefix") or [], cfg["repo"], cfg["python"]
        self.work = cfg["work_dir"]
        self.log = log

    def cmd(self, args: list[str]) -> list[str]:
        return self.prefix + [shlex.quote(a) for a in args] if self.prefix else list(args)

    def run(self, args: list[str], stdin: bytes | None = None, stream: bool = False) -> bytes:
        full = self.cmd(args)
        self.log(f"$ {' '.join(args)}")
        if stream:  # print the output while it runs (batches)
            p = subprocess.Popen(full, stdin=subprocess.PIPE if stdin else None, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            if stdin:
                p.stdin.write(stdin)
                p.stdin.close()
            lines = []
            for line in p.stdout:
                text = line.decode(errors="replace").rstrip()
                lines.append(text)
                self.log(f"  {text}")
            if p.wait():
                raise LoopStop(f"command failed ({p.returncode}): {' '.join(args)}\n" + "\n".join(lines[-10:]))
            return "\n".join(lines).encode()
        r = subprocess.run(full, input=stdin, capture_output=True)
        if r.returncode:
            raise LoopStop(f"command failed ({r.returncode}): {' '.join(args)}\n"
                           + r.stderr.decode(errors="replace")[-2000:])
        return r.stdout

    def py(self, script: str, *args: str, stream: bool = False) -> bytes:
        return self.run([self.python, f"{self.repo}/scripts/vlm_loop/{script}", *args], stream=stream)

    def write(self, path: str, data: bytes):
        d = path.rsplit("/", 1)[0]
        self.run(["bash", "-c", f"mkdir -p {shlex.quote(d)} && cat > {shlex.quote(path)}"], stdin=data)

    def read(self, path: str) -> bytes:
        return self.run(["cat", path])

    def fetch(self, base: str, rels: list[str], dest: Path):
        """Copy files or folders under base (paths relative to it) into dest, keeping the relative paths."""
        data = self.run(["tar", "cf", "-", "-C", base, *rels])
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:  # Python before 3.11.4: allow only plain files and folders inside dest
                for m in t.getmembers():
                    if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                        raise LoopStop(f"unsafe path in the copied files: {m.name}")
                t.extractall(dest)


# ---------------------------------------------------------------- the question writer

WRITER_TOOLS = "Read,Glob,Grep"  # read only; nothing that runs code, writes files or reaches the web


def run_writer(prompt: str, run_dir: Path, wcfg: dict, rd: Path, log) -> tuple[str, dict]:
    """`claude -p` in run_dir, restricted mode: only the read tools, file access confined to run_dir, no user
    or project settings, no MCP servers. -> (report text, {"opened": [...], "refused": [...]}) where opened are
    the file tool calls that worked and refused the ones that failed (for example a path outside run_dir)."""
    folder = str(run_dir.resolve())
    cmd = [wcfg.get("cli") or "claude", "-p", "--restricted", "--strict-mcp-config", "--tools", WRITER_TOOLS,
           "--allowedTools", f"Read(/{folder}/**)", f"Glob(/{folder}/**)", f"Grep(/{folder}/**)",
           "--output-format", "stream-json", "--verbose"]
    if wcfg.get("model"):
        cmd += ["--model", wcfg["model"]]
    if wcfg.get("effort"):
        cmd += ["--effort", wcfg["effort"]]
    log(f"writer: claude -p --restricted --tools {WRITER_TOOLS} in {folder}"
        f" (model {wcfg.get('model') or 'default'}, effort {wcfg.get('effort') or 'default'})")
    try:
        r = subprocess.run(cmd, input=prompt.encode(), capture_output=True, cwd=folder,
                           timeout=wcfg.get("timeout_s", 3600))
    except subprocess.TimeoutExpired:
        raise LoopStop(f"writer took longer than {wcfg.get('timeout_s', 3600)} s")
    (rd / "writer_stream.jsonl").write_bytes(r.stdout)
    result, calls, failed = None, {}, set()
    for line in r.stdout.decode(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        if ev.get("type") == "assistant":
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    target = inp.get("file_path") or inp.get("path") or inp.get("pattern")
                    calls[c.get("id")] = f"{c.get('name')}: {target}"
        elif ev.get("type") == "user":
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                    failed.add(c.get("tool_use_id"))
        elif ev.get("type") == "result":
            result = ev
    if r.returncode or result is None or result.get("is_error"):
        raise LoopStop(f"writer failed ({r.returncode}): {r.stderr.decode(errors='replace')[-1500:]}"
                       f"{(result or {}).get('result', '')[-500:]}")
    files = {"opened": [f for i, f in calls.items() if i not in failed],
             "refused": [f for i, f in calls.items() if i in failed]}
    return result.get("result") or "", files


def questions_from(report: str, n: int) -> list[dict]:
    blocks = re.findall(r"```json\s*\n(.*?)```", report, flags=re.S)
    if not blocks:
        raise LoopStop("the writer report has no ```json block")
    try:
        qs = json.loads(blocks[-1] if len(blocks) == 1 else max(blocks, key=len))
    except ValueError as e:
        raise LoopStop(f"the ```json block of the writer report is not valid JSON: {e}")
    if not isinstance(qs, list) or len(qs) != n:
        raise LoopStop(f"the writer gave {len(qs) if isinstance(qs, list) else 'no list of'} questions, not {n}")
    return qs


def outside(files: list[str], run_dir: Path) -> list[str]:
    root = str(run_dir.resolve())
    bad = []
    for f in files:
        path = f.split(": ", 1)[1] if ": " in f else f
        if path and path != "None" and not (path.startswith(root) or not path.startswith("/")):
            bad.append(f)
    return bad


# ---------------------------------------------------------------- the loop

class Loop:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.run_dir = Path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "loop_status.json"
        self.log_file = open(self.run_dir / "loop.log", "a")
        self.remote = Remote(cfg["remote"], self.log)
        self.n_q = cfg.get("n_questions", 5)
        try:
            self.status = json.load(open(self.status_path))
        except (OSError, ValueError):
            self.status = {"name": cfg["name"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "rounds": []}
        self.status.pop("finished", None)
        self.status.pop("stop_reason", None)

    def log(self, text: str):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def save(self, step: str):
        self.status["step"] = step
        tmp = self.status_path.with_suffix(".tmp")
        json.dump(self.status, open(tmp, "w"), indent=1)
        tmp.replace(self.status_path)

    def check_stop(self):
        if STOP_SIGNAL or (self.run_dir / "STOP").exists():
            raise LoopStop("stopped on request (STOP file or SIGTERM)")

    def prepare(self):
        """Labels and sheets into the run folder (once)."""
        rd, rem = self.run_dir, self.remote
        if not (rd / "labels.csv").is_file():
            (rd / "labels.csv").write_bytes(Path(self.cfg["labels"]).read_bytes() if Path(self.cfg["labels"]).is_file()
                                            else rem.read(self.cfg["labels"]))
        rem.write(f"{rem.work}/labels.csv", (rd / "labels.csv").read_bytes())
        with open(rd / "labels.csv", newline="") as f:
            self.labels = {r["video"]: r["label"] for r in csv.DictReader(f)}
        if not ((rd / "frames11").is_dir() and len(list((rd / "frames11").glob("demo_*.jpg"))) == len(self.labels)
                and len(list((rd / "crops11").glob("demo_*.jpg"))) == len(self.labels)):
            sheets = self.cfg.get("sheets_dir") or f"{rem.work}/sheets"
            self.save("making the frame sheets")
            rem.py("make_sheets.py", self.cfg["set_dir"], sheets, "--towel_color", self.cfg["towel_color"],
                   stream=True)
            rem.fetch(sheets, ["frames11", "crops11"], rd)

    def batch(self, name: str, set_dir: str, qfile: str, questions: list[dict]) -> str:
        """The questions as qfile (relative to the repo there), run on set_dir as GUI batch `name`.
        -> the output folder of the batch there."""
        rem, cfg = self.remote, self.cfg
        rem.write(f"{rem.repo}/{qfile}", json.dumps(questions, indent=1).encode())
        self.save(f"batch {name}")
        rem.py("run_batch.py", "--set_dir", set_dir, "--name", name, "--questions", qfile,
               "--gui", cfg.get("gui", "http://localhost:7861"), "--gpus", ",".join(map(str, cfg.get("gpus", [0]))),
               "--towel_color", cfg["towel_color"], "--min_free_gb", str(cfg.get("min_free_gb", 18)),
               "--model", cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
               "--wait_minutes", str(cfg.get("wait_minutes", 5)),
               "--max_wait_hours", str(cfg.get("max_wait_hours", 12)),
               stream=True)
        self.check_stop()
        return f"{rem.repo}/outputs/gui/{name}"

    def run_round(self, r: int, questions: list[dict]) -> dict:
        rd, rem, cfg = self.run_dir / f"round{r}", self.remote, self.cfg
        json.dump(questions, open(rd / "questions.json", "w"), indent=1)
        name = f"{cfg.get('batch_prefix', cfg['name'] + '_round')}{r}"
        out = self.batch(name, cfg["set_dir"], f"outputs/vlm_loop/{cfg['name']}_round{r}_questions.json", questions)
        calibrate = cfg.get("calibrate", True)
        rem.py("score.py", out, f"{rem.work}/labels.csv", f"{rem.work}/round{r}",
               *(["--calibrate"] if calibrate else []))
        rem.fetch(f"{rem.work}/round{r}", ["score.json", "pyes.csv"]
                  + (["score_as_run.json", "questions_calibrated.json"] if calibrate else []), rd)
        rem.fetch(out, [f"{v}/vlm_frames" for v in sorted(self.labels)], rd / "qwen_frames")
        s = json.load(open(rd / "score.json"))
        row = {"round": r, "correct": s["correct"], "videos": s["videos"], "bad_passed": s["bad_passed"],
               "good_flagged": s["good_flagged"], "margin": s["margin"], "batch": name,
               "questions": [q["id"] for q in questions]}
        self.log(f"round {r}: {s['correct']}/{s['videos']} right, bad passed {len(s['bad_passed'])} "
                 f"{s['bad_passed']}, good flagged {len(s['good_flagged'])} {s['good_flagged']}")
        return row

    def prompt(self, r: int) -> str:
        rounds = [x for x in self.status["rounds"] if x["round"] < r]
        names = [f"round{x['round']}/" for x in rounds]
        earlier = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        scores = ", ".join(f"round{x['round']} {x['correct']} of {x['videos']} right" for x in rounds)
        n_good = sum(v == "good" for v in self.labels.values())
        template = HERE / self.cfg.get("prompt_template", "prompt_template.txt")
        return template.read_text().format(
            folder=str(self.run_dir.resolve()) + "/", earlier_rounds=earlier, scores=scores,
            last_round=f"round{r - 1}", n_questions=self.n_q, n_videos=len(self.labels),
            n_good=n_good, n_bad=len(self.labels) - n_good)

    def done_row(self, r: int):
        return next((x for x in self.status["rounds"] if x["round"] == r), None)

    def run(self):
        cfg = self.cfg
        self.save("preparing")
        self.prepare()
        best, since = None, 0
        for r in range(0, cfg.get("max_rounds", 8) + 1):
            self.check_stop()
            rd = self.run_dir / f"round{r}"
            rd.mkdir(exist_ok=True)
            row = self.done_row(r) if (rd / "score.json").is_file() else None
            if row is None:
                if r == 0:
                    q0 = cfg.get("round0_questions", "scripts/vlm_questions.json")
                    questions = json.loads(self.remote.read(f"{self.remote.repo}/{q0}"))
                elif (rd / "questions.json").is_file():  # the writer finished before a stop: same questions
                    questions = json.load(open(rd / "questions.json"))
                    self.log(f"round {r}: questions of the earlier start are used again")
                else:
                    if (self.run_dir / "test").exists():
                        raise LoopStop("the run folder has test results; the writer must not see them, "
                                       "so no new round is written")
                    text = self.prompt(r)
                    (rd / "prompt.txt").write_text(text)
                    self.save(f"round {r}: question writer")
                    report, files = run_writer(text, self.run_dir, cfg.get("writer", {}), rd, self.log)
                    (rd / "writer_report.md").write_text(report)
                    json.dump(files, open(rd / "files_read.json", "w"), indent=1)
                    out_of_folder = outside(files["opened"], self.run_dir)
                    if out_of_folder:
                        raise LoopStop(f"round {r}: the writer read outside the run folder: {out_of_folder}")
                    if files["refused"]:
                        self.log(f"round {r}: refused file calls of the writer: {files['refused']}")
                    questions = questions_from(report, self.n_q)
                    self.check_stop()
                row = self.run_round(r, questions)
                self.status["rounds"] = [x for x in self.status["rounds"] if x["round"] != r] + [row]
                self.save(f"round {r} done")
            if best is None or row["correct"] > best:
                best, since = row["correct"], 0
            elif r > 0:
                since += 1
            if since >= cfg.get("patience", 2):
                return f"no gain in {since} rounds in a row"
        return f"reached max_rounds {cfg.get('max_rounds', 8)}"

    def pick_questions(self, r: int) -> list[dict]:
        """The questions of round r with the thresholds set on the set (as run if not calibrated)."""
        rd = self.run_dir / f"round{r}"
        qf = rd / "questions_calibrated.json"
        return json.load(open(qf if qf.is_file() else rd / "questions.json"))

    def finish(self, reason: str):
        rows = self.status["rounds"]
        rank = lambda x: (x["correct"], -len(x["bad_passed"]), x["margin"], -x["round"])  # noqa: E731
        pick = max(rows, key=rank) if rows else None
        self.status.update(stop_reason=reason, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           pick=pick["round"] if pick else None)
        self.save("finished")
        self.write_summary()
        self.log(f"finished: {reason}; pick round {pick['round'] if pick else '-'}")

    def write_summary(self):
        rows = sorted(self.status["rounds"], key=lambda x: x["round"])
        reason, r, test = self.status.get("stop_reason"), self.status.get("pick"), self.status.get("test")
        pick = next((x for x in rows if x["round"] == r), None)
        lines = [f"# VLM question loop {self.cfg['name']}", "",
                 f"Stop: {reason}. Pick: round {r} ({pick['correct']}/{pick['videos']})." if pick
                 else f"Stop: {reason}.", ""]
        if test and "correct" in test:
            lines += [f"Test: the round {test['round']} questions, thresholds unchanged, on {test['videos']} other "
                      f"videos: {test['correct']}/{test['videos']} right (bad {test['bad_caught']}/"
                      f"{test['bad_total']}), bad passed {len(test['bad_passed'])} {test['bad_passed']}, "
                      f"good flagged {len(test['good_flagged'])} {test['good_flagged']}.", ""]
        elif test:
            lines += [f"Test: {test['error']}", ""]
        lines += ["| round | right | bad passed | good flagged | margin | questions |", "|---|---|---|---|---|---|"]
        lines += [f"| {x['round']} | {x['correct']}/{x['videos']} | {len(x['bad_passed'])} | {len(x['good_flagged'])} "
                  f"| {x['margin']} | {', '.join(x['questions'])} |" for x in rows]
        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n")
        summary = dict(self.status)
        if pick:
            summary["pick_questions"] = self.pick_questions(r)
        json.dump(summary, open(self.run_dir / "summary.json", "w"), indent=1)

    def run_test(self):
        """Config "test": the picked round's questions, with the thresholds set on the loop's set, run once on
        the test set and scored without changing any threshold -> test/ (questions.json, score.json, pyes.csv).
        The writer never sees it: the test runs after the last round, and no round is written once test/ exists."""
        cfg, rem, t, r = self.cfg, self.remote, self.cfg["test"], self.status.get("pick")
        if r is None:
            raise LoopStop("no picked round to test")
        td = self.run_dir / "test"
        if (td / "score.json").is_file():
            self.log("test: scored before, not run again")
        else:
            self.check_stop()
            questions = self.pick_questions(r)
            td.mkdir(exist_ok=True)
            json.dump(questions, open(td / "questions.json", "w"), indent=1)
            name = t.get("batch") or f"{cfg['name']}_test"
            out = self.batch(name, t["set_dir"], f"outputs/vlm_loop/{cfg['name']}_test_questions.json", questions)
            rem.py("score.py", out, t["labels"], f"{rem.work}/test")
            rem.fetch(f"{rem.work}/test", ["score.json", "pyes.csv"], td)
        s = json.load(open(td / "score.json"))
        self.status["test"] = {"round": r, "batch": t.get("batch") or f"{cfg['name']}_test",
                               **{k: s[k] for k in ("correct", "videos", "bad_caught", "bad_total",
                                                    "bad_passed", "good_flagged")}}
        self.save("test done")
        self.write_summary()
        self.log(f"test: round {r} questions on {s['videos']} other videos: {s['correct']}/{s['videos']} right "
                 f"(bad {s['bad_caught']}/{s['bad_total']}), bad passed {len(s['bad_passed'])} {s['bad_passed']}, "
                 f"good flagged {len(s['good_flagged'])} {s['good_flagged']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, on_sigterm)
    loop = Loop(json.load(open(args.config)))
    try:
        reason = loop.run()
    except (LoopStop, Exception) as e:  # every stop ends with the status and the summary written
        reason = f"stopped: {e}" if isinstance(e, LoopStop) else f"stopped by an error: {type(e).__name__}: {e}"
        loop.log(reason)
        loop.finish(reason)
        sys.exit(1)
    loop.finish(reason)
    if loop.cfg.get("test"):
        try:
            loop.run_test()
        except (LoopStop, Exception) as e:
            msg = (f"test stopped: {e}" if isinstance(e, LoopStop)
                   else f"test stopped by an error: {type(e).__name__}: {e}")
            loop.log(msg)
            loop.status["test"] = {"error": msg}
            loop.save("test stopped")
            loop.write_summary()
            sys.exit(1)


if __name__ == "__main__":
    main()
