---
name: precise-file-editing
description: Fast, precise, verifiable edits to files through shell one-liners — perl/sed in-place substitution, heredoc whole-file writes, targeted reads with line numbers, and the check-edit-verify loop that keeps each edit surgical. Use when editing config files, code, or any text file via bash_tool, when writing a whole file in one step, or when you need to change one exact passage without touching anything else.
---

# Precise File Editing

bash_tool runs strings; this skill is how to make those strings do surgical
file work in the fewest round trips. Every pattern here stays inside the
guardrail's normal rules when you work on files inside the project root —
no special mode needed.

## The one-pass habit: check, edit, verify in a single command

Never spend three round trips where one chained command does all of it:

```bash
grep -c 'listen_port = 8080' app.conf && \
perl -pi -e 's/\blisten_port = 8080\b/listen_port = 9090/' app.conf && \
grep -n 'listen_port' app.conf
```

The leading `grep -c` is the safety gate: it prints the match count and
returns nonzero when the pattern is absent, so `&&` refuses to run the edit
against a pattern that isn't there. The trailing `grep -n` shows the result
in the same output. One call, full evidence.

If the count is not 1, do not blind-edit: read the surrounding context first
(`grep -n -C3 'pattern' file`), then anchor your pattern with enough
neighboring text to be unique.

## Substitution: perl is the workhorse

```bash
perl -pi -e 's/old/new/g' file        # every occurrence, in place
perl -pi -e 's/old/new/'  file        # first occurrence per line
perl -0pi -e 's/old/new/g' file       # slurp mode: patterns may span lines
```

Why perl over sed for anything nontrivial:

- **`\Q...\E` quotemeta**: `s/\Q$old\E/new/` treats everything between `\Q`
  and `\E` as literal text. Dots, stars, brackets and dollar signs in the
  thing you are replacing stop being regex metacharacters. When copying text
  straight out of a file, wrap it in `\Q...\E` unless you intentionally want
  pattern behavior.
- **Non-greedy and span-line matching** with `-0` (slurp the whole file, so
  `s/<a>.*?<\/a>//s` works across newlines).
- **Multiple edits in one pass**: `perl -pi -e 's/a/b/; s/c/d/' file`.

sed stays fine for simple, single-line literal swaps, and on this platform
BSD sed needs the empty suffix argument: `sed -i '' 's/a/b/' file`. The
guardrail treats that empty token correctly (it is a flag suffix, not a
path), so it passes validation inside the project root.

## Delimiters: dodge escaping entirely

When the text contains slashes (paths, URLs, closing HTML tags), switch the
delimiter instead of escaping: `perl -pi -e 's#/old/path#/new/path#' file`.
Any punctuation character works after the `s`.

## Writing a whole file: quoted heredoc

```bash
cat > service.conf <<'EOF'
host = 127.0.0.1
port = 8080
path = /data/records    # the quoted delimiter means $ and backticks stay literal
EOF
```

The quoted `'EOF'` delimiter disables expansion — `$PATH`, backticks and
`\n` in the body stay as literal text. Omit the quotes only when you
deliberately want variable expansion. Append with `>>` instead of `>`. To
edit part of a file rather than all of it, chain: write the new tail to a
temp file, then `cat tail.txt >> target` or splice with `awk`.

## Targeted reading: lines, not files

Do not dump a whole file when you need three lines from it:

```bash
sed -n '120,145p' file          # an exact line range
grep -n -C3 'keyword' file      # matches with context, with line numbers
awk 'NR>=200 && NR<=260' file    # range with computation available if needed
head -40 file / tail -n +100 file
wc -l file                       # orientation before range reads
```

`grep -n` first, then `sed -n 'X,Yp'` around the reported line — that loop
answers most "where is it and what does the surrounding code look like"
questions in two calls.

## Line surgery: awk for structure sed can't express

```bash
awk '!/^#/' file > file.new && mv file.new file      # drop all comment lines
awk 'NR==42 {print "replacement"} NR!=42' file      # replace one line by number
awk '/START/,/END/' file                            # print a marker-bounded block
```

## Discipline

- **Literal paths for destructive verbs.** The guardrail refuses `$VAR`/`~`
  operands on deletes and overwrites because it cannot prove where they
  resolve. That is a prompt to write the path out, not to route around.
- **Edit inside the project root whenever the file allows it** — in-place
  edits there are ordinary permitted work. If the target is a system file
  (e.g. under `/etc`), the edit itself is recoverable but still normally
  refused; say the command and its purpose to the human and ask, and only
  then consider `skip_permissions` — it is their override to grant, never
  yours to assume.
- **After any multi-file edit, show your work**: `git diff --stat` (or
  `diff` against the backup) in the same command. The reader should be able
  to see exactly what changed from the tool result alone.
- **One concern per command.** A command that edits three unrelated things
  is three edits to verify, fused into one unreviewable blob.
