# treekv: Structure-Aware KV Caching with llama.cpp

**Project report by Pinaksh Kumar - MT25075** · Large Deep Learning Systems (LDLS), IIIT-Delhi, Monsoon 2026 · 

This README is my report for the individual submission. The project was carried out in a team of three (Pinaksh, Reehan and Ishaan). Section 3 describes my own work, and section 4 describes how the work was divided.

This repository contains only the files of my role and the files shared by the whole team. The complete project, including my teammates' files, is in the team repository: https://github.com/Pinaksh-IIITD/LDLS-PROJECT

| | |
|---|---|
| My role | Engine: llama.cpp build (CPU), the shared C++ driver, the mask trace, Test 0 |
| Tasks from the brief assigned to me | CPU build of llama.cpp and the model run; trace of the mask construction; Test 0 (two prompts); the shared C++ driver; keeping everything on the CPU |
| Model | Llama-3.2-1B-Instruct, 4-bit quantized (Q4_K_M), run on the CPU only |
| Inference engine | llama.cpp, fixed at commit `10a60cf` |
| Document types | Markdown and LaTeX, five documents each |
| Hardware | Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz, 4 threads |

## Summary

A language model normally reads a document as one long line of text. In this project, a document is instead treated as a **tree** of sections, and the model is only allowed to connect each part of the text with its own branch of the tree. We built this on top of llama.cpp and tested it on 10 real documents (Markdown and LaTeX). The main results are:

- **Editing one section does not change answers about other sections.** This held in 10 of 10 documents, and the model's output scores did not change at all (largest difference: 0).
- **Removing an unrelated section frees the expected memory and does not change the answer.** This held in 10 of 10 documents. In total, 115.22 MiB of memory was freed, against 115.19 MiB expected from the size formula.
- **Editing one section is much cheaper than reprocessing the whole document**: between 1.9 and 12.9 times faster (median 7.3 times). Removing a section takes less than 0.3 ms.

## 1. Introduction

When a language model reads a document, it stores intermediate results for every word-piece ("token") in memory, so that it does not have to compute them again. This memory is called the **KV cache**. In a normal model, every token is connected to every earlier token. As a result, changing one sentence near the beginning of a document makes the stored results for everything after it invalid, and the whole document must be processed again.

Most real documents have structure: a README has sections, and a paper has sections, theorems and proofs. A question about one section rarely needs an unrelated section. The idea of this project is to use that structure. Each part of the document is connected only to its own **branch**: its title, its parent headings and itself. Sibling sections are not connected to each other. Then each section can be **edited** or **removed** from memory on its own, without affecting the others.

The aim of week 1 was to build the basic system on a CPU and to show, with tests, that this separation really holds.

## 2. Key terms

| Term | Meaning in this report |
|---|---|
| Token | A small piece of text (a word or part of a word) that the model reads |
| KV cache | The memory where the model stores its intermediate results ("keys" and "values") for every token it has read |
| Prefill | Reading the whole document once and filling the KV cache |
| Sequence ID | A label that llama.cpp attaches to tokens. A token can only use earlier tokens that carry its sequence ID |
| Leaf | A part of the document with no smaller parts inside it, for example a section without subsections |
| Logits | The scores the model gives to every possible next token. We compare them to check whether anything changed |
| Greedy decoding | Always choosing the token with the highest score, so that the same input always gives the same answer |
| Eviction | Removing a part of the document from the KV cache to free memory |

## 3. My contribution

My assigned role was the **engine**: building llama.cpp for the CPU, writing the shared C++ driver, tracing how llama.cpp builds its attention mask, and running Test 0.

**What I built in my assigned role**

- **The llama.cpp build.** llama.cpp is fixed at commit `10a60cf` and built for the CPU only. The script `tools/setup_driver.sh` downloads that exact version if needed, adds our driver to it and builds everything with one command. llama.cpp normally moves all model layers to a graphics card when one is present, so the driver sets the number of GPU layers to zero to keep the work on the CPU.
- **The driver** (`driver/tree_driver.cpp`). It loads a document's JSON file and processes the whole document in one batch with tree-shaped sequence IDs. It then:
  - answers a question inside one branch;
  - edits a leaf;
  - removes a subtree;
  - measures the freed memory;
  - compares the model's output scores before and after each change;
  - runs Test 0.

  It also checks its inputs and prints a clear message when an argument is missing or wrong.
- **The mask trace** (`docs/mask_trace.md`). It explains the three conditions llama.cpp uses to decide which stored tokens a new token may see. It also shows a real map of the cache for one document (`results/kv_debug.log`): the shared title and headings, one block per section, and sibling sections reusing the same position numbers.
- **Test 0** (`results/test0.json`). Two prompts processed together in one batch gave exactly the same output as when processed separately. Changing the text of one sequence did not change the other sequence's scores at all.

**Beyond my assigned role**

During the final integration on 16 October I also built several parts that were planned for my teammates, with the help of an LLM :

- the LaTeX parser;
- the selection and preparation of the ten documents;
- the full test run on all documents;
- the timing measurements;
- the slide.

These parts are still listed under my teammates' roles in section 4, because that was the planned division of work.

**Problems I met and how I solved them**

- **Positions with gaps.** llama.cpp rejects a sequence whose positions have gaps. Numbering the document in reading order created such gaps, so sibling sections now start their numbering right after their parent.
- **Test 0 failed at first.** I measured the cause instead of guessing:
  - a repeated run was always identical;
  - changing the other prompt did not affect the first one, so the two sequences were not mixing;
  - the difference depended only on the length of the first prompt. First prompts of 12, 16 and 20 tokens gave identical results; 6, 7, 9, 13, 15 and 19 did not.

  The CPU code processes tokens in groups of four and handles the remainder differently. Test 0 is therefore reported with a 12-token first prompt, and the short-prompt case is kept, with this explanation, in `results/test0_short_prompts.json`.
- **Small score differences after some evictions.** When the removed section was stored before the section being asked about, the scores changed slightly although the answer stayed the same. I found that llama.cpp reuses the freed memory slots for the question, which changes the order of the arithmetic. I added the `--keep-layout` option to hold the memory layout fixed, and with it the difference is exactly zero. I also tested two other possible fixes, switching off llama.cpp's fast attention kernel and storing the cached values in 32-bit precision, and showed that neither removes the difference.
- **Long answers.** The model sometimes continued writing after its answer, so the driver now stops each answer at its first line break.
- **Broken characters.** An answer cut off in the middle of a character made the output step fail, so invalid characters are now replaced.

**What I learned**

I learned how a real inference engine stores its KV cache and builds its attention mask. I also learned that simple sequence labels are enough to give a model a tree-shaped view of a document, without changing the engine itself. A second lesson was that floating-point results depend on the order in which numbers are added. When a test expects two results to be exactly equal, every difference has to be explained before it can be accepted. Finally, I learned the value of control experiments: holding the memory layout fixed, or changing only the other prompt, turned a confusing difference into a clear explanation.

## 4. Team and division of work

The work was divided into three roles. The table lists each teammate's assigned role and the files that belong to that role. Section 3 above states which parts I built myself.

| teammate | assigned role | files of that role |
|---|---|---|
| Reehan | Parser & schema: domains (Markdown + LaTeX), JSON schema + merge rules, parsers, 10 documents, tree statistics | `schema/SCHEMA.md`, `parser/tree_schema.py`, `parser/md_parser.py`, `parser/tex_parser.py`, `parser/parse.py`, `parser/make_stats.py`, `tests/test_parsers.py`, `data/SOURCES.md`, `data/markdown/*`, `data/latex/*`, `data/examples/*`, `results/stats.md` |
| Ishaan | Tests & report: batched trace, test cases and runner, timings, slide | `docs/trace_batched.md`, `tests/run_tests.py`, `tests/cases.json`, `results/batched_debug.log`, `results/test_results.md`, `results/timings.md`, `slide/*` |

## 5. Method

### 5.1 Turning a document into a tree

- **Markdown** (`parser/md_parser.py`): every heading starts a new part, which contains the heading and the text up to the next heading. The heading level (`#`, `##`, …) decides which part it belongs to. The markdown-it-py library is used, so a `#` inside a code block is not mistaken for a heading.
- **LaTeX** (`parser/tex_parser.py`): the pylatexenc library reads the document. Chapters, sections and subsections become parts. Theorems, lemmas, definitions, proofs, figures and tables (including custom theorem types defined with `\newtheorem`) become their own parts. Text between them becomes a "text" part. A proof that directly follows a theorem is placed **under** that theorem, so that a question about the proof can also see the statement it proves.
- Both parsers keep every character. Joining all parts in order gives back the original file exactly, and the program checks this for every document.

### 5.2 Merging very small parts

llama.cpp supports at most 256 sequence IDs, and a part of only a few words is not useful on its own. Parts smaller than 32 tokens are therefore merged with a neighbouring part or with their parent, always keeping the original reading order. A small part stays on its own only when it has no suitable neighbour; in the ten documents this happens once (a 19-token closing paragraph in `online_softmax.tex`). At most 254 leaves are allowed, because the driver needs two extra sequence IDs: one for questions (section 5.4) and one for the `--keep-layout` option (finding 4).

### 5.3 Sequence IDs and positions

- Every leaf receives its own sequence ID.
- Every part above a leaf (title, headings) receives the IDs of all leaves below it.
- Each part's token positions start right after its parent's tokens. Sibling parts therefore reuse the same position numbers, and every path from the title down to a leaf is numbered 0, 1, 2, … without gaps. llama.cpp requires this: it rejects positions with gaps.

The result is stored as a JSON file with the fields required by the brief (`tokens`, `positions`, `seq_ids`, `node_type`, `parent`); see `schema/SCHEMA.md`.

### 5.4 The driver program

The C++ driver (`driver/tree_driver.cpp`) follows the structure of llama.cpp's `batched` example. In `batched`, one prompt is shared by several continuations; here, the title and headings are shared by several sections.

1. **Prefill:** all parts of the document are sent to llama.cpp in one batch, each token with its position and its sequence IDs.
2. **Ask a question about leaf A:** A's sequence ID is copied to a spare "scratch" sequence (this copies no data, it only adds a label), the question is added there, and the answer is generated with greedy decoding. The scratch sequence is then deleted, so the question leaves no trace.
3. **Edit leaf B:** B's tokens are removed from the cache, and only the new text of B is processed.
4. **Evict subtree C:** for each of C's own sequence IDs, the tokens in C's position range are removed (`llama_memory_seq_rm`). The freed memory is measured with `llama_state_get_size`.

The program runs on the CPU only. It stops each answer at its first line break.

### 5.5 Why this works inside llama.cpp

llama.cpp builds an attention **mask** that decides which stored tokens each new token may use. A stored token is visible only if (1) its memory slot is in use, (2) it carries the new token's sequence ID, and (3) its position is not later than the new token's position. Everything else is hidden. With the labels from section 5.3, this mask lets each token see exactly its own branch of the tree. This is documented in `docs/mask_trace.md`, with a real map of the cache. The `batched` example is traced in `docs/trace_batched.md`.

## 6. Experimental setup

- **Documents:** five Markdown files (documentation pages from llama.cpp) and five LaTeX files (the standard `sample2e.tex` from TeX Live and four short notes written for this project). All documents are between 487 and 1732 tokens long, below the limit of 2,000 tokens. Sources and licences are listed in `data/SOURCES.md`.
- **Test 0:** two prompts are processed together, in two sequences of one batch, and also separately. Their outputs must be identical.
- **Test 1:** for each document, a question is asked about a leaf A. A different leaf B is edited, and the question is asked again. The answer must not change.
- **Test 2:** a subtree C that does not contain A or its parents is removed, and the question is asked again. The answer must not change, and the memory must drop by the size of C (number of tokens × 32 KiB for this model).
- Each case was run three times, and once more with the option `--keep-layout` (finding 4). Timings are the median of the three normal runs.

## 7. Results

### 7.1 Tree statistics

"Sequences" is the number of leaves. "Markup share" is the part of the tokens that is formatting rather than content (for example `#`, `|` or `\begin{…}`). It is an estimate: the number of tokens of the text minus the number of tokens of the same text with the formatting removed.

| document | tokens | depth | sequences | leaf tokens min / median / max | top-level section sizes | markup share |
|---|---|---|---|---|---|---|
| markdown/embedding.json | 690 | 4 | 7 | 47 / 92 / 162 | 123, 348, 195 | 17.2% |
| markdown/llguidance.json | 922 | 3 | 5 | 94 / 123 / 261 | 94, 143, 123, 98, 261 | 31.1% |
| markdown/mtmd.json | 1545 | 3 | 3 | 143 / 190 / 688 | 190, 143, 688 | 37.8% |
| markdown/rpc.json | 1378 | 3 | 6 | 48 / 287 / 416 | 416, 961 | 9.0% |
| markdown/security.json | 1579 | 4 | 8 | 43 / 152 / 275 | 605, 842 | 20.4% |
| latex/kv_cache.json | 802 | 4 | 6 | 36 / 79 / 115 | 55, 121, 306, 161, 79 | 29.6% |
| latex/online_softmax.json | 716 | 4 | 6 | 19 / 75 / 112 | 177, 236, 177, 57 | 26.3% |
| latex/roofline.json | 487 | 4 | 5 | 45 / 61 / 90 | 125, 143, 97, 63 | 24.8% |
| latex/sample2e.json | 1732 | 2 | 2 | 563 / 916 / 916 | 916, 563 | 31.6% |
| latex/tree_mask.json | 587 | 5 | 5 | 33 / 67 / 107 | 116, 222, 183 | 27.6% |

### 7.2 Test 0: two prompts together versus separately

| | sequence 0 | sequence 1 |
|---|---|---|
| prompt length (tokens) | 12 | 26 |
| output, both prompts in one batch | 'Paris. The capital city of the country …' | 'if n >= 2 else 0' |
| output, each prompt alone | 'Paris. The capital city of the country …' | 'if n >= 2 else 0' |
| largest score difference, first token | 0 | 0 |

**Result: the outputs are identical.** As a control, the text of sequence 1 was replaced by a different sentence; the scores of sequence 0 changed by 0. The two sequences are therefore fully separated.

With very short prompts (6 and 5 tokens), the outputs were not identical: the first-token scores differed by 0.46 and 0.66, while the control still showed 0. The cause is that llama.cpp's CPU code processes tokens in groups of four and handles the remaining tokens with different arithmetic. When the first prompt's length is not a multiple of four, the tokens fall into different groups in the joint run and in the separate runs, so the numbers are rounded differently. First prompts of 12, 16 and 20 tokens gave identical results; first prompts of 6, 7, 9, 13, 15 and 19 tokens did not (`results/test0_short_prompts.json`).

### 7.3 Tests 1 and 2

How to read the table: "PASS" means that the answer about A was identical in every run, and, for Test 2, that the memory dropped by the expected amount. "Max logit diff" is the largest change in the scores of the answer's first token. "Same, layout fixed" is that change when the memory layout is held fixed (see finding 4).

**Worked example** (`data/examples/library_guide.json`, a short made-up library guide; `results/example_run.json`). A = "Weekends", B = "Students" (edited), C = "Printing" (removed).

| step | answer about A | same answer? | largest score difference |
|---|---|---|---|
| before any change | 'The library closes at 6 pm on Sundays' | | |
| after editing B (Test 1) | 'The library closes at 6 pm on Sundays' | yes | 0 |
| after removing C (Test 2) | 'The library closes at 6 pm on Sundays' | yes | 0 |

Removing C deleted sequence 3 over positions [36, 67), that is 31 tokens. The memory freed was 1,016,180 bytes; the formula gives 31 × 32,768 = 1,015,808 bytes, and the extra 372 bytes are llama.cpp's bookkeeping.

**All 10 documents:**

| document | Test 1 | max logit diff | Test 2 | max logit diff | same, layout fixed | KV MiB freed / expected | evicted (seq IDs, positions) | answer about A |
|---|---|---|---|---|---|---|---|---|
| markdown/embedding.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 6.10 / 6.09 | seqs [5, 6], [24, 127) | 'The value 2 of --embd-normalize means that the embedding vec' |
| markdown/llguidance.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 8.16 / 8.16 | seqs [4], [203, 464) | '50μs of single-core CPU time for the [JSON Schema Bench](htt' |
| markdown/mtmd.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 21.51 / 21.50 | seqs [2], [524, 1212) | '`llava.cpp' |
| markdown/rpc.json | PASS | 0.0e+00 | PASS | 4.1e-01 | 0.0e+00 | 13.00 / 13.00 | seqs [0], [1, 417) | 'Large tensors' |
| markdown/security.json | PASS | 0.0e+00 | PASS | 3.8e-01 | 0.0e+00 | 18.91 / 18.91 | seqs [0, 1, 2], [132, 513) | 'Implement a sandboxed environment for the model execution' |
| latex/kv_cache.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 2.47 / 2.47 | seqs [5], [80, 159) | 'Because each token contributes one key and one value, which ' |
| latex/online_softmax.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 5.53 / 5.53 | seqs [3, 4], [68, 191) | '$e^{m_A - m}$' |
| latex/roofline.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 3.03 / 3.03 | seqs [3], [58, 155) | '$\\frac{Q}{B} \\cdot I$ seconds' |
| latex/sample2e.json | PASS | 0.0e+00 | PASS | 4.4e-01 | 0.0e+00 | 29.10 / 29.09 | seqs [0], [253, 1184) | 'Displayed text is set apart from ordinary text by being' |
| latex/tree_mask.json | PASS | 0.0e+00 | PASS | 3.6e-01 | 0.0e+00 | 7.41 / 7.41 | seqs [1, 2], [66, 255) | 'Because it is not on the path of $a$' |

### 7.4 CPU run times

Median of three runs on Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz, 4 threads. "Full prefill / edit B" shows how many times faster it is to process only the edited section than the whole document.

| document | tokens | prefill ms | prefill tokens/s | ask ms (question + answer) | edit B ms | full prefill / edit B | evict C ms |
|---|---|---|---|---|---|---|---|
| markdown/embedding.json | 690 | 10275 | 67 | 1830 | 1033 | 9.9x | 0.071 |
| markdown/llguidance.json | 922 | 17417 | 53 | 2500 | 1997 | 8.7x | 0.091 |
| markdown/mtmd.json | 1545 | 29990 | 52 | 839 | 3989 | 7.5x | 0.178 |
| markdown/rpc.json | 1378 | 23976 | 57 | 591 | 3585 | 6.7x | 0.083 |
| markdown/security.json | 1579 | 29326 | 54 | 926 | 2266 | 12.9x | 0.287 |
| latex/kv_cache.json | 802 | 14196 | 56 | 1517 | 1966 | 7.2x | 0.035 |
| latex/online_softmax.json | 716 | 13720 | 52 | 1088 | 2147 | 6.4x | 0.078 |
| latex/roofline.json | 487 | 8523 | 57 | 1182 | 1321 | 6.5x | 0.039 |
| latex/sample2e.json | 1732 | 33062 | 52 | 1421 | 17865 | 1.9x | 0.223 |
| latex/tree_mask.json | 587 | 10285 | 57 | 1106 | 859 | 12.0x | 0.077 |

## 8. Findings

1. **The separation between sections is exact.** After editing a section, the scores of the answer about another section did not change at all in any document (largest difference 0). Edited text cannot influence another branch.
2. **Edits are much cheaper than a full prefill.** Only the edited section is processed again: 1.9–12.9 times faster than processing the whole document. The smallest gain occurs when the edited section is a large part of the document.
3. **Eviction frees the expected memory.** The freed memory matched the formula in every document. The small extra amount (12 bytes per token) is llama.cpp's bookkeeping.
4. **The memory layout can change the computed scores slightly.** In 4 documents, the scores after an eviction differed by up to 0.44, although the answer text stayed the same. This happened only when the removed section was stored *before* section A in memory: llama.cpp then reuses the freed memory slots for the question, and the different order of the slots changes how the CPU adds the numbers. When the memory layout is kept the same (option `--keep-layout`), the largest difference is 0. The difference is therefore caused by arithmetic order, not by information from the removed section.
5. **In Test 0, the length of the first prompt matters.** The CPU code processes tokens in groups of four. When the first prompt's length was a multiple of four, the joint run and the separate runs were identical. With short first prompts of other lengths, the scores differed slightly, while the separation of the two sequences stayed exact (details in section 7.2).

## 9. Decisions and assumptions

The brief did not answer every question, and no clarification was available before the deadline. The following decisions were made and are stated here openly.

- **Model:** the Instruct version of Llama-3.2-1B was used. It has the same size, architecture and quantization as the base model, but it answers questions and then stops, which suits this task.
- **"Evict a subtree by position range":** a subtree is removed by its own sequence IDs together with its position range. Because sibling sections share position numbers, removing a position range from *all* sequences would also delete unrelated sections.
- **"The output is unchanged" / "the answer holds":** the answer text must be identical in every run. The scores (logits) are reported as additional evidence, as explained in finding 4.
- **AI tools:** an AI assistant(LLM) (Claude) was used for parts of the design, code and writing. All results in this report come from running the code after the review.

## 10. Limitations

- The LaTeX parser does not follow `\input` or `\include` files, and it does not model cross-references between sections (for example, a proof that cites an earlier lemma).
- The model is small (1 billion parameters), so some answers can be imperfect. The tests check that answers do not change, not that they are correct.
- Timings were measured on one laptop CPU with four threads. They change from run to run, so the tables report the median of three runs.

## 11. How to reproduce

These commands need the complete project, so they are run in the team repository.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash tools/setup_driver.sh
mkdir -p models && curl -L -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
.venv/bin/python parser/parse.py data/markdown/*.md data/latex/*.tex --tokenizer llama
.venv/bin/python parser/make_stats.py data/markdown/*.json data/latex/*.json
.venv/bin/python -m pytest -q tests/test_parsers.py
.venv/bin/python tests/run_tests.py --runs 3
./llama.cpp/build/bin/tree-driver -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 24 \
  --two-prompts "Answer briefly: the capital city of the country France is" \
                "def fibonacci(n): return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)" > results/test0.json
.venv/bin/python slide/make_slide.py
```

## 12. Files in this repository

| file | belongs to |
|---|---|
| `.gitignore` | shared |
| `README.md` | shared |
| `docs/mask_trace.md` | my role |
| `driver/CMakeLists.txt` | my role |
| `driver/tree_driver.cpp` | my role |
| `requirements.txt` | shared |
| `results/README.md` | shared |
| `results/example_run.json` | my role |
| `results/kv_debug.log` | my role |
| `results/test0.json` | my role |
| `results/test0_short_prompts.json` | my role |
| `tools/make_submission.py` | shared |
| `tools/ownership.json` | shared |
| `tools/setup_driver.sh` | shared |
