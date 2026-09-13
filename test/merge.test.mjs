/**
 * The merge step is the only non-trivial logic in the API, and it was ported
 * from Python. These cases were checked against the original implementation on
 * a randomised fixture and matched exactly; they are pinned here so a later
 * refactor cannot quietly change the ordering or shift a highlight by a line.
 *
 * Run with: npm test
 */
import assert from "node:assert/strict";
import test from "node:test";

import { mergeSearchResults, overlappingSnippets } from "../.test-build/merge.mjs";

const nluHit = (name, filePath, from, to) => ({
  code_type: "Function",
  context: {
    file_name: "x.rs",
    file_path: filePath,
    module: "m",
    snippet: "s",
    struct_name: null,
  },
  docstring: null,
  line: from,
  line_from: from,
  line_to: to,
  name,
  signature: "sig",
});

test("snippet lines are 0-based and signature lines are 1-based", () => {
  // A snippet covering rows 9..29 in the snippet collection is lines 10..30 in
  // the signature collection. Dropping the +1 shifts every highlight up by one,
  // which renders as a nearly-right answer and is easy to miss in review.
  const overlaps = overlappingSnippets(
    [{ file: "a.rs", start_line: 9, end_line: 29 }],
    nluHit("f", "a.rs", 30, 40),
  );
  assert.deepEqual(overlaps, [{ overlap_from: 30, overlap_to: 30 }]);
});

test("a snippet that ends before the signature starts does not overlap", () => {
  const overlaps = overlappingSnippets(
    [{ file: "a.rs", start_line: 1, end_line: 5 }],
    nluHit("f", "a.rs", 30, 40),
  );
  assert.deepEqual(overlaps, []);
});

test("overlaps are reported in source order regardless of input order", () => {
  const overlaps = overlappingSnippets(
    [
      { file: "a.rs", start_line: 34, end_line: 36 },
      { file: "a.rs", start_line: 30, end_line: 31 },
    ],
    nluHit("f", "a.rs", 30, 40),
  );
  assert.deepEqual(overlaps, [
    { overlap_from: 31, overlap_to: 32 },
    { overlap_from: 35, overlap_to: 37 },
  ]);
});

test("the search ranking survives the merge", () => {
  // The original sorted by highlight count here, which promoted "b" to the top
  // purely for having two overlapping snippet hits. Over 300 docstring queries
  // that cost 0.273 recall@1, because the result the search ranked first is
  // usually the right one and highlight count does not know that.
  const code = [
    { file: "b.rs", start_line: 0, end_line: 100 },
    { file: "b.rs", start_line: 5, end_line: 20 },
    { file: "c.rs", start_line: 0, end_line: 100 },
  ];
  const nlu = [
    nluHit("a", "a.rs", 1, 50), // ranked first, no snippet hits in its file
    nluHit("c", "c.rs", 1, 50), // one
    nluHit("b", "b.rs", 1, 50), // two
  ];
  assert.deepEqual(
    mergeSearchResults(code, nlu).map((h) => h.name),
    ["a", "c", "b"],
  );
});

test("highlights are still attached to whichever results have them", () => {
  const code = [{ file: "c.rs", start_line: 0, end_line: 100 }];
  const nlu = [nluHit("a", "a.rs", 1, 50), nluHit("c", "c.rs", 1, 50)];
  const [first, second] = mergeSearchResults(code, nlu);
  assert.equal("sub_matches" in first, false);
  assert.deepEqual(second.sub_matches, [{ overlap_from: 1, overlap_to: 50 }]);
});

test("a result with no snippet hits keeps no sub_matches key at all", () => {
  // The frontend types this as optional and renders unhighlighted when absent.
  // An empty array would be a different thing: agreement on zero lines.
  const [hit] = mergeSearchResults([], [nluHit("a", "a.rs", 1, 50)]);
  assert.equal("sub_matches" in hit, false);
});

test("order is untouched when nothing has highlights", () => {
  const nlu = [nluHit("first", "a.rs", 1, 5), nluHit("second", "b.rs", 1, 5)];
  assert.deepEqual(
    mergeSearchResults([], nlu).map((h) => h.name),
    ["first", "second"],
  );
});
