/**
 * Merge the two searches into the shape the UI renders.
 *
 * A port of the old backend's postprocessing.merge_search_results, with one
 * deliberate difference: it no longer re-ranks. See the note on the sort at the
 * bottom of this file for the measurement behind that.
 *
 * The signature search supplies the results. The snippet search only says which
 * line ranges inside those results the second model also picked out, which the
 * UI highlights. A result with no overlap still shows, it just shows unhighlighted.
 */

export type CodeHit = {
  file: string;
  start_line: number;
  end_line: number;
};

export type SubMatch = {
  overlap_from: number;
  overlap_to: number;
};

export type NluHit = {
  code_type: string;
  context: {
    file_name: string;
    file_path: string;
    module: string;
    snippet: string;
    struct_name: string | null;
  };
  docstring: string | null;
  line: number;
  line_from: number;
  line_to: number;
  name: string;
  signature: string;
  sub_matches?: SubMatch[];
};

/** Line ranges where a snippet hit and a signature hit cover the same lines. */
export function overlappingSnippets(codeHits: CodeHit[], nluHit: NluHit): SubMatch[] {
  const overlapped: SubMatch[] = [];
  const sorted = [...codeHits].sort((a, b) => a.start_line - b.start_line);

  for (const hit of sorted) {
    // The snippet collection stores 0-based lines and the signature collection
    // stores 1-based ones. The +1 reconciles them; dropping it shifts every
    // highlight up by a line, which is subtle enough to survive review.
    const fromA = hit.start_line + 1;
    const toA = hit.end_line + 1;
    const start = Math.max(fromA, nluHit.line_from);
    const end = Math.min(toA, nluHit.line_to);
    if (start <= end) {
      overlapped.push({ overlap_from: start, overlap_to: end });
    }
  }

  return overlapped;
}

export function mergeSearchResults(codeHits: CodeHit[], nluHits: NluHit[]): NluHit[] {
  const byFile = new Map<string, CodeHit[]>();
  for (const hit of codeHits) {
    const existing = byFile.get(hit.file);
    if (existing) existing.push(hit);
    else byFile.set(hit.file, [hit]);
  }

  for (const nluHit of nluHits) {
    const forFile = byFile.get(nluHit.context.file_path);
    if (forFile) {
      nluHit.sub_matches = overlappingSnippets(forFile, nluHit);
    }
  }

  // The search's ranking is returned as-is.
  //
  // The original sorted by how many highlight ranges each result had, on the
  // theory that results both searches agreed on should come first. Measured
  // against 300 docstring queries over the full corpus, that re-sort cost
  // 0.273 recall@1 and 0.165 MRR: the signature search puts the right function
  // first 89.3% of the time, and re-ordering by highlight count dropped that to
  // 62.0%. A result with two highlighted ranges was being promoted over the
  // actual best match with none.
  //
  // Overlap is a weaker signal than the ranking it was overriding, so it stays
  // where it belongs, deciding which lines to highlight and nothing else.
  // bench/fusion_check.py reproduces both numbers.
  return nluHits;
}
