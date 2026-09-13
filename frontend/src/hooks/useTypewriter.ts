import { useEffect, useState } from "react";

/**
 * Cycles a placeholder-style string through a list of phrases, typing each
 * out character by character then pausing before the next one.
 *
 * The visible string is derived from a character count rather than held in its
 * own state. The earlier version called setDisplayed("") synchronously inside
 * the effect to clear the previous phrase, which React 19's lint rules flag as
 * a cascading render: the effect ran, set state, and forced a second render
 * before the browser painted. Deriving it means the reset happens in the same
 * update that advances the phrase, so there is no intermediate render and no
 * flicker between phrases.
 */
export function useTypewriter(phrases: string[], enabled = true): string {
  const [index, setIndex] = useState(0);
  const [charCount, setCharCount] = useState(0);
  const phrase = phrases[index] ?? "";

  useEffect(() => {
    if (!enabled) return;

    const typeInterval = setInterval(() => {
      setCharCount((count) => (count >= phrase.length ? count : count + 1));
    }, 55);

    const nextPhraseTimeout = setTimeout(
      () => {
        setIndex((i) => (i + 1) % phrases.length);
        setCharCount(0);
      },
      phrase.length * 55 + 2400,
    );

    return () => {
      clearInterval(typeInterval);
      clearTimeout(nextPhraseTimeout);
    };
  }, [phrase, phrases.length, enabled]);

  return phrase.slice(0, charCount);
}
