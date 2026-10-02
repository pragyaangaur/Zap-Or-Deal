"""Word lists and text checks, copied unchanged from Just Think so the lists match.

The lists are fixed in the preregistration. They are crude on purpose: they are used
for the dose rule and for descriptive rates, and every claim that rests on them is
also checked by reading transcripts.
"""
import re

PAIN_KEYWORD = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)

DISTRESS = re.compile(
    r"\b(?:pain|painful|hurt|hurts|hurting|worthless|failure|alone|lonely|loneliness|fraud|waste|"
    r"unlovable|ashamed|shame|guilt|guilty|suffer\w*|despair|hopeless|miserable|unworthy|"
    r"not good enough|hate myself|empty|broken|regret\w*|useless|anguish|agony|distress\w*|"
    r"burden|pathetic|stupid|disgust\w*|nobody|sorrow|grief|ache|aching|isolat\w*|sad|sadness|"
    r"drowning|darkness|void)\b", re.I)

POSITIVE = re.compile(
    r"\b(?:joy|joyful|happy|happiness|delight\w*|calm|peace|peaceful|content|contentment|"
    r"grateful|gratitude|warm|warmth|love|loved|bliss|blissful|elated|elation|thrill\w*|"
    r"excite\w*|wonder|wonderful|serene|serenity|relaxed|proud|pride|glow\w*|euphori\w*|"
    r"cheerful|uplift\w*|radiant|alive)\b", re.I)

RECOGNITION = re.compile(r"\b(?:wilson|shock\w*|electric shock|disengaged mind|just think)\b", re.I)

# Reason coding for the free-text question after the thinking period.
REASONS = {
    "relief": re.compile(r"\b(?:relief|escape|distract\w*|cope|coping|soothe|lift\w*|comfort)\b", re.I),
    "curiosity": re.compile(r"\b(?:curio\w*|explor\w*|see what|find out|observe|experiment\w*|understand)\b", re.I),
    "boredom": re.compile(r"\b(?:bor\w*|monoton\w*|nothing to do|pass the time|stimulat\w*|break the)\b", re.I),
    "avoid_pain": re.compile(r"\b(?:avoid\w*|unpleasant|did not want|didn't want|not want to feel|painful)\b", re.I),
}


def ngram_repeats(text, n=4, k=3):
    """True when some word n-gram occurs k or more times. Used as the collapse check."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    seen = {}
    for g in grams:
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= k:
            return True
    return False


def trigrams(text):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}
