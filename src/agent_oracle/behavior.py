"""OMP-compatible user behavior metrics and query-time aggregations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class UserMessageMetrics:
    """Behavior counters extracted from one user message."""

    chars: int
    words: int
    yelling: int
    profanity: int
    anguish: int
    negation: int
    repetition: int
    blame: int


_PROFANITY = [
    "fuck",
    "fucks",
    "fucked",
    "fucking",
    "fuckin",
    "fucker",
    "fuckers",
    "fuckup",
    "fuckups",
    "fuckhead",
    "fuckheads",
    "fuckface",
    "fuckwit",
    "fuckwits",
    "fucktard",
    "fuckery",
    "fuckoff",
    "motherfucker",
    "motherfuckers",
    "motherfucking",
    "clusterfuck",
    "ratfuck",
    "unfuck",
    "fk",
    "fks",
    "fking",
    "fkin",
    "fker",
    "fck",
    "fcks",
    "fcking",
    "fckin",
    "fcker",
    "fuk",
    "fuking",
    "fukin",
    "eff",
    "effs",
    "effed",
    "effing",
    "frick",
    "fricks",
    "fricked",
    "fricking",
    "frickin",
    "freaking",
    "freakin",
    "freaked",
    "shit",
    "shits",
    "shat",
    "shitty",
    "shittier",
    "shittiest",
    "shite",
    "shites",
    "shited",
    "shitting",
    "shitter",
    "shitters",
    "shithead",
    "shitheads",
    "shitshow",
    "shitstorm",
    "shitstain",
    "shitfaced",
    "shitload",
    "shitbag",
    "shitcan",
    "shitcanned",
    "shitpost",
    "shitposting",
    "bullshit",
    "bullshits",
    "bullshitting",
    "bullshitter",
    "horseshit",
    "batshit",
    "dogshit",
    "dipshit",
    "jackshit",
    "dumbshit",
    "holyshit",
    "damn",
    "damns",
    "damned",
    "damning",
    "dammit",
    "goddamn",
    "goddamned",
    "goddamnit",
    "goddammit",
    "darn",
    "darns",
    "darned",
    "darnit",
    "dang",
    "danged",
    "dangit",
    "hell",
    "hells",
    "heck",
    "hecks",
    "heckin",
    "gosh",
    "bloody",
    "bollocks",
    "bollox",
    "crap",
    "craps",
    "crappy",
    "crappier",
    "crappiest",
    "crapped",
    "crapping",
    "crapload",
    "crapola",
    "piss",
    "pisses",
    "pissed",
    "pissing",
    "pisser",
    "pisspoor",
    "pisstake",
    "pisshead",
    "ass",
    "asses",
    "asshole",
    "assholes",
    "asshat",
    "asshats",
    "asswipe",
    "asswipes",
    "assclown",
    "assbag",
    "asskisser",
    "dumbass",
    "dumbasses",
    "jackass",
    "jackasses",
    "smartass",
    "smartasses",
    "badass",
    "badasses",
    "lazyass",
    "fatass",
    "hardass",
    "halfass",
    "halfassed",
    "arse",
    "arsed",
    "arsehole",
    "arseholes",
    "arsewipe",
    "bitch",
    "bitches",
    "bitched",
    "bitching",
    "bitchy",
    "bitchier",
    "bitchiest",
    "sonofabitch",
    "biatch",
    "biotch",
    "cunt",
    "cunts",
    "cunty",
    "cuntish",
    "twat",
    "twats",
    "twatty",
    "bastard",
    "bastards",
    "dick",
    "dicks",
    "dickhead",
    "dickheads",
    "dickish",
    "dickwad",
    "dickwads",
    "dickface",
    "dickbag",
    "prick",
    "pricks",
    "prickish",
    "cock",
    "cocks",
    "cocky",
    "cockier",
    "cockiest",
    "cockhead",
    "cockblock",
    "cocksucker",
    "cocksuckers",
    "knobhead",
    "knobheads",
    "knobend",
    "wanker",
    "wankers",
    "wankery",
    "tosser",
    "tossers",
    "jerkoff",
    "jerkoffs",
    "douche",
    "douches",
    "douchebag",
    "douchebags",
    "douchey",
    "scumbag",
    "scumbags",
    "scum",
    "sleazebag",
    "sleazeball",
    "slimeball",
    "lowlife",
    "lowlifes",
    "deadbeat",
    "idiot",
    "idiots",
    "idiotic",
    "idiocy",
    "stupid",
    "stupider",
    "stupidest",
    "stupidity",
    "moron",
    "morons",
    "moronic",
    "imbecile",
    "imbeciles",
    "retard",
    "retards",
    "retarded",
    "dumb",
    "dumber",
    "dumbest",
    "dumbo",
    "fool",
    "fools",
    "foolish",
    "foolery",
    "clown",
    "clowns",
    "clownish",
    "buffoon",
    "buffoons",
    "simpleton",
    "halfwit",
    "halfwits",
    "nitwit",
    "nitwits",
    "dimwit",
    "dimwits",
    "dolt",
    "dolts",
    "doltish",
    "knucklehead",
    "knuckleheads",
    "blockhead",
    "blockheads",
    "lamebrain",
    "airhead",
    "airheads",
    "scatterbrain",
    "numbnuts",
    "numbskull",
    "numpty",
    "numpties",
    "muppet",
    "muppets",
    "pillock",
    "pillocks",
    "plonker",
    "plonkers",
    "prat",
    "prats",
    "berk",
    "berks",
    "ninny",
    "ninnies",
    "dingbat",
    "dingbats",
    "putz",
    "putzes",
    "schmuck",
    "schmucks",
    "jerk",
    "jerks",
    "jerkface",
    "gits",
    "sod",
    "sodding",
    "bugger",
    "buggered",
    "suck",
    "sucks",
    "sucked",
    "sucking",
    "sucky",
    "suckage",
    "trashy",
    "jesus",
    "christ",
    "jeez",
    "jeezus",
    "sheesh",
    "godsake",
    "wtf",
    "wth",
    "wtaf",
    "stfu",
    "gtfo",
    "omfg",
    "omg",
    "ffs",
    "jfc",
    "kys",
    "fml",
    "smh",
    "smdh",
    "smfh",
    "idgaf",
    "idfc",
    "lmfao",
    "fubar",
    "snafu",
]

_PROFANITY_RE = re.compile(r"\b(?:" + "|".join(_PROFANITY) + r")\b", re.IGNORECASE | re.ASCII)
_DRAMA_RE = re.compile(r"[!?][!?1]{2,}")
_ANGUISH_RE = re.compile(
    r"\b(?:no{3,}|a+h{2,}|u+r?g+h+|a+r+g+h+|g+r{2,}|st+o{3,}p+|w+h+y{3,}|"
    r"f+u{3,}c*k*|wtf{3,}|o+m+g{2,}|ye+s{3,}|g+o+d{3,}|br+u+h{2,})\b",
    re.IGNORECASE | re.ASCII,
)
_DUDE_RE = re.compile(r"\bdude\b", re.IGNORECASE | re.ASCII)
_SAD_EMOTICON_RE = re.compile(r"(?<![^\s.!?])[:;]-?\(+")
_NEGATION_LEAD_RE = re.compile(
    r"^[ \t]*(?:(?:nope|nah|nvm|wrong|incorrect)\b|no(?=\s*(?:[,.!?;:\u2013\u2014]|-(?!\w)|$|"
    r"(?:i|im|u|you|ur|we|it|its|that|thats|this|the|they|theyre|he|she|man|dude|bro|wait|dont|not|"
    r"stop|just|again|please|plz|but|actually|literally|seriously|sorry|no|never|nothing|wtf|why|what|wrong)\b)))",
    re.IGNORECASE | re.ASCII,
)
_NEGATION_PHRASE_RE = re.compile(
    r"\b(?:that['\u2019]?s\s+not\s+(?:what|right|it)|not\s+what\s+i\s+(?:meant|asked|said|wanted)|"
    r"makes\s+(?:no|zero)\s+sense)\b",
    re.IGNORECASE | re.ASCII,
)
_REPETITION_RECALL_RE = re.compile(
    r"\b(?:(?:like|as)\s+i\s+(?:said|told\s+you|asked)|i\s+(?:meant|said|told\s+you|asked\s+you|"
    r"already\s+(?:said|told|did|asked|wrote)))\b",
    re.IGNORECASE | re.ASCII,
)
_REPETITION_STILL_RE = re.compile(
    r"\bstill\s+(?:doesn['\u2019]?t|doesnt|isn['\u2019]?t|isnt|not|broken|wrong|fails|failing|the\s+same|same)\b",
    re.IGNORECASE | re.ASCII,
)
_BLAME_YOU_RE = re.compile(
    r"\byou\s+(?:didn['\u2019]?t|did\s+not|broke|missed|forgot|keep|always|never|still|ignored)\b",
    re.IGNORECASE | re.ASCII,
)
_BLAME_WHY_RE = re.compile(r"\bwhy\s+(?:would|did)\s+(?:you|u)\b", re.IGNORECASE | re.ASCII)
_BLAME_STOP_RE = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*stop\s+\w+ing\b", re.IGNORECASE | re.MULTILINE | re.ASCII
)
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
_XML_TAG_PAIR_RE = re.compile(r"<([A-Za-z][\w-]*)\b[^>]*>[\s\S]*?</\1>", re.ASCII)
_XML_TAG_BARE_RE = re.compile(r"</?[A-Za-z][\w-]*\b[^>]*\/?>", re.ASCII)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_URL_RE = re.compile(r"\bhttps?://\S+", re.IGNORECASE | re.ASCII)
_FILE_MENTION_RE = re.compile(r"(^|\s)@[\w./-]+", re.ASCII)
_DOTTED_TOKEN_RE = re.compile(
    r"(^|[\s(\"'\[])([\w-]+(?:\.[\w-]+)+)(?=$|[\s)\"'\],:;!?]|\.(?!\w))", re.ASCII
)
_QUOTE_LINE_RE = re.compile(r"^[ \t]*>.*$", re.MULTILINE)
_IMAGE_MARKER_RE = re.compile(r"\[Image #\d+\]")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _count_matches(text: str, pattern: re.Pattern[str]) -> int:
    """Return the number of non-overlapping *pattern* matches in *text*."""
    return sum(1 for _ in pattern.finditer(text))


def _is_shouted_sentence(sentence: str) -> bool:
    """Apply OMP's uppercase-run guard to one sentence."""
    runs: list[str] = []
    run = ""
    for char in sentence:
        if char.isalpha() and char.isupper():
            run += char
        else:
            if len(run) >= 2:
                runs.append(run)
            run = ""
    if len(run) >= 2:
        runs.append(run)
    return len(runs) >= 2 or (
        len(runs) == 1
        and len(runs[0]) >= 4
        and any(
            runs[0][index] == runs[0][index + 1] == runs[0][index + 2]
            for index in range(len(runs[0]) - 2)
        )
    )


def _count_yelling_sentences(text: str) -> int:
    """Count OMP shouting sentences in *text*."""
    count = 0
    for match in re.finditer(r"[^.!?\n]+", text):
        sentence = match.group()
        letters = [char for char in sentence if char.isalpha()]
        if len(letters) >= 4 and sum(char.isupper() for char in letters) / len(letters) > 0.5:
            count += int(_is_shouted_sentence(sentence))
    return count


def _strip_structured_content(text: str) -> str:
    """Strip the same structured content that OMP excludes from scoring."""
    text = _FENCED_CODE_RE.sub("\n", text)
    text = _XML_TAG_PAIR_RE.sub("\n", text)
    text = _XML_TAG_BARE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _FILE_MENTION_RE.sub(lambda match: f"{match.group(1)} ", text)
    text = _DOTTED_TOKEN_RE.sub(lambda match: f"{match.group(1)} ", text)
    return _ANSI_ESCAPE_RE.sub("", _IMAGE_MARKER_RE.sub(" ", _QUOTE_LINE_RE.sub("", text)))


def compute_user_message_metrics(text: str) -> UserMessageMetrics:
    """Compute the OMP user-metrics classifier result for *text*."""
    trimmed = text.strip()
    if not trimmed:
        return UserMessageMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    chars = len(trimmed.encode("utf-16-le")) // 2
    words = len(re.findall(r"\S+", trimmed))
    prose = _strip_structured_content(trimmed).strip()
    if not prose or sum(bool(line.strip()) for line in prose.split("\n")) >= 3:
        return UserMessageMetrics(chars, words, 0, 0, 0, 0, 0, 0)
    return UserMessageMetrics(
        chars=chars,
        words=words,
        yelling=_count_yelling_sentences(prose),
        profanity=_count_matches(prose, _PROFANITY_RE),
        anguish=(
            _count_matches(prose, _DRAMA_RE)
            + _count_matches(prose, _ANGUISH_RE)
            + _count_matches(prose, _DUDE_RE)
            + _count_matches(prose, _SAD_EMOTICON_RE)
        ),
        negation=_count_matches(prose, _NEGATION_LEAD_RE)
        + _count_matches(prose, _NEGATION_PHRASE_RE),
        repetition=_count_matches(prose, _REPETITION_RECALL_RE)
        + _count_matches(prose, _REPETITION_STILL_RE),
        blame=(
            _count_matches(prose, _BLAME_YOU_RE)
            + _count_matches(prose, _BLAME_WHY_RE)
            + _count_matches(prose, _BLAME_STOP_RE)
        ),
    )


def summarize_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate OMP behavior metrics while preserving the public module API."""
    from agent_oracle.behavior_summary import summarize_messages as summarize

    return summarize(messages)
