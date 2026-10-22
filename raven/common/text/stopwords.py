"""Words that say nothing about what a piece of research is about, for keyword detection and title checks.

Academic filler ("additionally", "certainly"), the vocabulary every paper shares ("ability", "approach"),
bibliographic metadata, rights notices and publisher names. `academic_stopwords` is all of them, meant to
be used beside a language's ordinary stopwords (`raven.common.nlptools.default_stopwords`), not instead.

Used by `raven.visualizer.importer` for keyword detection, through the Visualizer's `custom_stopwords`
setting, and by `raven.papers.scopebib` (planned) to tell an informative title from a thin one. One list,
so that a content word means the same thing in both.
"""

__all__ = ["filler_stopwords", "scilang_stopwords", "metadata_stopwords", "copyright_stopwords",
           "publisher_stopwords", "misc_stopwords",
           "academic_stopwords"]

# This is based on an automated frequency analysis (without these) on a few small datasets of research literature,
# with some manual editing afterward.
#
# When editing this, keep in mind what the lists are for — telling what a piece of research is about —
# and keep them field-agnostic. In the Visualizer's keyword detection, a good `embedding_model` does the
# heavy lifting.
#
# E.g. don't filter:
#  - "best": e.g. "best practices" (software engineering)
#  - "change": e.g. "rate of change" (physics)
#  - "concern": e.g. "separation of concerns" (software engineering)
#  - "condition": e.g. "condition number" (linear algebra)
#  - "field": e.g. "field theory" (physics)
#  - "improved": some names of methods, e.g. "improved hypercube sampling (IHS)" (statistics)
#  - "lower": e.g. "lower triangular" (matrix algebra)
#  - "method": e.g. "finite element method" (applied mathematics, engineering sciences)
#  - "nearly": e.g. "nearly orthogonal" (machine learning, NLP)
#  - "new": geographic names, e.g. "New York"
#  - "paper": e.g. "paper materials" (papermaking)
#  - "performance": e.g. "high performance computing (HPC)"
#  - "potential": e.g. "potential flow" (fluid dynamics), "electric potential"
#  - "review" (also "survey"): vs. a regular study; a useful distinction
#  - "specfic": e.g. "specific heat capacity" (physics)
#  - "system": equation system vs. a single equation
#  - "term": some particular term in an equation (mathematics)
#  - "time": time-dependent analysis vs. steady-state analysis
#  - "used": recycling, green transition... but usually just "we used method X...". Borderline case?
#  - "value": e.g. "value network"
#  - "well": e.g. "oil well"
#  - "work": e.g. "principle of virtual work" (engineering sciences)
#  - "yield": e.g. "yield statement" (programming)

# filler words
filler_stopwords = ["additionally", "afterward", "already", "also", "although", "amid", "amidst", "among",
                    "amount",
                    "become", "beyond",
                    "certain", "certainly", "continuously", "could",
                    "directly", "due",
                    "especially",
                    "finally", "furthermore",
                    "firstly", "secondly", "thirdly",
                    "herein", "however",
                    "indeed",
                    "may", "moreover", "much", "must",
                    "nevertheless",
                    "often", "onto", "overall",
                    "particular",
                    "per",
                    "respectively",
                    "should",
                    "thereby", "though", "thus", "together", "toward", "towards",
                    "upon",
                    "via",
                    "way",
                    "within", "without", "would",
                    "yet"]

# typical scientific language
scilang_stopwords = ["ability", "able",
                     "academia", "academic",
                     "achieve", "achieved",
                     "advantage",
                     "assess",
                     "aim", "application", "analysis", "approach", "area", "article",
                     "approximately",
                     "available",
                     "benefit",
                     "better", "breakthrough",
                     "case",
                     "characterize", "characterise",
                     "compare", "compared", "comparison",
                     "consider", "consideration", "considered", "considering",
                     "corresponding",  # also "corresponding author"
                     "currently",  # but not "current" (as in electrical current)
                     "demonstrate", "demonstrated", "demonstration",
                     "describe",
                     "develop", "developed", "development",
                     "discussed",
                     "enhanced", "evaluate", "evaluated", "evaluation", "exhibit",
                     "establish",
                     "examine", "examined", "examination",
                     "facilitate",
                     "finding",
                     "given", "good",
                     "help", "helped",
                     "identified", "identify",
                     "importance", "important",
                     "improve",
                     "indicate", "indicated", "indication",
                     "introduce",  # what about "introduction"? Useful tutorial text type?
                     "investigate", "investigation",
                     "involve",
                     "main", "mainly",
                     "maintained",
                     "novel",
                     "offer",
                     "possible",
                     "present", "promising", "promote", "proposed",
                     "progress",
                     "prospect",  # what about "prospecting" as in looking for valuable metals?
                     "provide", "provides",
                     "publish", "published",
                     "recent", "related", "reported", "research",
                     "regard",
                     "result", "resulted", "resulting",
                     "reveal", "revealed",
                     "select",
                     "severe",
                     "show", "showed", "shown",
                     "significant", "significantly",
                     "similar", "similarly",
                     "study",
                     "substantial", "substantially",
                     "success", "successful", "successfully",
                     "suggest", "suggested",
                     "suitable", "suited",
                     "summarize", "summarized", "summary",
                     "tackle", "tackled", "tackling",
                     "technical",
                     "test", "typical", "typically",
                     "understand", "understanding", "understood",
                     "undertaken", "undertook",
                     "utilization", "utilize", "utilized", "utilizing",
                     "vary",
                     "widely", "widespread"]
metadata_stopwords = ["academy", "author", "conference", "journal", "institute", "proceedings", "report"]
copyright_stopwords = ["all", "right", "reserved"]

# some common publisher names
publisher_stopwords = ["elsevier", "elsevi",  # "elsevi" is an incorrect lemmatization of "elsevier", though I suppose it makes sense (and leaves me wondering what kind of publication would be the elseviest).
                       "springer",  # we can't do the same for "spring", even though "springer" is much springer than a regular spring. (As of spaCy 3.8.14 / en_core_web_sm 3.8.0 we no longer need to: "Springer" lemmatizes to "springer", not "spring".)
                       "wiley",
                       "llc"]  # Some Company Name LLC

# uncategorized
misc_stopwords = ["center", "constructed", "containing",
                  "different",
                  "excellent",
                  "great",
                  "higher", "highly",
                  "include", "included", "including",
                  "increase", "increased",
                  "like",
                  "made", "make",
                  "role",
                  "sub",  # "<sub>...</sub>" in HTML abstracts
                  "sudden", "superior",
                  "superb",
                  "usage", "use", "using",
                  "various"]

# The final set just combines all of the above.
academic_stopwords = frozenset(filler_stopwords + scilang_stopwords +
                               metadata_stopwords + copyright_stopwords +
                               publisher_stopwords + misc_stopwords)
