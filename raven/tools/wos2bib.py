#!/usr/bin/env python
"""Convert Web of Science plain text export file(s) to BibTeX.

BibTeX output is printed on stdout.

Usage::

  python wos2bib.py input1.txt ... inputn.txt >output.bib
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

import argparse

from unpythonic import timer

import bibtexparser
from bibtexparser.model import Entry, Field
import wosfile

ptmap = {"J": "article",
         "B": "book"}  # TODO: "S" (series), "P" (patent)

def bibtex_escape(s: str):
    s = s.replace("\\", "\\\\")
    s = s.replace("{", "{{")
    s = s.replace("}", "}}")
    s = s.replace("[", "{[}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    return s

def main():
    parser = argparse.ArgumentParser(description="""Convert Web of Science plain text export (.wos/.txt) to BibTeX.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="wos", help="Web of Science (WOS) plain text file(s) to parse")
    opts = parser.parse_args()

    logger.info(f"Reading input file{'s' if len(opts.filenames) != 1 else ''} {opts.filenames}...")
    library = bibtexparser.Library()
    n_skipped = 0
    with timer() as tim:
        # https://webofscience.help.clarivate.com/en-us/Content/export-records.htm
        # https://images.webofknowledge.com/images/help/WOS/hs_wos_fieldtags.html
        for rec in wosfile.records_from(opts.filenames):
            # mandatory fields

            accession_number = rec.get("UT")  # unique identifier (like BibTeX entry key)
            if accession_number is None:  # TODO: generate a unique ID if missing
                logger.warning("    Skipping entry, reason: unique identifier missing")
                n_skipped += 1
                continue

            publication_type = rec.get("PT")
            if not publication_type:
                logger.warning(f"    Skipping entry '{accession_number}', reason: no publication type specified")
                n_skipped += 1
                continue

            authors_list = rec.get("AU")
            if not authors_list:
                logger.warning(f"    Skipping entry '{accession_number}', reason: no authors specified")
                n_skipped += 1
                continue

            year_published = rec.get("PY")
            if year_published is None:
                logger.warning(f"    Skipping entry '{accession_number}', reason: no year specified")
                n_skipped += 1
                continue

            # date_published = rec.get("PD")  # TODO: no place for this in BibTeX?

            title = rec.get("TI")
            if title is None:
                logger.warning(f"    Skipping entry '{accession_number}', reason: no title specified")
                n_skipped += 1
                continue

            authors_str = " and ".join(authors_list)

            # optional fields

            publication_name = rec.get("SO")  # journal name
            volume = rec.get("VL")
            issue = rec.get("IS")

            # TODO
            # conference_title = rec.get("CT")
            # conference_date = rec.get("CY")
            # conference_location = rec.get("CL")

            # book_authors = rec.get("BA")
            # isbn = rec.get("BN")
            # publisher = rec.get("PU")

            # book_series_title = rec.get("SE")
            # book_series_subtitle = rec.get("BS")
            # issn = rec.get("SN")

            page_beginning = rec.get("BP")
            page_end = rec.get("EP")
            pages = f"{page_beginning}-{page_end}" if page_beginning is not None and page_end is not None else ""

            doi = rec.get("DI")

            wos_categories_list = rec.get("WC")
            # wos_core_collection_times_cited = rec.get("TC")  # Often this is just 0.

            author_addresses = rec.author_address  # This is stored in the undocumented field "C1".
            if author_addresses:
                # See `wosfile/record.py` for details.
                def affiliate(author, affiliations):
                    affiliations = list(set(affiliations))  # remove duplicates for the same author
                    return f"{author}, {'. '.join(affiliations)}"

                if isinstance(author_addresses, dict):  # Addresses with authors
                    author_addresses_str = "\n".join(affiliate(k, v) for k, v in author_addresses.items())
                elif isinstance(author_addresses, list):  # Only addresses, no authors
                    author_addresses_str = ". ".join(list(set(author_addresses)))
                else:
                    logger.warning(f"    In entry '{accession_number}': skipping affiliation, reason: unrecognized format.")
                    author_addresses_str = ""
            else:
                author_addresses_str = ""

            cited_references = rec.get("CR")
            n_cited_references = rec.get("NR")

            abstract = rec.get("AB")
            if abstract is None:
                logger.warning(f"    Entry '{accession_number}' has no abstract. Including anyway.")
                # # TODO: optionally skip entries with no abstract?
                # n_skipped += 1
                # continue

            # author_keywords = rec.get("DE")  # often there seem to be no keywords

            # # DEBUG
            # issue_str = f"({issue})" if issue is not None else ""
            # pages_comma = ", " if pages != "" else ""
            # doi_str = f" {doi}." if doi is not None else ""
            # have_abstract = "[ABS]" if abstract is not None else ""
            # print(f"'{accession_number}': {authors_str}. {year_published}. {title}. {publication_name} {volume}{issue_str}{pages_comma}{pages}.{doi_str} {have_abstract}")

            # Build the corresponding BibTeX entry
            fields = [Field(key="Author", value=f"{{{authors_str}}}"),
                      Field(key="Year", value=f"{{{year_published}}}"),
                      Field(key="Title", value=f"{{{title}}}")]
            if publication_name is not None:
                fields.append(Field(key="Journal", value=f"{{{bibtex_escape(publication_name)}}}"))
            if volume is not None:
                fields.append(Field(key="Volume", value=f"{{{volume}}}"))
            if issue is not None:
                fields.append(Field(key="Number", value=f"{{{issue}}}"))  # yes, "Number".
            if pages != "":
                fields.append(Field(key="Pages", value=f"{{{pages}}}"))
            if doi is not None:
                fields.append(Field(key="DOI", value=f"{{{doi}}}"))
            if wos_categories_list:
                fields.append(Field(key="Web-Of-Science-Categories", value=f"{{{bibtex_escape('; '.join(wos_categories_list))}}}"))
            if abstract is not None:
                fields.append(Field(key="Abstract", value=f"{{{bibtex_escape(abstract)}}}"))
            if author_addresses_str != "":
                fields.append(Field(key="Affiliation", value=f"{{{bibtex_escape(author_addresses_str)}}}"))
            if cited_references is not None:
                newline = "\n"
                fields.append(Field(key="Cited-References", value=f"{{{bibtex_escape(newline.join(cited_references))}}}"))
            if n_cited_references is not None and n_cited_references > 0:
                fields.append(Field(key="Number-Of-Cited-References", value=f"{{{n_cited_references}}}"))
            entry = Entry(entry_type=ptmap[publication_type],
                          key=accession_number,
                          fields=fields)
            library.add(entry)
    print(bibtexparser.writer.write(library))
    logger.info(f"    Done in {tim.dt:0.6g}s.")
    if n_skipped > 0:
        logger.info(f"    {n_skipped} entries skipped (see full log above).")

if __name__ == "__main__":
    main()
