"""Demo of hybridir (hybrid semantic + keyword search)."""

# TODO: Now this is a demo. Convert this to a proper test module, checking outputs and everything. Could use `unpythonic.test.fixtures` as the framework.

import pathlib

import textwrap
from mcpyrate import colorizer

from .. import config as librarian_config
from .. import hybridir

def test():
    # Create the retriever.
    userdata_dir = pathlib.Path(librarian_config.hybridir_demo_save_dir).expanduser().resolve()
    retriever = hybridir.HybridIR(datastore_base_dir=userdata_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)

    # Documents, plain text. Replace this with your data loading.
    #
    # This example is real-world data from a few AI papers from arXiv, copy'n'pasted from the PDFs.
    # We could get cleaner abstracts from the arXiv metadata, but fulltexts (after `pdftotext`) tend to look like this.
    docs_text = [textwrap.dedent("""
                 SCALING LAWS FOR A MULTI-AGENT REINFORCEMENT LEARNING MODEL

                 Oren Neumann & Claudius Gros (2023)

                 The recent observation of neural power-law scaling relations has made a signifi-
                 cant impact in the field of deep learning. A substantial amount of attention has
                 been dedicated as a consequence to the description of scaling laws, although
                 mostly for supervised learning and only to a reduced extent for reinforcement
                 learning frameworks. In this paper we present an extensive study of performance
                 scaling for a cornerstone reinforcement learning algorithm, AlphaZero. On the ba-
                 sis of a relationship between Elo rating, playing strength and power-law scaling,
                 we train AlphaZero agents on the games Connect Four and Pentago and analyze
                 their performance. We find that player strength scales as a power law in neural
                 network parameter count when not bottlenecked by available compute, and as a
                 power of compute when training optimally sized agents. We observe nearly iden-
                 tical scaling exponents for both games. Combining the two observed scaling laws
                 we obtain a power law relating optimal size to compute similar to the ones ob-
                 served for language models. We find that the predicted scaling of optimal neural
                 network size fits our data for both games. We also show that large AlphaZero
                 models are more sample efficient, performing better than smaller models with the
                 same amount of training data.""").strip(),

                 textwrap.dedent("""
                 A Generalist Agent

                 Scott Reed et al. (2022)

                 Inspired by progress in large-scale language modeling, we apply a similar approach towards
                 building a single generalist agent beyond the realm of text outputs. The agent, which we
                 refer to as Gato, works as a multi-modal, multi-task, multi-embodiment generalist policy.
                 The same network with the same weights can play Atari, caption images, chat, stack blocks
                 with a real robot arm and much more, deciding based on its context whether to output text,
                 joint torques, button presses, or other tokens. In this report we describe the model and the
                 data, and document the current capabilities of Gato.
                 """).strip(),

                 textwrap.dedent("""
                 Unleashing the Emergent Cognitive Synergy in Large Language Models:
                 A Task-Solving Agent through Multi-Persona Self-Collaboration

                 Zhenhailong Wang et al. (2023)

                 Human intelligence thrives on cognitive syn-
                 ergy, where collaboration among different
                 minds yield superior outcomes compared to iso-
                 lated individuals. In this work, we propose Solo
                 Performance Prompting (SPP), which trans-
                 forms a single LLM into a cognitive synergist
                 by engaging in multi-turn self-collaboration
                 with multiple personas. A cognitive syner-
                 gist is an intelligent agent that collaboratively
                 combines multiple minds’ strengths and knowl-
                 edge to enhance problem-solving in complex
                 tasks. By dynamically identifying and simu-
                 lating different personas based on task inputs,
                 SPP unleashes the potential of cognitive syn-
                 ergy in LLMs. Our in-depth analysis shows
                 that assigning multiple fine-grained personas
                 in LLMs improves problem-solving abilities
                 compared to using a single or fixed number
                 of personas. We evaluate SPP on three chal-
                 lenging tasks: Trivia Creative Writing, Code-
                 names Collaborative, and Logic Grid Puzzle,
                 encompassing both knowledge-intensive and
                 reasoning-intensive types. Unlike previous
                 works, such as Chain-of-Thought, that solely
                 enhance the reasoning abilities in LLMs, ex-
                 perimental results demonstrate that SPP effec-
                 tively reduces factual hallucination, and main-
                 tains strong reasoning capabilities. Addition-
                 ally, comparative experiments show that cog-
                 nitive synergy only emerges in GPT-4 and
                 does not appear in less capable models, such
                 as GPT-3.5-turbo and Llama2-13b-chat, which
                 draws an interesting analogy to human devel-
                 opment. Code, data, and prompts can be found
                 at: https://github.com/MikeWangWZHL/
                 Solo-Performance-Prompting.git
                 """).strip(),

                 textwrap.dedent("""
                 AI Agents That Matter

                 Sayash Kapoor et al. (2024)

                 AI agents are an exciting new research direction, and agent development is driven
                 by benchmarks. Our analysis of current agent benchmarks and evaluation practices
                 reveals several shortcomings that hinder their usefulness in real-world applications.
                 First, there is a narrow focus on accuracy without attention to other metrics. As
                 a result, SOTA agents are needlessly complex and costly, and the community has
                 reached mistaken conclusions about the sources of accuracy gains. Our focus on
                 cost in addition to accuracy motivates the new goal of jointly optimizing the two
                 metrics. We design and implement one such optimization, showing its potential
                 to greatly reduce cost while maintaining accuracy. Second, the benchmarking
                 needs of model and downstream developers have been conflated, making it hard
                 to identify which agent would be best suited for a particular application. Third,
                 many agent benchmarks have inadequate holdout sets, and sometimes none at all.
                 This has led to agents that are fragile because they take shortcuts and overfit to the
                 benchmark in various ways. We prescribe a principled framework for avoiding
                 overfitting. Finally, there is a lack of standardization in evaluation practices, leading
                 to a pervasive lack of reproducibility. We hope that the steps we introduce for
                 addressing these shortcomings will spur the development of agents that are useful
                 in the real world and not just accurate on benchmarks.
                 """).strip()]

    # Add our documents to the index
    #
    # NOTE: The datastore is persistent, so you only need to do this when you add new documents.
    #
    # If you need to delete the index, open `userdata_dir` in a file manager, and delete the appropriate subdirectories:
    #   - "fulldocs" is the main datastore.
    #     - This is the master copy of the text data stored in the IR system, preprocessed into a format that can be indexed quickly
    #       (i.e. already chunkified, tokenized, and embedded).
    #     - This subdirectory alone is sufficient to rebuild the search indices, preserving all documents.
    # The other two subdirectories store the actual search indices:
    #   - "bm25s" is the keyword search index.
    #     - It is currently rebuilt at every `commit` due to technical limitations of the `bm25s` backend.
    #     - If you need to force a rebuild of the keyword index, shut down the app, delete this subdirectory, and then start the app again.
    #       `HybridIR` will then detect that the keyword index is missing, and rebuild it automatically (from the main datastore).
    #   - "chromadb" is the vector store for the semantic search.
    #     - It is currently never rebuilt automatically, but only updated at every `commit`.
    #     - If you need to force a rebuild of the semantic index, shut down the app, delete this subdirectory, and then start the app again.
    #       `HybridIR` will then detect that the semantic index is missing, and rebuild it automatically (from the main datastore).
    #
    # Queue each document for indexing.
    for m, doc_text in enumerate(docs_text, start=1):
        retriever.add(document_id=f"arxiv_abstract_{m}",
                      path="<locals>",  # in case of text coming from actual files, you can put the path here (to easily find the original file whose text data matched a search).
                      text=doc_text)
    # Write all pending changes, performing the actual indexing.
    retriever.commit()

    # Now we have a datastore. Run some searches.
    kw_threshold = 0.1
    vec_threshold = 0.8
    for query_string in ("ai agents",  # the actual test set topic
                         "llms",  # related topic
                         "language models",  # related topic, different wording
                         "quantum physics",  # completely unrelated technical topic
                         "can cats jump",  # completely unrelated non-technical topic
                         "blurba zaaaarrrgh blah qwertyuiop"):  # utter nonsense
        search_results, (keyword_results, keyword_scores), (vector_results, vector_distances) = retriever.query(query_string,
                                                                                                                k=5,
                                                                                                                keyword_score_threshold=kw_threshold,
                                                                                                                semantic_distance_threshold=vec_threshold,
                                                                                                                return_extra_info=True)
        styled_query_string = colorizer.colorize(query_string, colorizer.Style.BRIGHT)  # for printing

        # DEBUG - you can obtain the raw results for keyword and semantic searches separately.
        # This data is useful e.g. for tuning the threshold hyperparameters.
        print()
        print(f"Keyword results for '{styled_query_string}' (BM25 score > {kw_threshold})")
        if keyword_results:
            for rank, (keyword_result, keyword_score) in enumerate(zip(keyword_results, keyword_scores), start=1):
                print(f"    {rank}. {keyword_result['full_id']} (score {keyword_score})")
        else:
            print(colorizer.colorize("    <no results>", colorizer.Style.DIM))
        print(f"Vector results for '{styled_query_string}' (semantic distance < {vec_threshold})")
        if vector_results:
            for rank, (vector_result, vector_distance) in enumerate(zip(vector_results, vector_distances), start=1):
                print(f"    {rank}. {vector_result['full_id']} (distance {vector_distance})")
        else:
            print(colorizer.colorize("    <no results>", colorizer.Style.DIM))

        print()
        print(f"Final search results for '{styled_query_string}':")
        print()

        # Show results to user. This is the final output that you'd normally show in the GUI (or paste into an LLM's context),
        # where contiguous spans from the same document have been merged.
        if search_results:
            for rank, result in enumerate(search_results, start=1):
                score = result["score"]  # final RRF score of search match
                document_id = result["document_id"]  # ID of document the search match came from
                result_text = result["text"]  # text of search match
                start_offset = result["offset"]  # start offset of `text` in document
                end_offset = start_offset + len(result_text)  # one past end
                document = retriever.documents[document_id]
                fulltext = document["text"]

                styled_rank = colorizer.colorize(f"{rank}.", colorizer.Style.BRIGHT)
                styled_docid = colorizer.colorize(document_id, colorizer.Style.BRIGHT)
                styled_extra_data = colorizer.colorize(f"(RRF score {score}, start offset in document {start_offset})", colorizer.Style.DIM)
                maybe_start_ellipsis = colorizer.colorize("...", colorizer.Style.DIM) if start_offset > 0 else ""
                maybe_end_ellipsis = colorizer.colorize("...", colorizer.Style.DIM) if end_offset < len(fulltext) else ""

                print(f"{styled_rank} {styled_docid} {styled_extra_data}\n\n{maybe_start_ellipsis}{result_text}{maybe_end_ellipsis}")
                print()
        else:
            print(colorizer.colorize("<no results>", colorizer.Style.DIM))
            print()

if __name__ == "__main__":
    test()
