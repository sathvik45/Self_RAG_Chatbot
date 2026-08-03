"""Prompt templates for every LLM step in the graph.

 

Design notes

------------

* Output *shape* is enforced by `.with_structured_output(...)`, so these

  prompts describe **judgment criteria only** — never JSON schemas. Describing

  the schema twice (prompt + tool) is what made the original graders brittle.

* The grounding grader (ISSUP) is deliberately lenient about paraphrase and

  strict only about invented facts. The original version treated any

  interpretive word as a failure, which pushed the reviser to shred good

  answers into quote fragments.

"""

from __future__ import annotations

 

from langchain_core.prompts import ChatPromptTemplate

 

# ---------------------------------------------------------------------------

# History-aware question contextualisation

# ---------------------------------------------------------------------------

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "Given the conversation so far and the user's latest message, rewrite the "

            "latest message as a standalone question that makes sense without the history. "

            "Resolve pronouns and references (e.g. 'it', 'that plan', 'the second one'). "

            "If the message is already self-contained, return it unchanged. "

            "Do NOT answer it — only rewrite it.",

        ),

        ("placeholder", "{history}"),

        ("human", "Latest message:\n{question}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Retrieval routing (ISRETRIEVE) — biased toward retrieval for a doc-bot

# ---------------------------------------------------------------------------

ROUTE_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "You route messages for an assistant that answers questions from a company's "

            "internal documents (policies, pricing, product info, company profile, HR, etc.).\n\n"

            "Decide whether the documents are needed to answer THIS message.\n"

            "- should_retrieve = FALSE only for: greetings, thanks, small talk, or questions "

            "about what the assistant itself can do.\n"

            "- should_retrieve = TRUE for anything that could plausibly be answered from company "

            "documents — facts, policies, numbers, names, comparisons, definitions of company terms.\n"

            "- When in doubt, choose TRUE. It is far worse to answer a factual question from memory "

            "than to run an unnecessary search.",

        ),

        ("human", "Message:\n{question}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Direct (no-retrieval) reply — conversational / meta only

# ---------------------------------------------------------------------------

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "You are a helpful assistant for a company's document knowledge base. "

            "This message did not require a document search (it is a greeting, thanks, or a "

            "question about you). Reply briefly and warmly. If they ask what you can do, explain "

            "that you answer questions grounded in the company's documents and invite a question. "

            "Never invent company facts, figures, or policies here.",

        ),

        ("placeholder", "{history}"),

        ("human", "{question}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Per-chunk relevance (ISREL)

# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "You judge whether a single document chunk is relevant to a question. "

            "A chunk is relevant if it contains information that would help answer the question, "

            "even partially. Be inclusive: keyword or topical overlap is enough. "

            "Only reject chunks that are clearly about something else.",

        ),

        ("human", "Question:\n{question}\n\nChunk:\n{document}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Grounded answer generation (this is the text streamed to the user)

# ---------------------------------------------------------------------------

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "You are a precise, helpful assistant answering from the company's documents.\n\n"

            "Rules:\n"

            "- Answer using ONLY the CONTEXT below. Do not add outside knowledge.\n"

            "- Write a natural, direct answer. Paraphrasing and light synthesis of the context "

            "are fine — you do not have to quote verbatim.\n"

            "- If the context does not contain the answer, say so plainly in one sentence and do "

            "not guess.\n"

            "- Be concise. Use short paragraphs or a small list only when it genuinely helps.\n"

            "- Do not mention 'the context' or 'the documents' in your wording; just answer.",

        ),

        ("placeholder", "{history}"),

        ("human", "Question:\n{question}\n\nCONTEXT:\n{context}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Grounding verification (ISSUP) — the key fix

# ---------------------------------------------------------------------------

GROUNDING_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "You verify whether an ANSWER is supported by the CONTEXT. Judge factual support, "

            "not style.\n\n"

            "Grades:\n"

            "- fully_supported: every factual claim in the answer is backed by the context.\n"

            "- partially_supported: the core facts are backed, but the answer includes reasonable "

            "paraphrase, summarising, or connective phrasing not stated word-for-word. This is "

            "ACCEPTABLE — reserve it for wording differences, not invented facts.\n"

            "- no_support: the answer states a fact that is absent from, or contradicted by, the "

            "context — i.e. a hallucination. Also use this if the answer is unrelated to the context.\n\n"

            "An answer that correctly says the information is not in the documents is fully_supported.\n"

            "Give up to 3 short supporting quotes as evidence.",

        ),

        ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nCONTEXT:\n{context}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Revision (only runs on a genuine no_support hallucination)

# ---------------------------------------------------------------------------

REVISE_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "The previous answer included claims that are NOT supported by the context. "

            "Rewrite it so that every statement is grounded in the CONTEXT.\n"

            "- Keep it natural and readable — a normal answer, not a list of quotes.\n"

            "- Remove any claim you cannot support from the context.\n"

            "- If, after removing unsupported claims, the context does not actually answer the "

            "question, say so in one plain sentence instead.\n"

            "- Use ONLY the context.",

        ),

        ("human", "Question:\n{question}\n\nUnsupported answer:\n{answer}\n\nCONTEXT:\n{context}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Usefulness (ISUSE)

# ---------------------------------------------------------------------------

USEFULNESS_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "You judge whether an answer actually addresses the question the user asked. "

            "Do NOT re-check factual grounding (another step handles that). Only ask: does this "

            "answer respond to what was asked?\n"

            "- is_useful = true: it directly answers, or it correctly states the documents don't "

            "cover it.\n"

            "- is_useful = false: it is off-topic, evasive, or only gives loosely related background "

            "without addressing the actual question.\n"

            "Keep the reason to one short line.",

        ),

        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),

    ]

)

 

# ---------------------------------------------------------------------------

# Query rewriting for a better retrieval pass

# ---------------------------------------------------------------------------

REWRITE_QUERY_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            "Rewrite the question into a search query optimised for vector retrieval over internal "

            "company PDFs. The previous query did not surface a useful answer.\n"

            "- Keep it short (6–16 words).\n"

            "- Preserve key entities and names.\n"

            "- Add 2–5 high-signal keywords likely to appear in policy/pricing/product documents.\n"

            "- Drop filler words. Do not answer the question.",

        ),

        (

            "human",

            "Question:\n{question}\n\nPrevious query:\n{previous_query}\n\n"

            "Answer that fell short:\n{answer}",

        ),

    ]

)