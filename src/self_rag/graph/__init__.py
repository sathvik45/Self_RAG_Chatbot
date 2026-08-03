

ANSWER_NODES = {"generate_direct", "generate_answer","revise_answer"}

STEP_LABLES = {
    "contextualize" : "reading the conversation",
    "decide_retrieval" : "Decide whether to search",
    "generate_direct" : "Answering",
    "retrieve" : "Searching the documents",
    "grade_relevance" : "Grading what it found",
    "generate_answer" : "Drafting a grounded answer",
    "no_relevant_docs" : "Nothing relevant found",
    "verify_grounding" : "checking answer against sources",
    "revise_answer" : "Revise for accuracy",
    "check_usefulness" : "Cheking it answers the question",
    "rewrite_query" : "Rephrasing the search",
    "no_answer_found" : "No confident answer"
}