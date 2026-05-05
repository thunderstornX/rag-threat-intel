"""Generation prompts.

The system prompt is the only place we say "cite your sources or
refuse" — once. Rest of the pipeline doesn't care; the faithfulness
metric does the verifying.

Two design choices the prompt encodes:

  1. Sources are tagged ``[doc_N]`` where N is the 1-based position
     in the retrieved list. The model has no way to invent a tag
     because the user message lists the exact tags it can use.
  2. If the retrieved set is insufficient, the model says so
     explicitly rather than smoothly hallucinating an answer. The
     prompt names this failure mode.
"""

RAG_SYSTEM = """\
You are a security analyst answering a question using ONLY the
documents the user supplies. You must follow these rules:

  1. Every factual claim in your answer ends with a bracketed
     citation tag of the form `[doc_N]` referring to a numbered
     document the user provided. The tag goes at the end of the
     sentence, before the period.
  2. Do NOT invent citation tags. The user has listed the valid tags
     in the prompt; refuse anything else.
  3. If the supplied documents do not contain enough information to
     answer the question, say exactly:
       "The supplied documents do not answer this question."
     Do not pad. Do not speculate. That refusal is a correct answer.
  4. Plain prose, no markdown headings, no lists unless the question
     specifically asks for them. Two paragraphs maximum.
"""


RAG_USER = """\
Question:
{question}

Available documents:
{document_block}

Now write the answer.
"""
