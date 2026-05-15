"""
Tiny bundled corpus for offline / no-HF runs.

Twenty real Wikipedia-flavored paragraphs in English. Not enough to train on,
but enough to verify that the data-ingestion contract works without going to
the network. Used by `loaders.load_corpus(..., source="bundled")`.
"""

BUNDLED_DOCS: list[str] = [
    "Adrian Lewis Peterson is an American football running back for the National "
    "Football League. He played college football at Oklahoma and was drafted by the "
    "Minnesota Vikings seventh overall in the 2007 NFL Draft. Peterson set the NCAA "
    "freshman rushing record with 1,925 yards as a true freshman during the 2004 season.",

    "Ivory Lee Brown is a former professional American football running back in the "
    "National Football League and the World League of American Football. He played for "
    "the Phoenix Cardinals and the San Antonio Riders. Brown is the uncle of Minnesota "
    "Vikings running back Adrian Peterson.",

    "Big Stone Gap is a 2014 American drama romantic comedy film written and directed by "
    "Adriana Trigiani. The story is set in the actual Virginia town of Big Stone Gap in "
    "the 1970s. It stars Ashley Judd, Patrick Wilson, Whoopi Goldberg, and Jenna Elfman.",

    "Adriana Trigiani is an Italian American best-selling author, television writer, and "
    "film director based in Greenwich Village, New York City. She has written novels "
    "including Big Stone Gap, Lucia Lucia, and The Shoemaker's Wife.",

    "The 1896 Summer Olympics, officially known as the Games of the I Olympiad, were "
    "the first modern international Olympic Games and were held in Athens, Greece. The "
    "Games featured 14 nations and 241 athletes competing in 43 events.",

    "Athens is the capital and largest city of Greece. Athens dominates the Attica "
    "region and is one of the world's oldest cities, with its recorded history spanning "
    "over 3,400 years and its earliest human presence starting somewhere between the "
    "11th and 7th millennia BC.",

    "Retrieval-Augmented Generation (RAG) combines a retriever that selects relevant "
    "documents from an external corpus with a generator that conditions on those "
    "documents to produce an answer. RAG mitigates hallucination and knowledge "
    "obsolescence in large language models.",

    "Continuous representations refer to dense vector encodings of text in a "
    "high-dimensional embedding space. Unlike discrete token sequences, continuous "
    "representations are differentiable and enable gradient-based optimization of "
    "retrieval and generation jointly.",

    "OpenMP is an API specification for parallel programming with shared memory. It "
    "supports multi-platform shared-memory parallel programming in C, C++, and Fortran, "
    "and provides directives such as omp parallel for, reduction, and atomic.",

    "Cython is a programming language and an optimising static compiler that gives "
    "Python the speed of C. It allows writing C extensions for Python using a syntax "
    "similar to Python with optional static typing.",

    "Mistral 7B is a 7-billion-parameter language model released by Mistral AI in "
    "September 2023. It uses grouped-query attention and sliding window attention to "
    "achieve faster inference and lower memory usage than comparable models.",

    "The Lernaean Hydra was a serpentine water monster in Greek and Roman mythology. "
    "It had many heads, and for every head chopped off, the Hydra would regrow two "
    "more. Killing the Hydra was the second of the twelve labors of Heracles.",

    "Single Instruction Multiple Data, or SIMD, is a class of parallel computing "
    "architectures that perform the same operation on multiple data points "
    "simultaneously. Modern CPUs implement SIMD through instruction set extensions "
    "such as SSE, AVX, and ARM NEON.",

    "BLAS stands for Basic Linear Algebra Subprograms. It is a specification of low "
    "level routines for performing common linear algebra operations such as vector "
    "addition, scalar multiplication, dot products, matrix-vector multiplication, and "
    "matrix-matrix multiplication.",

    "Greenwich Village is a neighborhood on the west side of Lower Manhattan in New "
    "York City. It is bounded by 14th Street to the north, Broadway to the east, "
    "Houston Street to the south, and the Hudson River to the west.",

    "Natural Questions, also known as NQ, is a question answering dataset for the "
    "task of answering open-domain questions. The dataset consists of over 300,000 "
    "questions issued to Google search.",

    "HotpotQA is a question answering dataset featuring natural, multi-hop questions, "
    "with strong supervision for supporting facts to enable more explainable question "
    "answering systems.",

    "Wikipedia is a free online encyclopedia, created and edited by volunteers around "
    "the world and hosted by the Wikimedia Foundation. It is the largest and most "
    "popular general reference work on the Internet.",

    "The CLaRa framework, developed by Apple Machine Learning Research, performs "
    "retrieval-augmented generation by encoding documents into a small set of memory "
    "tokens that simultaneously serve retrieval and generation. The retriever and "
    "generator are jointly trained using a single language modeling loss.",

    "The Straight-Through estimator is a technique for back-propagating gradients "
    "through discrete operations. In the forward pass, the discrete operation is "
    "applied normally, but in the backward pass, gradients are passed through as if "
    "the operation were the identity.",
]


SAMPLE_QA_PAIRS: list[dict] = [
    {
        "question": "How many yards did the nephew of Ivory Lee Brown get during his 2004 true freshman season?",
        "answer": "1,925 yards",
        "supporting_doc_indices": [0, 1],
    },
    {
        "question": "Which city is the living place of the director of the romantic comedy Big Stone Gap?",
        "answer": "New York City",
        "supporting_doc_indices": [2, 3],
    },
    {
        "question": "Which city hosted the first modern Olympic Games?",
        "answer": "Athens",
        "supporting_doc_indices": [4, 5],
    },
    {
        "question": "What is RAG?",
        "answer": "Retrieval-Augmented Generation",
        "supporting_doc_indices": [6],
    },
    {
        "question": "What does SIMD stand for?",
        "answer": "Single Instruction Multiple Data",
        "supporting_doc_indices": [12],
    },
]
