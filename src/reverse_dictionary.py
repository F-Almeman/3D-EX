import sys
import csv
import pandas as pd
import gdown
import numpy as np
import torch
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer, util
from transformers import BartTokenizer, BartForConditionalGeneration, AdapterType, TrainingArguments, AdapterTrainer

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
  
if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-m', '--model', help='Model type', default='all-MiniLM-L6-v2')
  parser.add_argument('-d', '--data', help='Input file (WORD, DEFINITION)', required=True)
  parser.add_argument('-wi','--word_instruction',help='Word instruction', default="no")
  parser.add_argument('-di','--def_instruction',help='Definition instruction', default="no")

  # Parse the argument
  args = parser.parse_args()

  df = pd.read_csv(args.data ,engine='python', na_values = [''], keep_default_na=False)
  f = open('reverse_dictionary_output.txt', 'w')

  if args.model == "instructor":
    embedder = INSTRUCTOR('hkunlp/instructor-large')
  else:
    embedder = SentenceTransformer('sentence-transformers/'+ args.model)

  corpus = df.WORD.values

  if args.word_instruction != "no":
    words_instructions = []
    for w in corpus:
      words_instructions.append([args.word_instruction, w])   
    corpus_embeddings = embedder.encode(words_instructions, convert_to_tensor=True)
  else: 
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

  # Query sentences:
  queries = df.DEFINITION.values

  # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
  top_k = 10

  results = []

  for idx in range(len(df)):
    query = df.DEFINITION.iloc[idx]
    gold_word = df.WORD.iloc[idx]
    
    if args.def_instruction != "no":
      query_embedding = embedder.encode([[args.def_instruction, query]], convert_to_tensor=True)
    else:
      query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n", file=f)
    print(f"Query: word={gold_word}, with definition: ", query, file=f)
    print(f"\nTop {top_k} results:", file=f)

    hits = []
    for score, idx in zip(top_results[0], top_results[1]):
      predicted_word = corpus[idx]
      print(predicted_word, "(Score: {:.4f})".format(score), f"| With definition: {queries[idx]}", file=f)
      if predicted_word == gold_word:
        hits.append(1)
      else:
        hits.append(0)
    results.append(hits)
    
  print("\n\n======================\n\n", file=f)
  print(f"\nMean reciprocal rank is {mean_reciprocal_rank(results)}", file=f)
    
  f.close() 
