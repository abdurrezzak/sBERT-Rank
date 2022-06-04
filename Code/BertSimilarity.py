from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time


class BertSimilarity:
    def __init__(self):
        self.key_phrases= None
        self.abstract = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # self.model_2 = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.kept_embedding = None  # allocated for largest abstract only

    def calculate_embeddings(self, texts):
        embeds1 = self.model.encode(texts)
        # embeds2 = self.model_2.encode(texts)
        
        return embeds1, None

    def cosine_(self, key_phrases, abstract, calculate_for_abstract):
        # calculate cosine similarities between key phrases and the abstract
        self.key_phrases = key_phrases
        self.abstract = abstract

        ts = time.time()
        kp_embeds, kp_embeds_2 = self.calculate_embeddings(self.key_phrases)

        if calculate_for_abstract:
            ab_embed, ab_embed_2 = self.calculate_embeddings([self.abstract])
        else:
            ab_embed, ab_embed_2 = self.kept_embedding

        # print("time", time.time() - ts)

        ts = time.time()
        cosines = cosine_similarity(
            ab_embed,
            kp_embeds
        )
        '''cosines2 = cosine_similarity(
            ab_embed_2,
            kp_embeds_2
        )'''

        # print('time cosine', time.time() - ts)
        # return (cosines[0] + cosines2[0]) /2
        return cosines[0]

# example case below
'''k_p = [
    "computer vision accuracy",
    "computer vision",
    "accuracy"
]

abst = "The ImageNet Large Scale Visual Recognition Challenge is a benchmark in object category classification and \
    detection on hundreds of object categories and millions of images. The challenge has been run annually from 2010 \
    to present, attracting participation from more than fifty institutions. This paper describes the creation of this \
     benchmark dataset and the advances in object recognition that have been possible as a result. We discuss the \
      challenges of collecting large-scale ground truth annotation, highlight key breakthroughs in categorical object\
       recognition, provide a detailed analysis of the current state of the field of large-scale image classification\
        and object detection, and compare the state-of-the-art computer vision accuracy with human accuracy. We\
         conclude with lessons learned in the five years of the challenge, and propose future\
          directions and improvements."

bert_sim = BertSimilarity()
print(bert_sim.cosine_(k_p, abst, True))'''


