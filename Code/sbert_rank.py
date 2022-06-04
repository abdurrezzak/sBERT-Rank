# coding: utf-8
import numpy as np
import time
from string import punctuation
import np_chunker
import stanza
import pandas as pd
from snowballstemmer import stemmer
import multiprocessing as mp
from BertSimilarity import BertSimilarity
from nltk.stem import PorterStemmer
import nltk

bert_sim = BertSimilarity()
'''stanza.download('en')
stanza.download('ar')'''

nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True, tokenize_pretokenized=True)
nlp_ar = stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True, tokenize_pretokenized=True)

ps = PorterStemmer()
ar_stemmer = stemmer('arabic')

dataset_ids = []
partial_no_partial = []
max_n_kps = []
max_n_toks = []
recalls = []
precisions = []
f1scores = []
results = []
all_docs_list = {}
abstracts_list = {}
key_phrases_list = {}


def dataset_preparer(dataset_ids_list, ds_size=None):

    print("Preparing datasets")

    for d_id in dataset_ids_list:

        if d_id == 7:
            df = pd.read_csv(f"../../Datasets/7/data_fully.csv")
        elif d_id == 1:  # arabic
            df = pd.read_csv(f"../../Datasets_ar/1/data.csv", encoding='utf-8')
        else:  # 6, 8, 9
            df = pd.read_csv(f"../../Datasets/{d_id}/data.csv")

        if ds_size == None:
            abstracts_list[d_id] = df["abstract"]
            key_phrases_list[d_id] = df['keys']
        else:
            abstracts_list[d_id] = df["abstract"][0:min(ds_size, len(df))]
            key_phrases_list[d_id] = df['keys'][0:min(ds_size, len(df))]

        all_docs = []

        print("Preparing dependencies for dataset", d_id)

        for i, a in enumerate(abstracts_list[d_id]):

            if d_id == 1:
                strr = '\n\n'.join(nltk.sent_tokenize(a))
                doc = nlp_ar(strr)
            else:
                doc = nlp('\n\n'.join(a.split('.')))

            all_docs.append(doc)

        all_docs_list[d_id] = all_docs

    print("All dependencies are ready!")


def calculate(p_id, max_num_of_tokens, partial_points, max_num_of_keys,\
    dataset_id, abstracts_, key_phrases_, all_docs_):

    start_index = 0
    key_phrases = key_phrases_
    sentences_list = abstracts_
    # all docs keep a list of document lists where each document list is for an abstract
    all_docs = all_docs_

    # number of sentences are kept
    n_s = len(sentences_list)

    # number of sentences from stanza pipeline should be the same with the original data
    n_s_stanza = len(all_docs)
    print('Number of abstracts:', n_s, 'Number of abstracts stanza:', n_s_stanza)

    # also initialize a list for NPs and their tags in sentences
    noun_phrases = []
    key_phrases_dat = []
    recs = []
    precs = []
    f1s = []

    # we iterate through the sentences and use our chunker
    valid = 0
    avg_recall = 0
    avg_pre = 0
    avg_f1 = 0

    stats_us = []
    stats_thm = []

    print("Starting to calculations")
    start_time = time.time()

    for i, abstract in enumerate(all_docs):

        kp_cosine_pairs_ours = {}

        tss_ = time.time()

        # calculating abstract embedding to not calculate it repeatedly
        bert_sim.kept_embedding = bert_sim.calculate_embeddings([sentences_list[i + start_index]])

        for s_id, sentence in enumerate(abstract.sentences):

            current_sentence = sentence
            current_sentence = ' '.join([w.text for w in current_sentence.words])

            c_np, c_obi_tag = np_chunker.noun_phraser(sentence)
            c_np_v2, _ = np_chunker.noun_phraser_v2(sentence)
            c_np.extend(c_np_v2)

            # make it all lowercase and clear from both sides
            c_np = [c.lower().strip().strip(punctuation) for c in c_np]

            c_np = list(dict.fromkeys(c_np))

            if len(c_np) > 0:
                tss = time.time()

                # print("TIME for abs", time.time() - tss)
                tss = time.time()

                sim_with_sen = np.array(bert_sim.cosine_(c_np, current_sentence, True))
                # print("TIME for sent", time.time() - tss)

                sen_with_abs = np.array(bert_sim.cosine_([current_sentence], sentences_list[i + start_index], False))

                sim_score = sim_with_sen * sen_with_abs # * (1.2 if len(c_np.split()) == 2 else 1.0)

                for j, c in enumerate(c_np):

                    coeff = (1.0 if len(c.split()) == 2 else 1.0)

                    if c in kp_cosine_pairs_ours:
                        kp_cosine_pairs_ours[c] = max(sim_score[j] * coeff, kp_cosine_pairs_ours[c])
                    else:
                        kp_cosine_pairs_ours[c] = sim_score[j] * coeff

        # print("TIME for all comparison", time.time() - tss_)

        current_nps = kp_cosine_pairs_ours.keys()
        # we do this because pd returns a string
        current_keys = key_phrases[i + start_index].strip('][').replace("'", "").split(", ")
        current_keys = [c.lower().strip().strip(punctuation) for c in current_keys]

        current_keys = list(dict.fromkeys(current_keys))

        inside_current_keys = []
        for ck in current_keys:
            flg = True
            if flg and ck in sentences_list[i]:
                inside_current_keys.append(ck)
                flg = False

        inside_current_keys = list(dict.fromkeys(inside_current_keys))
        inside_current_keys_len = len(inside_current_keys)

        stats_us.append(len(current_nps))
        stats_thm.append(len(inside_current_keys))

        if inside_current_keys_len > 0:

            cosines_their_kps = bert_sim.cosine_(inside_current_keys, sentences_list[i+start_index], False)
            kp_cosine_pairs_originals = {}

            for j, c_k in enumerate(inside_current_keys):
                kp_cosine_pairs_originals[c_k] = cosines_their_kps[j]

            kp_cosine_pairs_originals = sorted(kp_cosine_pairs_originals.items(), key=lambda item: item[1], reverse=True)
            kp_cosine_pairs_ours = sorted(kp_cosine_pairs_ours.items(), key=lambda item: item[1], reverse=True)

            kp_cosine_pairs_ours = kp_cosine_pairs_ours[0:min(len(kp_cosine_pairs_ours), max_num_of_keys)]

            tp = 0
            temp_cur_keys = inside_current_keys
            for prdd in kp_cosine_pairs_ours:
                prd = prdd[0]
                if dataset_id != 1:
                    prd = ' '.join([ps.stem(c_np_w) for c_np_w in prd.split()])
                else:
                    prd = ' '.join([ar_stemmer.stemWord(c_np_w) for c_np_w in prd.split()])

                f_ = True
                j = 0
                while f_ and j < len(temp_cur_keys):
                    org = temp_cur_keys[j]

                    if dataset_id != 1:
                        org = ' '.join([ps.stem(kw_org_w) for kw_org_w in org.split()])
                    else:
                        org = ' '.join([ar_stemmer.stemWord(kw_org_w) for kw_org_w in org.split()])

                    if not partial_points:
                        if prd == org:
                            tp += 1
                            temp_cur_keys[j] = "8888888888888"
                            f_ = False
                    else:
                        if f_ and (prdd[0] in temp_cur_keys[j] or temp_cur_keys[j] in prdd[0]):
                            tp += 1
                            temp_cur_keys[j] = "8888888888888"
                            f_ = False
                    j += 1
        
            # print('------', tp, len(kp_cosine_pairs_originals), len(kp_cosine_pairs_ours))

            rec = tp / (1 if len(kp_cosine_pairs_originals) < 1 else len(kp_cosine_pairs_originals))

            # rec = inside_current_keys_len/(1 if len(kp_cosine_pairs_originals) < 1 else len(kp_cosine_pairs_originals))

            prec = tp / (1 if min(max_num_of_keys, len(kp_cosine_pairs_ours)) < 1 else min(max_num_of_keys, len(kp_cosine_pairs_ours)))
            f1 = (0 if rec + prec == 0 else 2 * (rec * prec) / (rec + prec))
            
            
            # print(rec, prec, f1)
            recs.append(rec)
            precs.append(prec)
            f1s.append(f1)
            noun_phrases.append(kp_cosine_pairs_ours)
            key_phrases_dat.append(kp_cosine_pairs_originals)
            avg_recall += rec
            avg_pre += prec
            avg_f1 += f1

        else:
            rec = 0
            prec = 0
            f1 = 0
            # print(rec, prec, f1)
            recs.append(rec)
            precs.append(prec)
            f1s.append(f1)
            noun_phrases.append([])
            key_phrases_dat.append([])
            avg_recall += rec
            avg_pre += prec
            avg_f1 += f1

        if i % int(n_s/5) == 0:
            print("{:.2f}".format(100*i/n_s), "% of calculations are done...")

    print("Time taken for calculations:", time.time() - start_time)

    avg_pre /= n_s
    avg_recall /= n_s
    final_f1 = (0 if avg_recall + avg_pre == 0 else 2 * (avg_recall * avg_pre) / (avg_recall + avg_pre))

    dat = pd.DataFrame((list(zip(noun_phrases, key_phrases_dat, recs, precs, f1s))),
                       columns=['NPs', 'KPs', 'recall', "precision", "f1"])

    partial_or_no = "partial" if partial_points else "no_partial"

    # this is for dataset 7
    dat.to_csv(f"../../Datasets/{dataset_id}/summary_{max_num_of_keys}_first_{max_num_of_tokens}_token_{partial_or_no}"
               f"sim_alpha_4_stemmed.csv", encoding='utf-8')

    # this is for dataset 6 and 8
    # dat.to_csv(f"../../Datasets/8/summary_{max_num_of_keys}_{partial_or_no}.csv")

    stats_us = np.array(stats_us)
    stats_thm = np.array(stats_thm)

    print(f"summary_dataset_{dataset_id}_keys_{max_num_of_keys}_first_{max_num_of_tokens}_token_{partial_or_no}")
    print("mean # of nps:", stats_us.mean())
    print("mean # of keys:", stats_thm.mean())
    print("std # of nps:", stats_us.std())
    print("std # of keys:", stats_thm.std())
    print("Recall:", "{:.2f}".format(100*avg_recall))
    print("Precision:", "{:.2f}".format(100*avg_pre))
    print("F1:", "{:.2f}".format(100*final_f1))
    print('val:', valid/n_s)

    ll = [p_id, [dataset_id, partial_points, max_num_of_keys, max_num_of_tokens, round(100*avg_recall, 2),
                 round(100*avg_pre, 2), round(100*final_f1, 2)]]
    return ll


def result_appender(r):

    r = r[1]
    global dataset_ids
    global partial_no_partial
    global max_n_kps
    global max_n_toks
    global recalls
    global precisions
    global f1scores

    dataset_ids.append(r[0])
    partial_no_partial.append(r[1])
    max_n_kps.append(r[2])
    max_n_toks.append(r[3])
    recalls.append(r[4])
    precisions.append(r[5])
    f1scores.append(r[6])


if __name__ == '__main__':

    ts = time.time()

    d_list = [6, 7, 8, 9] # representing INSPEC, NUS, DUC and SemEval2017
    
    dataset_preparer(d_list, None)

    pool = mp.Pool(min(mp.cpu_count(), 6))

    for d_id in d_list:
        for partial_point in [False]:
            for max_num_of_key in [5, 10, 15]:
                pool.apply_async(calculate,
                                 args=(0, 9999999999, partial_point, max_num_of_key, d_id,
                                       abstracts_list[d_id], key_phrases_list[d_id], all_docs_list[d_id]),
                                 callback=result_appender)

    pool.close()
    pool.join()

    finals = pd.DataFrame((list(zip(dataset_ids, partial_no_partial, max_n_kps, max_n_toks, recalls, precisions, f1scores)))
                          , columns=['Dataset_id', 'Partial', '#kp', "#token", "recall", "precision", "f1"])

    finals.to_csv(f'Results/all_results_{time.time()}_sim_alpha_4_stemmed.csv')
    print("All calculations took", time.time()-ts)
