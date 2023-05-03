import numpy as np
import nltk
import nltk.data
import spacy
import os
import tensorflow_hub as hub
import copy
import spacy 
from tqdm import tqdm
# from rouge import Rouge
import sys

nltk.download('punkt')

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

src_file = sys.argv[1]
tar_file= sys.argv[2]
k=int(sys.argv[3])

def read_file(path, file_name, read_lead_only=False, read_first_doc=False):
    f = open(os.path.join(path, file_name),"r")
    lines = f.readlines()
    src_list = []
    tag="story_separator_special_tag"
    for line in lines:
        # remove tag; uncomment below for baseline
        line = line.replace(tag, "")
        # tokenzie line to sentences
        sent_list = sent_detector.tokenize(line.strip())
        src_list.append(sent_list)
    return src_list



src_list = read_file('', src_file)[:100]
with open(tar_file, 'r') as file:
    target = file.read().replace('\n', '')

def get_use_embedding(sentence):
  embedding = use_embed([sentence]) 
  embedding = embedding.numpy()
  return embedding[0]


def cosine_dist(A1,B1):
  A = np.array(A1)
  B = np.array(B1)
  mag_A = np.linalg.norm(A)
  mag_B = np.linalg.norm(B)

  dot_AB = np.dot(A,B)

  return (dot_AB)/(mag_A*mag_B)


def compute_cosine_mat(em_li, thres):
  length = len(em_li)
  cos_mat = np.zeros([length, length])
  conn_list = []
  for i in range(length):

    for j in range(i+1,length):

      cos_mat[i][j] = cosine_dist(em_li[index_map[i]],em_li[index_map[j]])
      cos_mat[j][i] = cos_mat[i][j]
      if(cos_mat[i][j]>thres):
        # print(i," --> ",j)
        conn_list.append((index_map[i],index_map[j]))
  return cos_mat, conn_list



def dfs(cur):
  if(flag[cur]==1):
    return
  # print(cur)
  flag[cur] = 1
  for key in graph_con[cur]:
    if(flag[key]==0):
      dfs(key)
  return


emb_doc = {}
conn_list_All = []
N_nodes = 0
sen_list=[]
ki=0
ct=0
index_map = {}     #  1  -->  1_9  
inv_index_map = {}  #  1_9 -->  1
for doc in src_list:
  s_li = []
  kj=0
  for sen in doc:
    emb_doc[str(ki)+"_"+str(kj)] = get_use_embedding(sen)

    index_map[ct] = str(ki)+"_"+str(kj)
    inv_index_map[str(ki)+"_"+str(kj)] = ct
    sen_list.append(sen)
    kj+=1
    ct+=1
    N_nodes+=1
  ki+=1



Ae, c_li = compute_cosine_mat(emb_doc,0.05)
graph_con = {}
flag = {}
for key in inv_index_map:
  graph_con[key] = []
  flag[key] = 0
for tup in c_li:
  graph_con[tup[0]].append(tup[1])
  graph_con[tup[1]].append(tup[0])


Are = copy.deepcopy(Ae)
for i in range(len(Are)):
  for j in range(len(Are[i])):
    if(i==j):
      Are[i][j] = len(graph_con[index_map[i]])
    elif(Are[i][j]>=0.05):
      Are[i][j]=-1
    else:
      Are[i][j]=0


Are = Are.astype('int64')
w, v = np.linalg.eig(Are)
v = np.real_if_close(v, tol=1)
w= np.real_if_close(w, tol=1)
eigen_asc = []

for i in range(len(w)):
  eigen_asc.append((w[i],i))
eigen_asc.sort(reverse=True)

w=np.sort(w)


li_k_vec = []
ind_eigv = [] 

for i in range(k):
  ind_eigv.append(eigen_asc[i][1])
  li_k_vec.append(v[eigen_asc[i][1]])

N_coor = np.array(li_k_vec).T

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
kmeans.fit(N_coor)

cluster_dict={}
for i in range(k):
  cluster_dict[i]=[]

for i in range(len(sen_list)):
  k_num=kmeans.labels_[i]
  cluster_dict[k_num].append(sen_list[i])



import spacy 
spacy_=spacy.load("en_core_web_sm")

def pos_tag(sentences):
  out=[]
  for sentence in sentences:
    sentence =sentence.replace("/","")
    word_dict=spacy_(sentence)
    temp=[]
    for word in word_dict:
      tagg=word.text+"/"+word.tag_
      temp.append(tagg)
    out.append(' '.join(temp))
  return out


import takahe

def compress_cluster(sentences, nb_words):
    compresser = takahe.word_graph(sentences, nb_words = nb_words,lang = 'en',punct_tag = "." )
    candidates = compresser.get_compression(10)
    reranker = takahe.keyphrase_reranker(sentences,candidates,lang='en')
    reranked_candidates = reranker.rerank_nbest_compressions()
    if len(reranked_candidates)>0:
        _, path = reranked_candidates[0]
        result = ' '.join([u[0] for u in path])
    else:
        result=' '
    # score, path = reranked_candidates[0]
    # result = ' '.join([u[0] for u in path])
    return result

summary=[]
for i in tqdm(range(k)):
  summary.append(compress_cluster(pos_tag(cluster_dict[i]),5))
ans="".join(summary)



# from rouge import Rouge
# rouge = Rouge()
# sc=rouge.get_scores(ans,target)

filename=sys.argv[1]+"_summary"
text_file = open(filename, "w")
text_file.write(ans)
text_file.close()
