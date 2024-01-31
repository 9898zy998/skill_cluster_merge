import json
import torch
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource
print(device)
ori_model_path = "/home/wxd/xiaodong/aliendao-main/dataroot/models/meta-llama/Llama-2-7b-hf"
def compute_cosine(a, b):
    a_vec=a.cpu().numpy()
    b_vec=b.cpu().numpy()
    norms1 = np.linalg.norm(a_vec)
    norms2 = np.linalg.norm(b_vec)
    #print(norms1,norms2)
    dot_products = np.sum(a_vec * b_vec)
    #print(dot_products )
    cos_similarities = dot_products / (norms1 * norms2)
    #print(cos_similarities)
    return cos_similarities

def get_embed(st,llama_model,tokenizer):
    t_input = tokenizer(st, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        last_hidden_state = llama_model(**t_input, output_hidden_states=True).hidden_states[-1].to(device)

    weights_for_non_padding = t_input.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0).to(device)

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1).to(device)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1).to(device)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
    y_0 = torch.mean(sentence_embeddings , dim=0)
    return y_0

gpu=0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
# model config
config_kwargs = {
    "trust_remote_code": True,
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "output_hidden_states": True
}
#ori_model_path="/home/wxd/xiaodong/aliendao-main/dataroot/models/meta-llama/Llama-2-7b-hf"


def train():
    config = AutoConfig.from_pretrained(ori_model_path, **config_kwargs)
    llama_model = AutoModelForCausalLM.from_pretrained(
        ori_model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        revision='main',
        device_map={'': torch.cuda.current_device()},
        use_flash_attention_2=True
    )
    tokenizer = AutoTokenizer.from_pretrained(ori_model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0 
    llama_model.eval()
    #得到json的所有向量
    X=[]
    data_path="raw_clusters_1212.json"
    #data_path="1.json"
    with open(data_path, 'r') as f:
        print(f'read_data from {data_path}')
        train_d = json.load(f)

    zr=np.zeros(4096)
    for i in range(len(train_d)):
        #print(train_d[i])
        a=get_embed(train_d[i],llama_model,tokenizer)
        #print(a)
        X.append(a)
        if(i%10==0):
            print(i)
    print(X)
    # #最大最小正则化
    # from sklearn.preprocessing import MinMaxScaler
    # #区间缩放，返回值为缩放到[0, 1]区间的数据
    # Standard_data=MinMaxScaler().fit_transform(X)
    # print(X,Standard_data)
    #给定一个向量，找出前5个相似的cluster
    # inds = np.argsort(p)[-topk:]
    # print("indexs={}".format(inds.tolist()))
    with open('output.txt', 'w') as f:
        for q in range(len(X)):
            S=[]
            for i in range(len(X)):
                S.append(compute_cosine(X[q],X[i]))
            
            #print(train_d[q],S)
            inds = np.argsort(S)[-3:]####?
            for ind in inds:
                if(S[ind]>0.95 and q!=ind):
                    print("can be merge",train_d[q],train_d[ind], file=f)
                    print("can be merge",train_d[q],train_d[ind],S[ind])
train()