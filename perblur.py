# %%
import pandas as pd
from constants import *
from data_utils import *
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from tqdm import tqdm

# %%

dataset_name = "ml-1m" # or lfm-100k
obf_ratio = 0.1 # max 10% of preferences
kth = 200  # 500 #for lfm
N_NEIGBORS = 50  # 100 for lfm

imputate = True  # imputate or False-->remove 

if imputate:
    obf_dataset_name = f"{dataset_name}_imputate_{obf_ratio}_perblur_perblur_perblur"
else:
    obf_dataset_name = f"{dataset_name}_remove_{obf_ratio}_perblur_perblur_perblur"
out_dataset_dir = f"{ROOT_DIR_STR}/obf_baseline"
import os

if not os.path.exists(f"{out_dataset_dir}/{obf_dataset_name}"):
    os.mkdir(f"{ROOT_DIR_STR}/obf_baseline/{obf_dataset_name}")
train_data, valid_data, test_data, inclination_data, user_data, dataset_name = (
    read_dataset_to_obfuscate(f"{ROOT_DIR_STR}/obfuscation/{dataset_name}")
)
dataset_src = pd.concat([train_data, valid_data], ignore_index=True)
interaction_matrix, user_info, token2item, token2user = transform_dataframe_to_sparse(
    dataset_src
)

# %%


X = interaction_matrix.copy()
user_knn = NearestNeighbors(n_neighbors=N_NEIGBORS)
user_knn.fit(X)


# %%
distances, indices = user_knn.kneighbors(X)
# %%
# distances.shape
X = interaction_matrix.copy()
user_counts = X
userlist = {}
user_rows = {}


for user, indices_user in tqdm(enumerate(indices)):
    mask = user_counts[user].nonzero()[1]
    row = np.sum(X[indices[user]], axis=0)
    row[0, mask] = 0
    user_rows[user] = np.asarray(row)[0]
for user, row in user_rows.items():
    userlist[user] = np.argpartition(row, kth=-kth)[-kth:]
# %%
import pickle

pickle.dump(
    userlist,
    open(
        f"{out_dataset_dir}/{obf_dataset_name}/ml-{kth}-personalized-counts.pkl", "wb"
    ),
)
# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import scipy.stats as ss

cv = StratifiedKFold(n_splits=10)
coefs = []
X = interaction_matrix.copy()
# sp.load_npz(f"{ROOT_DIR_STR}/obfuscation/ml-1m/interactions.npz")
avg_coefs = np.zeros(X.shape[1])
T = user_data["gender"].values  # .apply(lambda x: 1 if x=="M" else 0)
certainty = np.zeros(X.shape[1])
random_state = np.random.RandomState(0)
for train, test in cv.split(X, T):
    x, t = X[train], T[train]
    model = LogisticRegression(penalty="l2", random_state=random_state)
    model.fit(x, t)
    # rank the coefs:
    ranks = ss.rankdata(model.coef_[0])
    coefs.append(ranks)
    # print(len(model.coef_[0]),len(X_train[0]))
    avg_coefs += model.coef_[0]
    x_test = X[test]

    class_prob = np.max(model.predict_proba(x_test), axis=1)
    # correct, so that 1 means the classifier is very sure and 0 means it is not sure
    class_prob -= 0.5
    class_prob *= 2
    # certainty[test] = class_prob
    # set certainty to 0 for all missclassifications:
    t_pred = model.predict(x_test)
    # t_test = T[test]
    # for index, (pred, target) in enumerate(zip(t_pred, t_test)):
    #    #print(pred, target, index, test)
    #    if pred != target:
    #        certainty[test[index]] = 0
# %%

coefs = np.average(coefs, axis=0)
coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
coefs = np.asarray(list(sorted(coefs)))
# %%
values = coefs[:, 2]
index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
top_male = index_zero[0][0]
top_female = index_zero[0][-1]
L_m = coefs[:top_male, 1]

L_f = coefs[coefs.shape[0] - top_female :, 1]
L_f = list(reversed(L_f))
# %%
indicative = {"L_m": L_m, "L_f": L_f}

pickle.dump(
    indicative, open(f"{out_dataset_dir}/{obf_dataset_name}/ml-indicative.pkl", "wb")
)
# %%
X = (
    interaction_matrix.copy()
)  # sp.load_npz(f"{ROOT_DIR_STR}/obfuscation/ml-1m/interactions.npz")
user_ranks_sums = np.asarray(np.sum(X, axis=1)).flatten()
user_ranks_max_10 = np.asarray(np.floor(0.1 * np.sum(X, axis=1))).flatten()
user_ranks_max_5 = np.asarray(np.floor(0.05 * np.sum(X, axis=1))).flatten()
# %%
# imputate
if imputate:
    obf_matrix = X.copy()
    gender = user_data["gender"].values
    user_ranks_max = user_ranks_max_10 if obf_ratio == 0.1 else user_ranks_max_5
    topk = 50
    for user, user_row in enumerate(X):
        q = 0
        u_gender = gender[user]
        L_indicative = L_m if u_gender == "F" else L_m
        L_indicative = L_indicative[:topk]
        inter = np.intersect1d(userlist[user], L_indicative)
        limit_num_obf = int(user_ranks_max[user])
        if len(inter > 0) and limit_num_obf > 0:
            if len(inter) < limit_num_obf:
                obf_matrix[user, inter] = 1
            else:
                obf_matrix[user, inter[:limit_num_obf]] = 1
# %%
# removal
else:
    obf_matrix = X.copy()
    gender = user_data["gender"].values
    user_ranks_max = user_ranks_max_10 if obf_ratio == 0.1 else user_ranks_max_5
    topk = 50
    for user, user_row in enumerate(X):
        q = 0
        u_gender = gender[user]
        L_indicative = L_f if u_gender == "F" else L_m
        L_indicative = L_indicative[:topk]

        inter = np.intersect1d(user_row.nonzero()[1], L_indicative)
        limit_num_obf = int(user_ranks_max[user])
        if len(inter > 0) and limit_num_obf > 0:
            if len(inter) < limit_num_obf:
                obf_matrix[user, inter] = 0
            else:
                obf_matrix[user, inter[:limit_num_obf]] = 0
# %%
item_mapping = (
    token2item  # pd.read_csv(f"{ROOT_DIR_STR}/obfuscation/ml-1m/item_mapping.csv")
)
user_mapping = (
    token2user  # pd.read_csv(f"{ROOT_DIR_STR}/obfuscation/ml-1m/user_mapping.csv")
)
item2token = pd.Series(item_mapping.index.values)
user2token = pd.Series(user_mapping.index.values)
data_obf = []
for iduser, user_row in enumerate(obf_matrix):
    data_obf.append(
        [user2token[iduser], list(item2token.loc[user_row.nonzero()[1]].values)]
    )

df_obf = pd.DataFrame(data_obf, columns=["userID", "itemID"])
df_obf = df_obf.explode("itemID").reset_index(drop=True)
# df_obf.to_csv(f"{out_dataset_dir}/{obf_dataset_name}/data.csv",index=False)
df_obf = df_obf.merge(user_data, on="userID", how="left")
train_data_obf, valid_data_obf = split_by_inter_ratio(df_obf)
save_recbole_data(
    train_data_obf, valid_data_obf, test_data, f"{out_dataset_dir}/{obf_dataset_name}"
)
save_csr_matrix(f"{out_dataset_dir}/{obf_dataset_name}", df_obf)
# %%
