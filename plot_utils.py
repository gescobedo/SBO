import matplotlib.pyplot as plt
import seaborn as sns

from obfuscation import *


def plot_user_score(data_inter,data_user,ff_data,title, method):
    joined = data_inter.join(data_user[["userID","gender"]], on="userID",how="left", lsuffix='', rsuffix='r').dropna()
    joined["user_ff_score"]= joined["itemID"].apply(lambda x: ff_data["FF"].get(x,None))
    joined= joined.dropna()
    mean_user=joined.groupby(["userID","gender"])["user_ff_score"].apply(lambda x:calc_user_stereotyp_pref(x,method)).reset_index()
    sns.displot(data=mean_user, x="user_ff_score",hue="gender")
    plt.title(title)
    plt.show()
    
def plot_user_median_score(data_inter,ff_data, data_user,title):
    joined = data_inter.join(data_user[["userID","gender"]], on="userID",how="left", lsuffix='', rsuffix='r').dropna()
    joined["user_ff_score"]= joined["itemID"].apply(lambda x: ff_data["FF"].get(x,None))
    mean_user=joined.groupby(["userID","gender"])["user_ff_score"].median().reset_index()
    sns.displot(data=mean_user, x="user_ff_score",hue="gender")
    plt.title(title)
    plt.show()                     
