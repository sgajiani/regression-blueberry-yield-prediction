import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import Dark2
plt.style.use("dark_background")
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False

dark2 = sns.color_palette('Dark2')

def author():
    return 'Samir Gajiani'


def get_file_path(filename):
    data_folder_path = os.environ.get('DATA_FOLDER_PATH')
    file_path = os.path.join(data_folder_path, filename)
    return file_path



def check_missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_pct = round(df.isnull().sum() / len(df) * 100, 2)
    df_mis_val = pd.concat((df.dtypes, mis_val, mis_val_pct), axis=1).reset_index()
    df_mis_val = df_mis_val.rename(columns={'index': 'col', 0: 'dtype', 1:'miss_val', 2:'pct_miss_val'})
    df_mis_val = df_mis_val[df_mis_val.iloc[:,2] != 0]
    df_mis_val = df_mis_val.sort_values('pct_miss_val', ascending=False).reset_index(drop=True)
    return df_mis_val



def get_unique_val_count(cat_col):
    return round(cat_col.value_counts(dropna=False) / len(cat_col) * 100, 2)


def get_unique_cat_val(df):
    data_rows = []
    cols = [col for col in df.columns if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'object']
    for col in cols:
        unique_dict = {'col': col, 'unique': df[col].unique()}
        data_rows.append(unique_dict)
    return data_rows


def get_max_class(df):
    columns = ['col', 'class', 'pct']
    max_class_rows = []

    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'object':
            sr_pct = round(df[col].value_counts() / len(df[col]) * 100, 2)
            max_class, max_pct = sr_pct.idxmax(), sr_pct.max()
            max_dict = {'col': col, 'class': max_class, 'pct': max_pct}
            max_class_rows.append(max_dict)

    df_max_class = pd.DataFrame(max_class_rows, columns=columns)            
    return df_max_class



def imbalance_check(df):
    df_miss_val = check_missing_values(df)
    df_max_class = get_max_class(df)
    df_imb = pd.merge(df_max_class, df_miss_val, on='col', how='left')
    df_imb['pct_miss_val'] = df_imb['pct_miss_val'].fillna(0.0)
    df_imb['impute_mode_pct'] = df_imb['pct'] + df_imb['pct_miss_val']
    df_imb = df_imb[['col', 'class', 'pct', 'pct_miss_val', 'impute_mode_pct']]
    df_imb = df_imb.sort_values('impute_mode_pct', ascending=False).reset_index(drop=True)
    return df_imb



def get_outliers(col):
    iqr = col.quantile(0.75) - col.quantile(0.25)
    lower_bridge = col.quantile(0.25) - (iqr * 1.5)
    upper_bridge = col.quantile(0.75) + (iqr * 1.5)
    out_lower = [val for val in col if val < lower_bridge]
    out_upper = [val for val in col if val > upper_bridge]
    outlier = len(out_lower) + len(out_upper)
    outlier_pct = round(outlier / len(col) * 100, 2)
    return outlier_pct


def ordinal_encode_test(input_val, features): 
    feature_val = list(np.arange(len(features)))
    feature_key = features
    feature_dict = dict(zip(feature_key, feature_val))
    encoded_val = feature_dict[input_val]
    return encoded_val


def get_prediction(data, model):
    return model.predict(data)


def plot_cat_count(sr_data, x_label, y_label, title, index_on_x=1, xticklbl_rotate=0):
    fig = plt.figure(figsize=(8, 5))
    if index_on_x == 1:
        ax = sns.barplot(x=sr_data.index, y=sr_data.values, hue=sr_data.values, legend=False, palette='Dark2')
    else:
        ax = sns.barplot(x=sr_data.values, y=sr_data.index, hue=sr_data.values, legend=False, palette='Dark2')
    ax.set_xlabel(x_label, fontdict={'fontsize': 10})
    ax.set_ylabel(y_label, fontdict={'fontsize': 10})
    ax.set_title(title, fontdict={'fontsize': 10, 'fontweight': 'bold'})
    ax.tick_params(labelsize=10)
    if xticklbl_rotate > 0:
        ax.set_xticks(sr_data.index)
        ax.set_xticklabels(labels=sr_data.index, rotation=xticklbl_rotate)
    plt.tight_layout()
    plt.show()


def plot_num_dist(sr_data, x_label, y_label, title):
    color = random.choice(dark2)
    fig = plt.figure(figsize=(8, 5))
    ax= sns.histplot(sr_data, bins=30, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()
    

def plot_cat_pie(sr_data, title):
    plt.figure(figsize=(16, 8))
    plt.pie(sr_data.values, labels=sr_data.index, startangle=50, autopct='%1.1f%%', colors=dark2)
    center_circle = plt.Circle((0,0), 0.7, fc='black')
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    plt.title(title)
    plt.axis('equal')
    plt.legend(prop={'size': 8},loc='upper left')
    plt.show()



def plot_cat_count_by_var(ser_data, ser_hue, x_label, y_label, title, xticklbl_rotate=0):
    plt.figure(figsize=(20, 5))
    colors = dark2[:2]
    sns.countplot(x=ser_data, hue=ser_hue, palette=colors)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=xticklbl_rotate)
    plt.show()


def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.1)

    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = ['#9E3F00', '#eb5e00', '#ff781f']
    inner_colors = ['#ff6905', '#ff8838', '#ffa66b']

    ax[0].pie(
        outer_sizes,
        colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.4, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'}, 
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='black', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        hue=y,
        legend=False,
        palette='YlOrBr_r', orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i + 0.1, str(v))

    plt.setp(ax[1].get_yticklabels())
    plt.setp(ax[1].get_xticklabels())
    ax[1].set_xlabel(col)
    ax[1].set_ylabel('count')

    f.suptitle(f'{title_name}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()



