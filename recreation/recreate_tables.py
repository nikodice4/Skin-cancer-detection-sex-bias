# This script aims to recreate the plots and tables from the paper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu

def create_diagnosis_dist():
    df = pd.read_csv("data/metadata/fixed_metadata.csv")
    df = df[df["gender"].notna()]
    df = df.drop_duplicates("lesion_id", keep="first")
    # Remove rows with missing 'gender' and drop duplicate 'lesion_id'
    df = df.sort_values(["lesion_id"]) 

    cancer_conditions = ["BCC", "MEL", "SCC"]
    no_cancer_conditions = ["ACK", "NEV", "SEK"]

    def label_diagnostic(diagnostic):
        if diagnostic in cancer_conditions:
            return 'Skin cancer'
        elif diagnostic in no_cancer_conditions:
            return 'Skin disease'


    # Applying the function to the 'diagnostic' column to create a new 'cancer_label' column
    df['cancer_label'] = df['diagnostic'].apply(label_diagnostic)

    # Define a custom order for the x-axis, mapping diagnostics to labels
    custom_order = [
        'ACK (Skin disease)', 'NEV (Skin disease)', 'SEK (Skin disease)', 
        'BCC (Skin cancer)', 'MEL (Skin cancer)', 'SCC (Skin cancer)'
    ]

    # Prepare a pivot table for plotting
    df['diagnostic_info'] = df['diagnostic'] + " (" + df['cancer_label'] + ")"
    gender_diagnostic_counts = df.groupby(["gender", "diagnostic_info"]).size().reset_index(name="count")

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=gender_diagnostic_counts, x="diagnostic_info", y="count", hue="gender", palette="flare", order=custom_order)
    plt.title("Sex distribution of diagnoses", fontsize=15)
    plt.xlabel("Diagnoses", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.xticks(rotation=45)
    legend = plt.legend(title="Sex")
    sns.set_style("whitegrid")
    plt.tight_layout() 

    plt.savefig('recreation/figures/diagnosis_distribution.png')

def gen_plot_panel(x_var, y_var, data, min_jitter, max_jitter):
    # Define the plotting function similar to 'gen_plot_panel' from https://github.com/e-pet/adni-bias/blob/main/analysis.py
    # Perform OLS regression and get the results
    mdl = ols(formula=f'{y_var} ~ {x_var}', data=data)
    results = mdl.fit()
    slope = results.params[1]
    slope_std = results.bse[1]
    p_value = results.pvalues[1]
    # Bonferroni correction is not required here as we only do one test

    # Plot regression line and annotate with stats
    ax = plt.gca()
    slope_str = f"Slope (m): {slope:.3f} ± {slope_std:.3f}\n"
    p_str = f"p-value (corr): {p_value:.4f}\n"
    mu_str = f"Mean (μ): {data[y_var].mean():.3f} ± {data[y_var].std():.3f}"
    text_str = slope_str + p_str + mu_str
    t = ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9, verticalalignment='top')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))
    
    # Add the regression line
    sns.regplot(x=x_var, y=y_var, data=data, ax=ax, scatter=False, fit_reg=True, color="black", line_kws={'linewidth': 1})
    # Jitter the data points for scatter plot
    data[x_var] = data[x_var] + np.random.uniform(min_jitter, max_jitter, data[x_var].shape)
    # Plot the jittered scatter points
    sns.scatterplot(x=x_var, y=y_var, data=data, hue='variation_hue', s=30, ax=ax, palette="viridis", linewidth=0, alpha=0.7, legend=None)

def regression_plot_lr():

    data = pd.read_csv('data/results/lr_results/lr_results_aug.csv')
    # Adjusted version of the plotting code to include Bonferroni correction and statistical details for both plots

    # Set up the subplot grid
    plt.figure(figsize=(9, 7))
    ax0 = plt.subplot(2, 2, 1)
    ax1 = plt.subplot(2, 2, 2)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(2, 2, 4)

    # Make the first plot
    plt.sca(ax0)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='accuracy_female', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)

    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks'

    ax0.set_xlabel('')
    ax0.set_ylabel('ACC f')
    ax0.set_ylim(0.45, 1)  # Set y-axis limits

    # Make the second plot
    plt.sca(ax1)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='auroc_female', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)
    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks

    ax1.set_xlabel('')
    ax1.set_ylabel('AUROC f')
    ax1.set_ylim(0.45, 1)  # Set y-axis limits

    # Make the third plot
    plt.sca(ax2)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='accuracy_male', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)
    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks'

    ax2.set_xlabel('Ratios')
    ax2.set_ylabel('ACC m')
    ax2.set_ylim(0.45, 1)  # Set y-axis limits

    # Make the fourth plot
    plt.sca(ax3)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='auroc_male', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)
    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks

    ax3.set_xlabel('Ratios')
    ax3.set_ylabel('AUROC m')
    ax3.set_ylim(0.45, 1)  # Set y-axis limits

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.suptitle('LR ACC & AUROC for female and male')

    plt.savefig('recreation/figures/lr_regression_plot.png')

def regression_plot_cnn():

    data = pd.read_csv('data/results/cnn_results/model_training_results_cnn_5_10may_R.csv')
    # Adjusted version of the plotting code to include Bonferroni correction and statistical details for both plots

    # Set up the subplot grid
    plt.figure(figsize=(9, 7))
    ax0 = plt.subplot(2, 2, 1)
    ax1 = plt.subplot(2, 2, 2)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(2, 2, 4)

    # Make the first plot
    plt.sca(ax0)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='accuracy_female', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)

    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks'

    ax0.set_xlabel('')
    ax0.set_ylabel('ACC f')
    ax0.set_ylim(0.45, 1)  # Set y-axis limits

    # Make the second plot
    plt.sca(ax1)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='ROC_male', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)
    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks

    ax1.set_xlabel('')
    ax1.set_ylabel('AUROC f')
    ax1.set_ylim(0.45, 1)  # Set y-axis limits

    # Make the third plot
    plt.sca(ax2)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='accuracy_male', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)
    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks'

    ax2.set_xlabel('Ratios')
    ax2.set_ylabel('ACC m')
    ax2.set_ylim(0.45, 1)  # Set y-axis limits

    # Make the fourth plot
    plt.sca(ax3)
    df_for_plot = data.copy()
    df_for_plot['variation_hue'] = df_for_plot['variation']
    gen_plot_panel(x_var='variation', y_var='ROC_male', data=df_for_plot, min_jitter=-0.05, max_jitter=0.05)
    plt.xticks(ticks=np.linspace(0, data['variation'].max(), num=5))  # Adjust spacing between ticks

    ax3.set_xlabel('Ratios')
    ax3.set_ylabel('AUROC m')
    ax3.set_ylim(0.45, 1)  # Set y-axis limits

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.suptitle('CNN ACC & AUROC for female and male')

    plt.savefig('recreation/figures/cnn_regression_plot.png')

def mean_lr():
    data_lr = pd.read_csv('data/results/lr_results/lr_results_aug.csv')
    acc_female = round(data_lr["accuracy_female"].mean(), 3)
    acc_male = round(data_lr["accuracy_male"].mean(), 3)
    auroc_female = round(data_lr["auroc_female"].mean(), 3)
    auroc_male = round(data_lr["auroc_male"].mean(), 3)
    precision = round(data_lr["Precision"].mean(), 3)
    recall = round(data_lr["Recall"].mean(), 3)
    f1 = round(data_lr["F1-score"].mean(), 3)

    print(f'acc_female: {acc_female}\nacc_male: {acc_male}\nauroc_female: {auroc_female}\nauroc_male: {auroc_male}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}')

    return (f'acc_female: {acc_female}\nacc_male: {acc_male}\nauroc_female: {auroc_female}\nauroc_male: {auroc_male}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}')
 
def mean_cnn():
    data_cnn = pd.read_csv("data/results/cnn_results/runs-2-1.csv")
    data_cnn = data_cnn.drop(columns=["Start Time", "Duration", "Run ID", "Name", "Source Type", "Source Name", "User", "Status"])
    acc_female = data_cnn["Accuracy_female"].mean()
    acc_male = data_cnn["Accuracy_male"].mean()
    auroc_female = data_cnn["ROC_female"].mean()
    auroc_male = data_cnn["ROC_male"].mean()
    precision = data_cnn["Precision"].mean()
    recall = data_cnn["Recall"].mean()
    f1 = data_cnn["F1-score"].mean()

    print(f'acc_female: {acc_female}\nacc_male: {acc_male}\nauroc_female: {auroc_female}\nauroc_male: {auroc_male}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}')
    return (f'acc_female: {acc_female}\nacc_male: {acc_male}\nauroc_female: {auroc_female}\nauroc_male: {auroc_male}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}')

def get_p_value_mannwhitneyu(df, name):
    mwu_result_acc = mannwhitneyu(df["accuracy_female"], df["accuracy_male"], alternative="two-sided")
    mwu_result_auroc = mannwhitneyu(df["accuracy_female"], df["accuracy_male"], alternative="two-sided")
    print(f'{name} ACC mean f&m p-value with mannwhitneyu: {mwu_result_acc}')
    print(f'{name} AUROC mean f&m p-value with mannwhitneyu: {mwu_result_auroc}')

    return (f'{name} ACC mean f&m p-value with mannwhitneyu: {mwu_result_acc}\n{name} AUROC mean f&m p-value with mannwhitneyu: {mwu_result_auroc}')

def get_p_value_ttest_LR(df, name):
    # Define the formula using the column names directly
    mdl_f_ACC = ols(formula='variation ~ accuracy_female', data=df)
    results_f_ACC = mdl_f_ACC.fit()

    mdl_m_ACC = ols(formula='variation ~ accuracy_male', data=df)
    results_m_ACC = mdl_m_ACC.fit()

    mdl_f_AUROC = ols(formula='variation ~ auroc_female', data=df)
    results_f_AUROC = mdl_f_AUROC.fit()

    mdl_m_AUROC = ols(formula='variation ~ auroc_male', data=df)
    results_m_AUROC = mdl_m_AUROC.fit()
    # Accessing parameters, standard errors, and p-values by index can lead to errors if the formula changes
    # So, use the column name to make it more robust
    p_value_female_ACC = results_f_ACC.pvalues['accuracy_female']
    p_value_male_ACC = results_m_ACC.pvalues['accuracy_male']
    p_value_female_AUROC = results_f_AUROC.pvalues['auroc_female']
    p_value_male_AUROC = results_m_AUROC.pvalues['auroc_male']


    print(f'{name} Female ACC p-value with t-test {p_value_female_ACC}')
    print(f'{name} Male ACC p-value with t-test {p_value_male_ACC}')
    print(f'{name} Female AUROC p-value with t-test {p_value_female_AUROC}')
    print(f'{name} Male AUROC p-value with t-test {p_value_male_AUROC}')

    return (f'{name} Female ACC p-value with t-test {p_value_female_ACC}\n{name} Male ACC p-value with t-test {p_value_male_ACC}\n{name} Female AUROC p-value with t-test {p_value_female_AUROC}\n{name} Male AUROC p-value with t-test {p_value_male_AUROC}')

def get_p_value_ttest_CNN(df, name):
    # Define the formula using the column names directly
    mdl_f_ACC = ols(formula='variation ~ accuracy_female', data=df)
    results_f_ACC = mdl_f_ACC.fit()

    mdl_m_ACC = ols(formula='variation ~ accuracy_male', data=df)
    results_m_ACC = mdl_m_ACC.fit()

    mdl_f_AUROC = ols(formula='variation ~ ROC_female', data=df)
    results_f_AUROC = mdl_f_AUROC.fit()

    mdl_m_AUROC = ols(formula='variation ~ ROC_male', data=df)
    results_m_AUROC = mdl_m_AUROC.fit()
    # Accessing parameters, standard errors, and p-values by index can lead to errors if the formula changes
    # So, use the column name to make it more robust
    p_value_female_ACC = results_f_ACC.pvalues['accuracy_female']
    p_value_male_ACC = results_m_ACC.pvalues['accuracy_male']
    p_value_female_AUROC = results_f_AUROC.pvalues['ROC_female']
    p_value_male_AUROC = results_m_AUROC.pvalues['ROC_male']


    print(f'{name} Female ACC p-value with t-test {p_value_female_ACC}')
    print(f'{name} Male ACC p-value with t-test {p_value_male_ACC}')
    print(f'{name} Female AUROC p-value with t-test {p_value_female_AUROC}')
    print(f'{name} Male AUROC p-value with t-test {p_value_male_AUROC}')

    return (f'{name} Female ACC p-value with t-test {p_value_female_ACC}\n{name} Male ACC p-value with t-test {p_value_male_ACC}\n{name} Female AUROC p-value with t-test {p_value_female_AUROC}\n{name} Male AUROC p-value with t-test {p_value_male_AUROC}')

def p_values():
    data_CNN = pd.read_csv('data/results/cnn_results/model_training_results_cnn_5_10may_R.csv')
    data_LR = pd.read_csv('data/results/lr_results/lr_results_aug.csv')
    cnn_mannwhitney_p_value = get_p_value_mannwhitneyu(data_CNN, 'CNN')
    lr_mannwhitney_p_value = get_p_value_mannwhitneyu(data_LR, 'LR')
    cnn_ttest_p_value = get_p_value_ttest_CNN(data_CNN, 'CNN')
    lr_ttest_p_value = get_p_value_ttest_LR(data_LR, 'LR')

    return (f'{cnn_mannwhitney_p_value}\n{lr_mannwhitney_p_value}\n{cnn_ttest_p_value}\n{lr_ttest_p_value}')



if __name__ == "__main__":
    diagnosis_distribution = create_diagnosis_dist()
    lr_plot = regression_plot_lr()
    cnn_plot = regression_plot_cnn()
    mean_lr = mean_lr()
    mean_cnn = mean_cnn()
    p_values = p_values()

    with open("recreation/table_values/table_values.txt", "w") as text_file:
        text_file.write(f"Means for lr:\n{mean_lr}\nMeans for cnn:\n{mean_cnn}\nP-values:\n{p_values}")
