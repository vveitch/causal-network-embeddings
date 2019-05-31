import os
import seaborn as sns
import matplotlib.pyplot as plt
import relational_ERM.effect_estimates.compute_ate as ate
import pandas as pd


def make_reddit_prop_plt():
    sns.set()
    prop_expt = pd.DataFrame(ate.process_exogeneity_experiment())
    # prop_expt.to_pickle('prop_expt_tmp.pkl')
    # prop_expt = pd.read_pickle('prop_expt_tmp.pkl')

    # prop_expt = prop_expt[['exog', 'q_only', 'aiptw', 'tmle', 'very_naive']]
    # prop_expt = prop_expt.rename(index=str, columns={'exog': 'Exogeneity',
    #                                      'very_naive': 'Unadjusted',
    #                                                  'aiptw': 'AIPTW',
    #                                                  'tmle': 'TMLE'})
    prop_expt = prop_expt[['exog', 'q_only', 'tmle', 'very_naive']]
    prop_expt = prop_expt.rename(index=str, columns={'exog': 'Exogeneity',
                                                     'q_only': 'Simple',
                                                     'very_naive': 'Unadjusted',
                                                     'tmle': 'TMLE'})

    prop_expt = prop_expt.set_index('Exogeneity')

    plt.figure(figsize=(4.75, 3.5))
    # plt.figure(figsize=(2.37, 1.5))
    sns.scatterplot(data=prop_expt, legend='brief', s=75)
    plt.xlabel("Exogeneity", fontfamily='monospace')
    plt.ylabel("ATE Estimate", fontfamily='monospace')
    plt.tight_layout()

    fig_dir = '../output/figures'
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir,'region_exogeneity.pdf'))


def main():
    make_reddit_prop_plt()


if __name__ == '__main__':
    main()