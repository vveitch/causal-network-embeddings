from .compute_ate import ate_from_rerm_tsv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_tsv', type=str,
                        default='../output/demo_output/test_results.tsv')
    args = parser.parse_args()

    all_ate_estimates = ate_from_rerm_tsv(args.predictions_tsv)
    print(all_ate_estimates)
