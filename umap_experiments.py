#!/usr/bin/python3

from simulated_data import *
from mastery_methods import *
import argparse
from scipy.stats import spearmanr
import seaborn as sns
import pylab as plt

# values used in grid search for the best value of threshold for EMA method
THRESHOLDS = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99]
# valuse used in grid search for the best value of N for NCC method
N_THRESHOLDS = list(range(1, 25))
# separator for output
SEP = "\t"   
# 6 fixed colors from the viridis colormap 
viridis6 = [[0.565498, 0.84243, 0.262877], [0.20803, 0.718701, 0.472873], [0.127568, 0.566949, 0.550556], [0.190631, 0.407061, 0.556089], [0.267968, 0.223549, 0.512008], [0.267004, 0.004874, 0.329415]]


def wMAD(true_mastery, detected_mastery, w=5):
    return np.mean(list(map(lambda x: w*x if x > 0 else -x, true_mastery - detected_mastery)))


def find_best_params(method, data, candidates, metric=wMAD):
    results = []
    for p in candidates:
        if isinstance(p, tuple):
            md = method(*p)
        else:
            md = method(p)
        results.append((metric(data.true_mastery, md.detected_mastery), p))
    return sorted(results)[0][1]  


def rand_jitter(arr):
    return np.array(arr) + np.random.randn(len(arr)) * 0.11


def jitter_scatter(x, y):
    plt.scatter(rand_jitter(x), rand_jitter(y))
    plt.plot((min(x), max(x)), (min(x), max(x)))


def print_bkt_ncc_comparison(scenario_name, students):
    bkt_params = scenarios[scenario_name]["params"]
    train = SimulatedDataBKT(bkt_params, students)
    bkt_threshold = find_best_params(lambda p: BKTMastery(train, bkt_params, p), train,
                                     THRESHOLDS)
    ncc_threshold = find_best_params(lambda n: NCCMastery(train, n), train,
                                     N_THRESHOLDS)
    test = SimulatedDataBKT(bkt_params, students)
    bkt_test = BKTMastery(test, bkt_params, bkt_threshold)
    ncc_test = NCCMastery(test, ncc_threshold)
    print(scenario_name,
          ncc_threshold,
          bkt_threshold,
          round(wMAD(test.true_mastery, ncc_test.detected_mastery), 2),
          round(wMAD(test.true_mastery, bkt_test.detected_mastery), 2),
          round(spearmanr(bkt_test.detected_mastery, ncc_test.detected_mastery)[0], 2),
          sep=SEP)
    
    
def bkt_ncc_comparison_scenarios(students):
    print("sc", "NCC-n", "BKT-t", "NCC m", "BKT m", "Corr", sep=SEP)
    for scenario_name in sorted(scenarios):
        if scenario_name[0] == "B":  # only BKT scenarios
            print_bkt_ncc_comparison(scenario_name, students)


def print_ema_ncc_comparison(scenario_name, students):
    train = data_for_scenario(scenario_name, students)
    ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ema_params = find_best_params(lambda a, t: EMAMastery(train, a, t), train,
                                  [(a, t) for a in ALPHAS for t in THRESHOLDS])
    ema95_alpha = find_best_params(lambda a: EMAMastery(train, a, 0.95), train,
                                   ALPHAS)
    ncc_threshold = find_best_params(lambda n: NCCMastery(train, n), train,
                                     N_THRESHOLDS)
    test = data_for_scenario(scenario_name, students)
    ncc_test = NCCMastery(test, ncc_threshold)
    ema_test = EMAMastery(test, *ema_params)
    ema95_test = EMAMastery(test, ema95_alpha, 0.95)
    print(scenario_name,
          ncc_threshold,
          ema95_alpha,
          ema_params[0], ema_params[1],
          round(wMAD(test.true_mastery, ncc_test.detected_mastery), 2),
          round(wMAD(test.true_mastery, ema95_test.detected_mastery), 2),
          round(wMAD(test.true_mastery, ema_test.detected_mastery), 2),
          sep=SEP)


def ema_ncc_comparison_scenarios(students):
    print("sc", "NCC-n", "EMA-a",  "EMA-t", "NCC m", "EMA m", sep=SEP)
    for scenario_name in sorted(scenarios):
        print_ema_ncc_comparison(scenario_name, students)


def effort(mastery):
    return np.mean(mastery.detected_mastery)


def score(data, mastery, how_many=5):
    post_mastery_answers = []
    for s in range(data.students):
        for i in range(mastery.detected_mastery[s]+1, min(mastery.detected_mastery[s]+how_many+1, data.length)):
            post_mastery_answers.append(data.answer[s, i])
    return np.mean(post_mastery_answers)


def get_linestyle(scenario_name):
    if scenario_name[0] == "B":
        return "--"
    return "-"


def effort_score_graph_scenario(scenario_name, students):
    data = data_for_scenario(scenario_name, students)
    effort_score_results = []
    thresholds = [1, 3, 5, 8, 12, 18]
    for n in thresholds:
        ncc_mastery = NCCMastery(data, n)
        effort_score_results.append((effort(ncc_mastery), score(data, ncc_mastery)))
    efforts, scores = zip(*effort_score_results)
    colors = sns.color_palette()
    plt.plot(efforts, scores, linestyle=get_linestyle(scenario_name),
             color=colors[int(scenario_name[1]) - 1], label=scenario_name, alpha=0.7)  
    for i in range(len(thresholds)):
        plt.scatter(efforts[i], scores[i], s=60, marker=(i+3, 1), color=viridis6[i])  


def effort_score_graphs(students):
    for scenario_name in sorted(scenarios):
        if scenario_name[0] == 'L':
            effort_score_graph_scenario(scenario_name, students)
    plt.xlabel("effort")
    plt.ylabel("score")
    plt.xlim((0, 40))
    plt.ylim((0.3, 1.05))
    plt.legend(loc="lower right")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("-s", "--students", type=int)
    args = parser.parse_args()
    if args.students:
        students = args.students
    else:
        students = 10000

    if args.experiment == "bkt_ncc":
        bkt_ncc_comparison_scenarios(students)
    elif args.experiment == "ema_ncc":
        ema_ncc_comparison_scenarios(students)
    elif args.experiment == "esgraph":
        effort_score_graphs(students)
    else:
        print("Unknown experiment", args.experiment)

if __name__ == "__main__":
    main()
