# number of products n = 3, 4, ..., 10
# one job per number of product
# for each job, 2000 instances
# for each instance, run three, four, or five algorithms
# record revenue, assortment, opaque price by each of the algorithm
# dataframe with columns
#    algorithm, instance_id, revenue, assortment, opaque price

import sys
import numpy as np
import pandas as pd
from opaque_products import MNL


def get_instances(size, n_inst):
    vs = np.random.lognormal(0.5, 1, size=(n_inst, size))
    rs = np.random.lognormal(0.0, 1, size=(n_inst, size))
    return [(vs[k, :], rs[k, :]) for k in range(n_inst)]


def get_results(model:MNL, algorithm):
    if algorithm == "brute force":
        return model.brute_force()
    if algorithm == "nested by rev n val":
        return model.nested_by_rev_n_val()
    if algorithm == "singleton":
        return model.best_singleton()
    if algorithm == "optimal traditional":
        return model.nested_by_rev_trad()


if __name__ == "__main__":
    alg_names = ["brute force", "nested by rev n val", "singleton", "optimal traditional"]
    n = int(sys.argv[1])
    num_instances = int(sys.argv[2])
    name = sys.argv[3]
    np.random.seed(1405 + n * 10)
    instances = get_instances(n, num_instances)
    instance_df = pd.DataFrame(columns=["instance", "n", "v", "r"])
    result_df = pd.DataFrame(
        columns=["instance", "n", "v", "r", "alg", "rev", "asst", "size", "r_min", "opq_price"])
    for i, (v, r) in enumerate(instances):
        inst_num = n * 1e6 + i
        mnl_model = MNL(v, r)
        instance_df.loc[i] = [inst_num, n, v, r]
        for alg in alg_names:
            res = get_results(mnl_model, alg)
            row_df = pd.DataFrame({
                "instance": inst_num, "n": n, "v": str(v), "r": str(r), "alg": alg,
                "rev": res["rev"], "asst": str(res["asst"]), "opq_price": res["rho"],
                "size": np.sum(res["asst"]), "r_min": np.min(r[np.array(res["asst"], dtype=bool)])
            }, index=[0])
            result_df = pd.concat([result_df, row_df], ignore_index=True)
    result_df.to_csv("output/{}/{}_{}.csv".format(name, name, n), index=False)