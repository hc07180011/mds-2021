import copy
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class Jobs:

    def __init__(self) -> None:
        self.idxs = list()
        self.processing_times = list()
        self.due_dates = list()
        self.weights = list()
    
    def load_data(
        self,
        idxs: list[int],
        processing_times: list[int],
        due_dates: list[int],
        weights: list[int]
    ):
        self.idxs = idxs
        self.processing_times = processing_times
        self.due_dates = due_dates
        self.weights = weights

    def calculate_obj(self, order: list[int]) -> int:
        ret = 0
        for i in range(len(order)):
            p = 0
            for j in range(i + 1):
                p += self.processing_times[order[j]]
            ret += self.weights[order[i]] * \
                    max(p - self.due_dates[order[i]], 0)
        return ret

    def order_crossover(self, parent1, parent2):
        idx = np.random.randint(len(parent1) - 3)
        buf = parent1[idx: (idx + 4)]
        ret = list()
        c = 0
        for _ in range(len(parent1)):
            if parent2[c] not in buf:
                ret.append(parent2[c])
            c += 1
        for i in range(idx, idx + 4):
            ret.insert(i, parent1[i])
        return ret

    def reversion_mutation(self, parent):
        parent = parent.tolist()
        index = np.random.randint(len(parent) - 3)
        buf = parent[index: (index + 4)][::-1]
        ret = list()
        for i in range(0, index):
            ret.append(parent[i])
        c = 0
        for _ in range(index, index + 4):
            ret.append(buf[c])
            c += 1
        for i in range(index + 4, len(parent)):
            ret.append(parent[i])
        return ret


class Ans:

    def __init__(self) -> None:
        self.tabus = list()
        self.tartiness_records = list()
        self.obj = np.inf
        self.best_order = list()

    def reset_local(self) -> None:
        class _Local:
            def __init__(self) -> None:
                self.obj = np.inf
                self.left = -1
                self.right = -1
                self.idx = -1
        self.local = _Local()


def genetic_algorithm(jobs: Jobs, epochs: int) -> Ans:
    ans = Ans()

    job_order = np.arange(len(jobs.idxs))
    job_order_iters = list()
    for _ in range(10):
        np.random.shuffle(job_order)
        job_order_iters.append(copy.deepcopy(job_order))

    for _ in tqdm(range(epochs)):

        try_order = list()

        for __ in range(8):
            choices = np.random.choice(
                np.arange(10), 2, replace=False
            )
            try_order.append(
                jobs.order_crossover(
                    job_order_iters[choices[0]],
                    job_order_iters[choices[1]]
                )
            )

        for __ in range(2):
            try_order.append(
                jobs.reversion_mutation(
                    job_order_iters[np.random.randint(10)]
                )
            )

        fitness_list = list()
        for j in range(10):
            fitness_list.append(
                1 / jobs.calculate_obj(try_order[j])
            )
        fitness_list = np.cumsum(fitness_list) / np.sum(fitness_list)

        real_list = list()
        for ___ in range(10):
            val = np.random.uniform(0, 1)
            for k in range(10):
                if val <= fitness_list[k]:
                    real_list.append(
                        try_order[k]
                    )
                    break

        job_order_iters = copy.deepcopy(np.array(real_list))

        ans.reset_local()
        for j in range(len(job_order_iters)):
            current_obj = jobs.calculate_obj(
                job_order_iters[j]
            )
            if current_obj < ans.local.obj:
                ans.local.obj = current_obj
                ans.local.idx = j

        ans.tartiness_records.append(ans.local.obj)
        if ans.local.obj < ans.obj:
            ans.obj = ans.local.obj
            ans.best_order = job_order_iters[ans.local.idx]

    return ans


def _main() -> None:
    jobs = Jobs()
    jobs.load_data(
        np.arange(1, 21, 1).tolist(),
        list((
            10, 10, 13, 4, 9, 4, 8, 15, 7, 1,
            9, 3, 15, 9, 11, 6, 5, 14, 18, 3
        )),
        list((
            50, 38, 49, 12, 20, 105, 73, 45, 6, 64,
            15, 6, 92, 43, 78, 21, 15, 50, 150, 99
        )),
        list((
            10, 5, 1, 5, 10, 1, 5, 10, 5, 1,
            5, 10, 10, 5, 1, 10, 5, 5, 1, 5
        ))
    )

    ans = genetic_algorithm(jobs, 30000)

    print("Weighted tardiness: {}".format(ans.obj))
    print(", ".join([str(x) for x in ans.best_order]))

    plt.figure(figsize=(16, 8))
    plt.plot(ans.tartiness_records)
    plt.xlabel("#Epoch")
    plt.ylabel("Tardiness")
    plt.savefig("3-genetic.png")


if __name__ == "__main__":
    _main()
