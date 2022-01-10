import numpy as np
import matplotlib.pyplot as plt


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


def tabu_search(jobs: Jobs, window_size: int, epochs: int) -> Ans:

    def _calculate_obj(order: list[int]) -> int:
        ret = 0
        for i in range(len(order)):
            p = 0
            for j in range(i + 1):
                p += jobs.processing_times[order[j]]
            ret += jobs.weights[order[i]] * \
                    max(p - jobs.due_dates[order[i]], 0)
        return ret

    job_order = np.arange(len(jobs.idxs))
    np.random.shuffle(job_order)

    ans = Ans()

    for _ in range(epochs):

        ans.reset_local()

        for j in range(len(job_order) - 1):
            
            current_left = job_order[j]
            current_right = job_order[j + 1]

            consecutive = False
            for k in range(len(ans.tabus)):
                if (current_left == ans.tabus[k][0] and current_right == ans.tabus[k][1]) or \
                    (current_right == ans.tabus[k][0] and current_left == ans.tabus[k][1]):
                    consecutive = True

            if not consecutive:
                try_order = job_order.copy()
                try_order[j+1] = current_left
                try_order[j] = current_right
                current_obj = _calculate_obj(try_order)

                if current_obj < ans.local.obj:
                    ans.local.obj = current_obj
                    ans.local.left = current_left
                    ans.local.right = current_right
                    ans.local.idx = j

        ans.tabus.insert(0, [ans.local.left, ans.local.right])

        if len(ans.tabus) > window_size:
            ans.tabus.pop()

        temp = job_order[ans.local.idx]
        job_order[ans.local.idx] = job_order[ans.local.idx+1]
        job_order[ans.local.idx+1] = temp
        ans.tartiness_records.append(ans.local.obj)

        if ans.local.obj < ans.obj:
            ans.obj = ans.local.obj
            ans.best_order = list([
                x + 1 for x in job_order
            ])

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

    ans = tabu_search(jobs, 15, 1500)

    print("Weighted tardiness: {}".format(ans.obj))
    print(", ".join([str(x) for x in ans.best_order]))

    plt.plot(ans.tartiness_records)
    plt.xlabel("#Epoch")
    plt.ylabel("Tardiness")
    plt.savefig("3-tabu.png")


if __name__ == "__main__":
    _main()
