import numpy as np
import matplotlib.pyplot as plt


def _data_loader() -> np.array:
    return list((
        68, 71, 67, 69, 71,
        70, 69, 67, 70, 70,
        79, 79, 78, 78, 78,
        79, 79, 82, 82, 81
    ))


def _get_ucl(N: int, mu: float, sigma: float, lambda_: float, L: float) -> np.array:
    return list([
        mu + L * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * (i + 1))))
        for i in range(N)
    ])


def _get_lcl(N: int, mu: float, sigma: float, lambda_: float, L: float) -> np.array:
    return list([
        mu - L * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * (i + 1))))
        for i in range(N)
    ])


def _get_ewma(data: np.array, lambda_) -> np.array:
    z = list((0,))
    for i in range(len(data)):
        z.append(lambda_ * np.mean(data[:(i+1)]) + (1 - lambda_) * z[-1])
    return np.array(z[1:])


def _simulate(data, mu, sigma, lambda_, L) -> list[np.array]:

    UCL = _get_ucl(
        len(data),
        mu,
        sigma,
        lambda_,
        L
    )
    LCL = _get_lcl(
        len(data),
        mu,
        sigma,
        lambda_,
        L
    )
    EWMA = _get_ewma(
        data,
        lambda_
    )
    return UCL, EWMA, LCL


def _main() -> None:
    heart_rate_data = _data_loader()

    print(np.mean(heart_rate_data))

    UCL, EWMA, LCL = _simulate(
        heart_rate_data,
        mu=70.0,
        sigma=3.0,
        lambda_=0.1,
        L=2.81
    )
    plt.figure(figsize=(16, 4))
    plt.plot(UCL)
    plt.plot(EWMA)
    plt.plot(LCL)
    plt.legend(["UCL", "EWMA", "LCL"])
    plt.savefig("1-a.png")

    UCL, EWMA, LCL = _simulate(
        heart_rate_data,
        mu=70.0,
        sigma=3.0,
        lambda_=0.5,
        L=3.07
    )
    plt.figure(figsize=(16, 4))
    plt.plot(UCL)
    plt.plot(EWMA)
    plt.plot(LCL)
    plt.legend(["UCL", "EWMA", "LCL"])
    plt.savefig("1-b.png")

    UCL, EWMA, LCL = _simulate(
        heart_rate_data,
        mu=76.0,
        sigma=3.0,
        lambda_=0.1,
        L=2.81
    )
    plt.figure(figsize=(16, 4))
    plt.plot(UCL)
    plt.plot(EWMA)
    plt.plot(LCL)
    plt.legend(["UCL", "EWMA", "LCL"])
    plt.savefig("1-c.png")

    ARL = 1 / np.sum([
        EWMA[i] <= UCL[i] and EWMA[i] >= LCL[i]
        for i in range(len(EWMA))
    ])
    print(ARL, 1 / ARL)


if __name__ == "__main__":
    _main()