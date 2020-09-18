import numpy as np


def hamming_distance(g_hat, g_true, n_diag=False):
    if n_diag:
        non_diag = np.where(~np.eye(g_hat.shape[0], dtype=bool))
        res = (g_hat[non_diag] == g_true[non_diag]) \
            .sum() / len(non_diag[0])
    else:
        res = (g_hat == g_true).sum()
        # circles = ((g_hat == 3) & (g_true==2)).sum()/2
        # res = res + circles
        res = res / (g_hat.shape[0]**2)
    return res


def topology(g_hat, g_true):
    # non_diag = np.where(~np.eye(g_hat.shape[0], dtype=bool))
    if np.sum(g_hat) == np.sum(g_true) == 0:
        return 1
    else:
        d = g_hat.shape[0]
        non_diag = np.triu_indices(d, 0)
        g_hat = g_hat[non_diag]
        g_true = g_true[non_diag]
        g_hat[g_hat > 0] = 1
        g_true[g_true > 0] = 1
        sub = g_true - g_hat
        missing = (sub > 0).sum()
        added = (sub < 0).sum()
        # correct = (sub == 0).sum()
        correct = ((g_true == 1) == (g_true == g_hat)).sum()
        return correct/(correct + missing + added)


def precision_no(g_hat, g_true):
    true_pos = 0
    true_false_pos = 0
    for i in range(g_true.shape[1]):
        gt = np.where(g_true[i, i:] > 0)[0].tolist()
        gh = np.where(g_hat[i, i:] > 0)[0].tolist()
        true_false_pos = true_false_pos + len(gh)
        true_pos = true_pos + len(set(gh).intersection(gt))
    # for i in range(g_hat.shape[1]):
    #     gh = np.where(g_hat[i, i:] > 0)[0].tolist()
    #     true_false_pos = true_false_pos + len(gh)
    if true_false_pos == 0:
        if (g_true == 0).all():
            return 1
        else:
            return 0
    else:
        return true_pos / true_false_pos


def recall_no(g_hat, g_true):
    true_pos = 0
    true_pos_false_neg = 0
    # true_pos_circle = 0
    for i in range(g_true.shape[1]):
        gt = np.where(g_true[i, i:] > 0)[0].tolist()
        gh = np.where(g_hat[i, i:] > 0)[0].tolist()
        # gh_circle = np.where(g_hat[i, :] == 3)[0].tolist()
        true_pos = true_pos + len(set(gh).intersection(gt))
        # true_pos_circle = true_pos_circle + len(set(gh_circle).intersection(gt))/2
        true_pos_false_neg = true_pos_false_neg + len(gt)
    if true_pos_false_neg == 0:
        if (g_hat == 0).all():
            return 1
        else:
            return 0
    else:
        return true_pos/true_pos_false_neg


def f_score_no(g_hat, g_true):
    p = precision_no(g_hat, g_true)
    r = recall_no(g_hat, g_true)
    if (p == 0) and (r == 0):
        return 0
    else:
        return 2 * p * r / (p + r)


def precision(g_hat, g_true):
    true_pos = 0
    true_false_pos = 0
    # true_pos_circle = 0
    for i in range(g_true.shape[1]):
        gt = np.where(g_true[i, :] == 2)[0].tolist()
        gh = np.where(g_hat[i, :] == 2)[0].tolist()
        true_false_pos = true_false_pos + len(gh)
        # gh_circle = np.where(g_hat[i, :] == 3)[0].tolist()
        true_pos = true_pos + len(set(gh).intersection(gt))
        # true_pos_circle = true_pos_circle + len(set(gh_circle).intersection(gt))/2
        if g_hat[i, i] == 1:
            true_false_pos = true_false_pos + 1
            if g_true[i, i] == 1:
                true_pos = true_pos + 1
    # true_false_pos_circle = 0
    # for i in range(g_hat.shape[1]):
    #     gh = np.where(g_hat[i, :] == 2)[0].tolist()
    #     # gh_circle = np.where(g_hat[i, :] == 3)
    #     true_false_pos = true_false_pos + len(gh)
    #     # true_false_pos_circle = true_false_pos_circle + len(gh_circle)/2
    #     if g_hat[i, i] == 1:
    #         true_false_pos = true_false_pos + 1

    if true_false_pos == 0:
        if (g_true == 0).all():
            return 1
        else:
            return 0
    else:
        return true_pos / true_false_pos


def recall(g_hat, g_true):
    true_pos = 0
    true_pos_false_neg = 0
    for i in range(g_true.shape[1]):
        gt = np.where(g_true[i, :] == 2)[0].tolist()
        gh = np.where(g_hat[i, :] == 2)[0].tolist()
        true_pos = true_pos + len(set(gh).intersection(gt))
        true_pos_false_neg = true_pos_false_neg + len(gt)
        if g_true[i, i] == 1:
            if g_hat[i, i] == 1:
                true_pos = true_pos + 1
            true_pos_false_neg = true_pos_false_neg + 1

    if true_pos_false_neg == 0:
        if (g_hat == 0).all():
            return 1
        else:
            return 0
    else:
        return true_pos / true_pos_false_neg


def f_score(g_hat, g_true):
    p = precision(g_hat, g_true)
    r = recall(g_hat, g_true)
    if (p == 0) and (r == 0):
        return 0
    else:
        return 2 * p * r / (p + r)


if __name__ == "__main__":
    unit = np.diag(np.diag(np.ones([3, 3])))
    unit[0, 1] = 2
    unit[1, 0] = 1
    unit[0, 2] = 2
    unit[2, 0] = 1
    print(unit)
    infer = np.diag(np.diag(np.ones([3, 3])))
    infer[1, 1] = 0
    infer[2, 2] = 0
    print(infer)
    print(topology(infer, unit))
    print(precision_no(infer, unit))
    print(recall_no(infer, unit))
    print(f_score_no(infer, unit))
    print(hamming_distance(infer, unit))
    print(precision(infer, unit))
    print(recall(infer, unit))
    print(f_score(infer, unit))
