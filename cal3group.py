from os import read
import sys
import argparse
import numpy as np
import solve_common as sc

PI = 3.1415926536
initval = None

# calibration model
def func_model(X, data, r, opto):
    global initval
    if opto == 1:  # optimize scaling factors
        if initval is None:
            initval = np.copy(X)
        else:
            X[:3] = initval[:3]
            X[4:7] = 0
            X[8:11] = 0
    elif opto == 2:  # optimize offsets
        if initval is None:
            initval = np.copy(X)
        else:
            X[3:] = initval[3:]

    Marr = np.append(X[3:].reshape(3, 3), X[:3].reshape(3, 1), axis=1)
    Mat = Marr.tolist()
    cost = 0.0
    inv_r = 1.0 / r

    clist = []
    for reading in data:
        corrected = sc.transform(Mat, reading)
        clist.append(corrected)
        diff = sc.dist3(corrected, [0, 0, 0]) * inv_r - 1.0
        cost += diff * diff

    return cost


def optimize(data, center, target):
    # initial value for correction matrix
    # offset before scaling factors
    X0 = np.append(-np.array(center), [1, 0, 0, 0, 1, 0, 0, 0, 1])
    # optimization
    # optimize offset first
    result = sc.optimize(func_model, X0, thres=0,
                         args=(data, target, 2))
    # optimize scaling factor next
    result = sc.optimize(func_model, result.x, thres=0,
                         args=(data, target, 1))
    # optimize everything together
    result = sc.optimize(func_model, result.x, thres=0,
                         args=(data, target, 0))
    return result.x


def generate_matrix(result, scaling=1):
    return np.append(result[3:].reshape(3, 3), result[0:3].reshape(3, 1), axis=1) * scaling


def transform_data(mat, data, scaling):
    m = mat.tolist()
    p = (-mat[:, 3] / scaling).tolist()
    new_data = []
    nr_list = []
    or_list = []
    for d in data:
        transformed = sc.transform(m, d)
        newr = sc.dist3(transformed, [0, 0, 0])
        oldr = sc.dist3(d, p)
        # print('r = {:.5f}\tor = {:.5f}\tcorrected = {}'.format(newr, oldr, corrected))
        # print('{}, {}, {}'.format(corrected[0], corrected[1], corrected[2]))
        new_data.append(transformed)
        nr_list.append(newr)
        or_list.append(oldr)
    return new_data, nr_list, or_list


def read_data(file, cols, start, end):
    with open(file, 'r') as f:
        lines = f.readlines()

    if start and end:
        lines = lines[start - 1:end]
    elif start is None:
        lines = lines[:end]
    elif end is None:
        lines = lines[start - 1:]

    data = {}
    for i in range(len(cols)):
        data[i] = [[]]

    for line in lines:
        tokens = line.split(',')
        try:
            tmp = list(map(float, tokens))
        except Exception:
            continue

        for i in range(len(cols)):
            c = cols[i]
            data[i][0].append(tmp[c - 1:c + 2] + [1])

    return data


def estimate_center_radius(data):
    for v in data.values():
        data_list = v[0]

        dmax = np.max(data_list, axis=0)
        dmin = np.min(data_list, axis=0)
        center = (dmax + dmin) / 2
        c = center.tolist()[:3]

        r_list = []
        for d in data_list:
            r_list.append(sc.dist3(d, c))

        v.append(np.median(r_list))
        v.append(c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.MetavarTypeHelpFormatter)

    # general arguments
    parser.add_argument('-c', '--columns', type=str,
                        help='start columns for data', required=True)
    parser.add_argument('-s', '--start', type=int,
                        help='start row for data')
    parser.add_argument('-e', '--end', type=int,
                        help='end row for data')
    parser.add_argument('-i', '--input', type=str,
                        help='calibration input data', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='output file for corrected data')
    parser.add_argument('-m', '--matrix-output', type=str,
                        help='output file for correction matrices')
    parser.add_argument('-r', '--reference', type=float,
                        help='reference earth magnetic field strength')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='verbosity level')

    args = parser.parse_args()

    columns = list(map(int, args.columns.split(',')))
    print('Loading data ...')
    data = read_data(args.input, columns, args.start, args.end)
    print('Preprocessing data ...')
    estimate_center_radius(data)

    ref = args.reference
    print('\nCalibration phase 1')
    print('==========')
    for i in range(len(data)):
        data_list, r, c = data[i]
        print('Calibrate data list {}'.format(i + 1))
        result = optimize(data_list, c, r)
        data[i].append(result)
        mat = generate_matrix(result)

        transformed, nr_list, _ = transform_data(mat, data_list, 1)
        newr = np.median(nr_list)
        if not ref:
            ref = newr
        scaling = ref / newr
        mat = generate_matrix(result, scaling)
        transformed, nr_list, or_list = transform_data(mat, data_list, scaling)
        print('Mat =', mat)
        print('new center = {}, old center = {}'.format(
            [0, 0, 0], (-mat[:, 3] / scaling).tolist()))
        print('corrected r = {:.5f}, std = {:.5f}, old r = {:.5f}, std = {:.5f}'.format(
            np.median(nr_list), np.std(nr_list), np.median(or_list), np.std(or_list)))
