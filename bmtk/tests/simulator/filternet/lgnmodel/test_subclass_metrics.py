import pytest
import numpy as np

from bmtk.simulator.filternet.lgnmodel.cellmetrics import get_data_metrics_for_each_subclass


# Helper functions for comparing returned dictionaries
def cmp_vals(v1, v2):
    if isinstance(v1, (list, np.ndarray)):
        return len(v1) == len(v2) and np.allclose(v1, v2, equal_nan=True)
    else:
        return v1 == v2


def cmp_dicts(d1, d2):
    if len(d1) != len(d2):
        return False
    elif d1.keys() != d2.keys():
        return False
    else:
        for k in d1.keys():
            if isinstance(d1[k], dict):
                if not cmp_dicts(d1[k], d2[k]):
                    return False

            elif not cmp_vals(d1[k], d2[k]):
                return False

    return True


tOFF_expected = {
    'TF1': {'f0_exp': np.array([9.83333333, 4.0, 4.58333333, 5.20833333, 1.45833333]), 'f1_exp': np.array([4.49336233, 1.84107264, 1.91663364, 3.67203572, 1.52442197]), 'spont_exp': np.array([5.5]), 'si_exp': np.array([0.23827179]), 'si_inf_exp': np.array([0.06378631]), 'ttp_exp': np.array([69.5]), 'f0_std': np.array([[5.15432278, 5.15432278, 5.15432278, 5.15432278, 5.15432278]]), 'f1_std': np.array([[7.72476395, 7.72476395, 7.72476395, 7.72476395, 7.72476395]]), 'spont_std': np.array([[3.8404462, 3.8404462, 3.8404462, 3.8404462, 3.8404462]]), 'si_std': np.array([[0.02667022, 0.02667022, 0.02667022, 0.02667022, 0.02667022]]), 'nsub': 1, 'N_class': 17},
    'TF15': {'f0_exp': np.array([1.93055556, 1.70277778, 2.29166667, 3.225, 4.06666667]), 'f1_exp': np.array([1.32139779, 1.3522019 , 2.06846356, 3.2638154, 3.49902869]), 'spont_exp': np.array([1.05066667]), 'si_exp': np.array([0.24185306]), 'si_inf_exp': np.array([0.0697551]), 'ttp_exp': np.array([115.3]), 'f0_std': np.array([1.26499494, 1.49260652, 1.63086548, 1.69650376, 2.34439343]), 'f1_std': np.array([0.44423679, 0.90874906, 1.68247077, 3.21111534, 3.4937977 ]), 'spont_std': np.array([1.28108114]), 'si_std': np.array([0.03535043]), 'nsub': 5, 'N_class': 17},
    'TF2': {'f0_exp': np.array([5.33333333, 7.76190476, 6.4047619 , 6.28571429, 4.14285714]), 'f1_exp': np.array([3.94179055, 7.194137  , 8.20707042, 8.0345728 , 3.92143505]), 'spont_exp': np.array([3.40952381]), 'si_exp': np.array([0.27601501]), 'si_inf_exp': np.array([0.12669168]), 'ttp_exp': np.array([73.]), 'f0_std': np.array([0.0673435 , 3.09780114, 1.38054181, 4.91607572, 3.50186215]), 'f1_std': np.array([0.96222787, 3.50112099, 3.52145779, 6.60968552, 2.74017632]), 'spont_std': np.array([0.8727718]), 'si_std': np.array([0.01009505]), 'nsub': 2, 'N_class': 17},
    'TF4': {'f0_exp': np.array([6.32638889, 6.22255291, 9.67030423, 7.06117725, 5.95833333]), 'f1_exp': np.array([6.26269159, 6.97923619, 9.93226237, 8.05904947, 7.42318534]), 'spont_exp': np.array([3.94455026]), 'si_exp': np.array([0.26112935]), 'si_inf_exp': np.array([0.10188226]), 'ttp_exp': np.array([69.]), 'f0_std': np.array([4.8530305 , 4.05122902, 6.41271074, 5.24535188, 5.20929175]), 'f1_std': np.array([6.07695274, 6.06135475, 9.47140591, 8.07199734, 8.94210903]), 'spont_std': np.array([3.8404462]), 'si_std': np.array([0.02667022]), 'nsub': 6, 'N_class': 17},
    'TF8': {'f0_exp': np.array([5.57936508,  7.7281746,  6.57539683, 10.90079365,  4.32936508]), 'f1_exp': np.array([3.43806478, 4.7698664 , 6.30857421, 7.34213308, 3.99608696]), 'spont_exp': np.array([2.43777778]), 'si_exp': np.array([0.2506875]), 'si_inf_exp': np.array([0.08447917]), 'ttp_exp': np.array([83.16666667]), 'f0_std': np.array([4.84020936, 6.84848353, 4.00513565, 9.70563839, 1.94729605]), 'f1_std': np.array([0.73760654, 1.00074965, 3.8886161 , 3.00087377, 1.09563963]), 'spont_std': np.array([1.9388007]), 'si_std': np.array([0.02675742]), 'nsub': 3, 'N_class': 17}
}

tON_expected = {
    'TF8': {'f0_exp': np.array([ 9.25,  8.83333333, 10.70833333, 10.91666667,  5.45833333]), 'f1_exp': np.array([3.86117629, 4.976896  , 7.40574622, 4.20642425, 2.24653624]), 'spont_exp': np.array([2.6]), 'si_exp': np.array([0.29895665]), 'si_inf_exp': np.array([0.16492775]), 'ttp_exp': np.array([143.5]), 'f0_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'f1_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'spont_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 1, 'N_class': 1}
}

sOFF_expected = {
    'TF1': {'f0_exp': np.array([13.49166667, 11.13333333, 8.73333333, 6.88333333, 5.9]), 'f1_exp': np.array([9.94607624, 8.07727173, 6.98483996, 4.12928985, 3.62209597]), 'spont_exp': np.array([3.95]), 'si_exp': np.array([0.38762728]), 'si_inf_exp': np.array([0.17525456]), 'ttp_exp': np.array([100.9]), 'f0_std': np.array([7.30162415, 7.25110145, 6.45608832, 6.39317942, 6.65458541]), 'f1_std': np.array([6.03768837, 3.29528107, 4.34504274, 2.35583253, 2.07992171]), 'spont_std': np.array([3.7050641]), 'si_std': np.array([0.04421361]), 'nsub': 5, 'N_class': 28},
    'TF15': {'f0_exp': np.array([ 6.0625, 7.20833333, 9.94791667, 11.07291667, 14.77083333]), 'f1_exp': np.array([ 7.36103667,  9.70341838, 15.14123097, 17.36591505, 24.02652807]), 'spont_exp': np.array([5.075]), 'si_exp': np.array([0.45255799]), 'si_inf_exp': np.array([0.30511599]), 'ttp_exp': np.array([83.75]), 'f0_std': np.array([2.21879075, 3.33975077, 5.97974736, 8.13011769, 9.03699316]), 'f1_std': np.array([ 4.24074104,  6.62633957, 11.0345594 , 15.11216051, 17.62187523]), 'spont_std': np.array([2.49849955]), 'si_std': np.array([0.1037873]), 'nsub': 4, 'N_class': 28},
    'TF2': {'f0_exp': np.array([ 7.56666667, 11.33611111,  9.59444444, 7.08055556, 6.75]), 'f1_exp': np.array([ 6.73778133,  9.0378659 , 10.00650897,  7.58057442,  7.32088546]), 'spont_exp': np.array([4.53177778]), 'si_exp': np.array([0.41665936]), 'si_inf_exp': np.array([0.23331871]), 'ttp_exp': np.array([95.1]), 'f0_std': np.array([5.76180853, 7.47925629, 6.46050889, 5.60828621, 4.94895822]), 'f1_std': np.array([5.2963084 , 6.09000615, 6.66430582, 5.97057327, 5.64399237]), 'spont_std': np.array([5.3549573]), 'si_std': np.array([0.10763176]), 'nsub': 5, 'N_class': 28},
    'TF4': {'f0_exp': np.array([ 8.1712963, 9.23611111, 11.12037037, 7.87037037, 7.15740741]), 'f1_exp': np.array([5.43577483, 6.35315115, 8.73767611, 6.56488388, 6.1562911 ]), 'spont_exp': np.array([3.67907407]), 'si_exp': np.array([0.41317098]), 'si_inf_exp': np.array([0.22634197]), 'ttp_exp': np.array([103.61111111]), 'f0_std': np.array([5.17092585, 6.45332467, 6.91831194, 5.7824959 , 5.65108834]), 'f1_std': np.array([4.02343937, 5.68951366, 8.06784057, 6.13605835, 7.12420376]), 'spont_std': np.array([3.1256705]), 'si_std': np.array([0.06482528]), 'nsub': 9, 'N_class': 28},
    'TF8': {'f0_exp': np.array([ 6.50833333, 6.98333333,  8.76666667, 10.76666667, 5.36666667]), 'f1_exp': np.array([ 4.0920903 ,  6.29310598,  7.92442566, 10.39249638,  5.66199926]), 'spont_exp': np.array([3.62]), 'si_exp': np.array([0.3731543]), 'si_inf_exp': np.array([0.1463086]), 'ttp_exp': np.array([106.3]), 'f0_std': np.array([4.67608796, 4.08010076, 8.45389867, 8.37368356, 4.77291621]), 'f1_std': np.array([ 3.35000983,  5.66291795, 10.35818363, 11.77442353,  6.21616106]), 'spont_std': np.array([3.96320577]), 'si_std': np.array([0.05368065]), 'nsub': 5, 'N_class': 28}
}


sON_expected = {
    'TF1': {'f0_exp': np.array([16.25, 10.63541667, 10.16666667, 12., 11.23958333]), 'f1_exp': np.array([8.87029478, 6.38245283, 3.79573778, 7.45355501, 8.40652821]), 'spont_exp': np.array([5.2]), 'si_exp': np.array([0.50268476]), 'si_inf_exp': np.array([0.40536953]), 'ttp_exp': np.array([116.]), 'f0_std': np.array([8.16808378, 5.71055056, 7.99833605, 7.0745043 , 6.40379074]), 'f1_std': np.array([ 8.81001934,  6.34297228,  2.06424173,  9.40336167, 11.56841067]), 'spont_std': np.array([2.89251909]), 'si_std': np.array([0.08435855]), 'nsub': 4, 'N_class': 22},
    'TF15': {'f0_exp': np.array([11.81944444, 11.84722222, 15.23611111, 17.5625, 23.01388889]), 'f1_exp': np.array([13.29027739, 13.91257785, 19.68065105, 27.56254402, 33.61725534]), 'spont_exp': np.array([10.88]), 'si_exp': np.array([0.44384144]), 'si_inf_exp': np.array([0.28768289]), 'ttp_exp': np.array([57.]), 'f0_std': np.array([ 8.85847662, 10.23340647, 13.69037296, 17.0589511 , 19.66149689]), 'f1_std': np.array([15.70555478, 16.54386649, 26.01619138, 36.61966778, 42.86666364]), 'spont_std': np.array([8.93782971]), 'si_std': np.array([0.10457719]), 'nsub': 2, 'N_class': 22},
    'TF2': {'f0_exp': np.array([ 8.61111111, 13.61111111, 10.27777778,  7.09722222, 6.04166667]), 'f1_exp': np.array([ 9.41890447, 15.09077151, 13.14080637, 10.04369494,  6.57126062]), 'spont_exp': np.array([3.53333333]), 'si_exp': np.array([0.46470802]), 'si_inf_exp': np.array([0.32941603]), 'ttp_exp': np.array([135.83333333]), 'f0_std': np.array([1.78551917, 5.97235142, 5.61145488, 1.14134221, 3.80948196]), 'f1_std': np.array([3.83658203, 7.83466567, 9.24700393, 3.09582968, 5.75298831]), 'spont_std': np.array([2.57940562]), 'si_std': np.array([0.05302861]), 'nsub': 3, 'N_class': 22},
    'TF4': {'f0_exp': np.array([ 9.30208333,  9.33333333, 14.89583333,  9.92708333, 9.36458333]), 'f1_exp': np.array([ 7.4975099 ,  9.07276215, 13.76520907, 13.05372143, 12.30888582]), 'spont_exp': np.array([8.25]), 'si_exp': np.array([0.49895115]), 'si_inf_exp': np.array([0.3979023]), 'ttp_exp': np.array([91.5]), 'f0_std': np.array([4.09612156, 4.28620202, 3.67116202, 6.00514585, 8.12904992]), 'f1_std': np.array([ 5.23035758,  7.01703864,  8.8075514 , 10.58334845, 12.53238647]), 'spont_std': np.array([9.20307919]), 'si_std': np.array([0.07644605]), 'nsub': 4, 'N_class': 22},
    'TF8': {'f0_exp': np.array([3.53703704, 3.71759259, 4.63888889, 7.56018519, 4.74537037]), 'f1_exp': np.array([ 4.22084346,  4.92086698,  7.07342571, 10.49387957,  6.23417685]), 'spont_exp': np.array([1.76666667]), 'si_exp': np.array([0.45602181]), 'si_inf_exp': np.array([0.31204362]), 'ttp_exp': np.array([117.5]), 'f0_std': np.array([2.81423129, 2.68586657, 3.55267896, 4.72191584, 3.80180903]), 'f1_std': np.array([3.9601909 , 4.09113397, 6.22904832, 7.86011041, 4.98051883]), 'spont_std': np.array([1.14127122]), 'si_std': np.array([0.09678899]), 'nsub': 9, 'N_class': 22}
}


sus_sus_expected = {
    'TF15': {'f0_exp': np.array([ 8.07638889,  9.40972222,  9.84722222, 10.49305555, 14.07638889]), 'f1_exp': np.array([ 8.95501002,  9.77289281, 10.55540273, 10.54994517, 12.14030272]), 'spont_exp': np.array([6.71777778]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([3.34893628, 2.87753176, 3.67302689, 4.17389419, 0.52050916]), 'f1_std': np.array([ 9.17744617,  9.4349988 ,  9.74257621, 11.16450793, 10.32961628]), 'spont_std': np.array([5.25773176]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 2, 'N_class': 10},
    'TF4': {'f0_exp': np.array([12.55      , 12.19166667, 16.29166667, 12.41666667,  9.1]), 'f1_exp': np.array([ 5.68730662,  5.72036282,  9.80131341, 10.16059654,  5.4483171 ]), 'spont_exp': np.array([4.54]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([7.83529452, 4.75087711, 7.48621418, 5.30730956, 6.56852237]), 'f1_std': np.array([2.31310551, 3.35754287, 6.42517563, 6.53521828, 3.56479966]), 'spont_std': np.array([4.35235568]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 5, 'N_class': 10},
    'TF8': {'f0_exp': np.array([ 5.66666667,  3.61904762,  8.19047619, 15.23809524,  3.9047619]), 'f1_exp': np.array([ 2.67724768,  1.8791828 ,  6.74283244, 12.24987804,  3.50948733]), 'spont_exp': np.array([2.62095238]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([[6.38964355, 6.38964355, 6.38964355, 6.38964355, 6.38964355]]), 'f1_std': np.array([[4.43916839, 4.43916839, 4.43916839, 4.43916839, 4.43916839]]), 'spont_std': np.array([[4.35235568, 4.35235568, 4.35235568, 4.35235568, 4.35235568]]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 1, 'N_class': 10}
}


trans_sus_expected = {
    'TF1': {'f0_exp': np.array([36., 21.04166667, 22.41666667, 23.54166667, 12.04166667]), 'f1_exp': np.array([ 8.85216752,  7.71744362,  8.79852287, 16.1074593 ,  8.09474585]), 'spont_exp': np.array([5.]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'f1_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'spont_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 1, 'N_class': 7},
    'TF15': {'f0_exp': np.array([3.20833333, 3.45833333, 3.79166667, 5.125, 5.70833333]), 'f1_exp': np.array([4.43147768, 4.62166585, 5.53742765, 8.081975  , 8.51808459]), 'spont_exp': np.array([3.3]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'f1_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'spont_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 1, 'N_class': 7},
    'TF2': {'f0_exp': np.array([3.16666667, 3.29166667, 3.25, 1.75, 2.29166667]), 'f1_exp': np.array([2.32621476, 3.50080051, 2.85302699, 2.12566901, 2.3698903 ]), 'spont_exp': np.array([1.]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'f1_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'spont_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 1, 'N_class': 7},
    'TF4': {'f0_exp': np.array([ 5.91666667,  7.95833333, 13.375,  6.58333333, 5.41666667]), 'f1_exp': np.array([5.24514294, 3.8352691 , 4.99435957, 3.69495923, 1.75006614]), 'spont_exp': np.array([4.4]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'f1_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'spont_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 1, 'N_class': 7},
    'TF8': {'f0_exp': np.array([10.29166667, 13.125, 19.16666667, 21.83333333, 17.29166667]), 'f1_exp': np.array([10.79237938, 19.16833868, 29.71094753, 36.04648112, 25.37906864]), 'spont_exp': np.array([11.2]), 'si_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_inf_exp': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'ttp_exp': np.array([[np.nan, np.nan]]), 'f0_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'f1_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'spont_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'si_std': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), 'nsub': 1, 'N_class': 7}
}


@pytest.mark.parametrize("cell_subclass,expected_val",
                         [
                             ('tOFF', tOFF_expected),
                             ('tON', tON_expected),
                             ('sOFF', sOFF_expected),
                             ('sON', sON_expected),
                             ('sus_sus', sus_sus_expected),
                             ('trans_sus', trans_sus_expected)
                         ])
def test_get_data_metrics(cell_subclass, expected_val):
    cell_metrics = get_data_metrics_for_each_subclass(cell_subclass)
    assert(cmp_dicts(cell_metrics, expected_val))

    # Makes sure the singleton is caching values correctly
    cell_metrics_cached = get_data_metrics_for_each_subclass(cell_subclass)
    assert(cmp_dicts(cell_metrics, expected_val))


if __name__ == '__main__':
    cell_metrics = get_data_metrics_for_each_subclass('trans_sus')
    print(cell_metrics)

    #cell_metrics2 = get_data_metrics_for_each_subclass('tON')
    #print(cmp_dicts(cell_metrics, cell_metrics2))

