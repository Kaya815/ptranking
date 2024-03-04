
import os
from scipy import stats
from scipy.stats import ttest_rel

from ptranking.utils.bigdata.BigPickle import pickle_load

def do_t_test(main_vec, side_vec, wil=True):
    #print('main_vec', main_vec)
    if wil:
        p = stats.wilcoxon(main_vec, side_vec)
    else:
        p = stats.ttest_rel(main_vec, side_vec)

    return p

def get_run_path(root, metric):
    for file in os.listdir(root):
        if file.find(metric+'@5') >= 0:
            sub_dir = os.path.join(root, file)
            break

    for file in os.listdir(sub_dir): # OptIdeal
        if not file.find('DS_Store') >= 0:
            run_dir = os.path.join(sub_dir, file)
            break

    return run_dir + '/'

def pairwise_type_t(metric_str, threshold,
                    t1_str, t1_p, t1_ap, t1_nerr, t1_ndcg, t2_str, t2_p, t2_ap, t2_nerr, t2_ndcg,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50]):
    for ind in cutoff_inds:
        print('{}-{}-vs-{}-@-{}'.format(metric_str, t1_str, t2_str, cutoff[ind]))

        p_value = do_t_test(t1_p[:, ind], t2_p[:, ind], wil=False)
        if p_value[1] < threshold:
            tab_str = '\t!\t'
            print(tab_str, 'P', p_value)

        p_value = do_t_test(t1_ap[:, ind], t2_ap[:, ind], wil=False)
        if p_value[1] < threshold:
            tab_str = '\t!\t'
            print(tab_str, 'AP', p_value)

        p_value = do_t_test(t1_nerr[:, ind], t2_nerr[:, ind], wil=False)
        if p_value[1] < threshold:
            tab_str = '\t!\t'
            print(tab_str, 'nERR', p_value)

        p_value = do_t_test(t1_ndcg[:, ind], t2_ndcg[:, ind], wil=False)
        if p_value[1] < threshold:
            tab_str = '\t!\t'
            print(tab_str, 'nDCG', p_value)


def pairwise_t(metric_str, threshold,
               t1_str, t1_ndcg, t2_str, t2_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50]):
    for ind in cutoff_inds:
        print('{}-{}-vs-{}-@-{}'.format(metric_str, t1_str, t2_str, cutoff[ind]))

        p_value = do_t_test(t1_ndcg[:, ind], t2_ndcg[:, ind], wil=False)
        if p_value[1] < threshold:
            tab_str = '\t!\t'
            print(tab_str, 'nDCG', p_value)



def check_t_test_type(cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50]):
    root = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/'

    # TwinSigST
    TwinSigST_root = root + 'TwinSigST/gpu_grid_TwinRank/'
    TwinSigST_p_per_q_ndcg = pickle_load(file= get_run_path(TwinSigST_root, metric='P') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    TwinSigST_ap_per_q_ndcg = pickle_load(file= get_run_path(TwinSigST_root, metric='AP')+ 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    TwinSigST_nerr_per_q_ndcg = pickle_load(file= get_run_path(TwinSigST_root, metric='nERR')+ 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    TwinSigST_ndcg_per_q_ndcg = pickle_load(file= get_run_path(TwinSigST_root, metric='nDCG')+ 'TwinRank_all_fold_ndcg_at_ks_per_q.np')

    TwinSigST_p_per_q_nerr = pickle_load(file=get_run_path(TwinSigST_root, metric='P') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    TwinSigST_ap_per_q_nerr = pickle_load(file=get_run_path(TwinSigST_root, metric='AP') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    TwinSigST_nerr_per_q_nerr = pickle_load(file=get_run_path(TwinSigST_root, metric='nERR') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    TwinSigST_ndcg_per_q_nerr = pickle_load(file=get_run_path(TwinSigST_root, metric='nDCG') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')

    # SignTwinSig
    SignTwinSig_root = root + 'SignTwinSig/gpu_grid_TwinRank/'
    SignTwinSig_p_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='P') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSig_ap_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='AP') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSig_nerr_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='nERR') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSig_ndcg_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='nDCG') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')

    SignTwinSig_p_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='P') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSig_ap_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='AP') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSig_nerr_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='nERR') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSig_ndcg_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSig_root, metric='nDCG') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')

    # SignTwinSigAmp
    SignTwinSigAmp_root = root + 'SignTwinSigAmp/gpu_grid_TwinRank/'
    SignTwinSigAmp_p_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='P') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_ap_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='AP') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_nerr_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nERR') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_ndcg_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nDCG') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')

    SignTwinSigAmp_p_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='P') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_ap_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='AP') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_nerr_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nERR') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_ndcg_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nDCG') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')

    threshold = 0.01
    pairwise_type_t(metric_str='nDCG', threshold=threshold,
                    t1_str='TwinSigST', t1_p=TwinSigST_p_per_q_ndcg, t1_ap=TwinSigST_ap_per_q_ndcg, t1_nerr=TwinSigST_nerr_per_q_ndcg, t1_ndcg=TwinSigST_ndcg_per_q_ndcg,
                    t2_str='SignTwinSig', t2_p=SignTwinSig_p_per_q_ndcg, t2_ap=SignTwinSig_ap_per_q_ndcg, t2_nerr=SignTwinSig_nerr_per_q_ndcg, t2_ndcg=SignTwinSig_ndcg_per_q_ndcg,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_type_t(metric_str='nERR', threshold=threshold,
                    t1_str='TwinSigST', t1_p=TwinSigST_p_per_q_nerr, t1_ap=TwinSigST_ap_per_q_nerr, t1_nerr=TwinSigST_nerr_per_q_nerr, t1_ndcg=TwinSigST_ndcg_per_q_nerr,
                    t2_str='SignTwinSig', t2_p=SignTwinSig_p_per_q_nerr, t2_ap=SignTwinSig_ap_per_q_nerr, t2_nerr=SignTwinSig_nerr_per_q_nerr, t2_ndcg=SignTwinSig_ndcg_per_q_nerr,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_type_t(metric_str='nDCG', threshold=threshold,
                    t1_str='TwinSigST', t1_p=TwinSigST_p_per_q_ndcg, t1_ap=TwinSigST_ap_per_q_ndcg, t1_nerr=TwinSigST_nerr_per_q_ndcg, t1_ndcg=TwinSigST_ndcg_per_q_ndcg,
                    t2_str='SignTwinSigAmp', t2_p=SignTwinSigAmp_p_per_q_ndcg, t2_ap=SignTwinSigAmp_ap_per_q_ndcg, t2_nerr=SignTwinSigAmp_nerr_per_q_ndcg, t2_ndcg=SignTwinSigAmp_ndcg_per_q_ndcg,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_type_t(metric_str='nERR', threshold=threshold,
                    t1_str='TwinSigST', t1_p=TwinSigST_p_per_q_nerr, t1_ap=TwinSigST_ap_per_q_nerr, t1_nerr=TwinSigST_nerr_per_q_nerr, t1_ndcg=TwinSigST_ndcg_per_q_nerr,
                    t2_str='SignTwinSigAmp', t2_p=SignTwinSigAmp_p_per_q_nerr, t2_ap=SignTwinSigAmp_ap_per_q_nerr, t2_nerr=SignTwinSigAmp_nerr_per_q_nerr, t2_ndcg=SignTwinSigAmp_ndcg_per_q_nerr,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_type_t(metric_str='nDCG', threshold=threshold,
                    t1_str='SignTwinSig', t1_p=SignTwinSig_p_per_q_ndcg, t1_ap=SignTwinSig_ap_per_q_ndcg, t1_nerr=SignTwinSig_nerr_per_q_ndcg, t1_ndcg=SignTwinSig_ndcg_per_q_ndcg,
                    t2_str='SignTwinSigAmp', t2_p=SignTwinSigAmp_p_per_q_ndcg, t2_ap=SignTwinSigAmp_ap_per_q_ndcg, t2_nerr=SignTwinSigAmp_nerr_per_q_ndcg, t2_ndcg=SignTwinSigAmp_ndcg_per_q_ndcg,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_type_t(metric_str='nERR', threshold=threshold,
                    t1_str='SignTwinSig', t1_p=SignTwinSig_p_per_q_nerr, t1_ap=SignTwinSig_ap_per_q_nerr, t1_nerr=SignTwinSig_nerr_per_q_nerr, t1_ndcg=SignTwinSig_ndcg_per_q_nerr,
                    t2_str='SignTwinSigAmp', t2_p=SignTwinSigAmp_p_per_q_nerr, t2_ap=SignTwinSigAmp_ap_per_q_nerr, t2_nerr=SignTwinSigAmp_nerr_per_q_nerr, t2_ndcg=SignTwinSigAmp_ndcg_per_q_nerr,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])


def check_t_test_mart():
    # lambdaMart
    mart_root = '/T2Root/dl-box/T2_WorkBench/Project_output/Out_L2R/Tree/LightGBMLambdaMART/MSLRWEB30K_MiD_10_MiR_1_TrBat_1_EarlyStop_200_QS_StandardScaler/BT_gbdt_Metric_ndcg_Leaves_400_Trees_1000_MiData_50_MSH_200_LR_0.05_EvalAt_5/'

    mart_ndcg = pickle_load(file=mart_root + 'MSLRWEB30K_LightGBMLambdaMART_all_fold_ndcg_at_ks_per_q.np')
    mart_nerr = pickle_load(file=mart_root + 'MSLRWEB30K_LightGBMLambdaMART_all_fold_err_at_ks_per_q.np')

    # SignTwinSigAmp
    root = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/'
    SignTwinSigAmp_root = root + 'SignTwinSigAmp/gpu_grid_TwinRank/'
    SignTwinSigAmp_p_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='P') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_ap_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='AP') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_nerr_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nERR') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_ndcg_per_q_ndcg = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nDCG') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')

    SignTwinSigAmp_p_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='P') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_ap_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='AP') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_nerr_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nERR') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_ndcg_per_q_nerr = pickle_load(
        file=get_run_path(SignTwinSigAmp_root, metric='nDCG') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')

    threshold = 0.01

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='SignTwinSigAmp-P', t2_ndcg=SignTwinSigAmp_p_per_q_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='SignTwinSigAmp-P', t2_ndcg=SignTwinSigAmp_p_per_q_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='SignTwinSigAmp-AP', t2_ndcg=SignTwinSigAmp_ap_per_q_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='SignTwinSigAmp-AP', t2_ndcg=SignTwinSigAmp_ap_per_q_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='SignTwinSigAmp-nERR', t2_ndcg=SignTwinSigAmp_nerr_per_q_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='SignTwinSigAmp-nERR', t2_ndcg=SignTwinSigAmp_nerr_per_q_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_t(metric_str='nDCG', threshold=threshold,
                    t1_str='MART', t1_ndcg=mart_ndcg,
                    t2_str='SignTwinSigAmp-nDCG', t2_ndcg=SignTwinSigAmp_ndcg_per_q_ndcg,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
                    t1_str='MART', t1_ndcg=mart_nerr,
                    t2_str='SignTwinSigAmp-nDCG', t2_ndcg=SignTwinSigAmp_ndcg_per_q_nerr,
                    cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])


def check_t_test_mart_type(type_root, type):
    # lambdaMart
    mart_root = '/T2Root/dl-box/T2_WorkBench/Project_output/Out_L2R/Tree/LightGBMLambdaMART/MSLRWEB30K_MiD_10_MiR_1_TrBat_1_EarlyStop_200_QS_StandardScaler/BT_gbdt_Metric_ndcg_Leaves_400_Trees_1000_MiData_50_MSH_200_LR_0.05_EvalAt_5/'

    mart_ndcg = pickle_load(file=mart_root + 'MSLRWEB30K_LightGBMLambdaMART_all_fold_ndcg_at_ks_per_q.np')
    mart_nerr = pickle_load(file=mart_root + 'MSLRWEB30K_LightGBMLambdaMART_all_fold_err_at_ks_per_q.np')

    SignTwinSigAmp_p_per_q_ndcg = pickle_load(
        file=get_run_path(type_root, metric='P') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_ap_per_q_ndcg = pickle_load(
        file=get_run_path(type_root, metric='AP') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_nerr_per_q_ndcg = pickle_load(
        file=get_run_path(type_root, metric='nERR') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')
    SignTwinSigAmp_ndcg_per_q_ndcg = pickle_load(
        file=get_run_path(type_root, metric='nDCG') + 'TwinRank_all_fold_ndcg_at_ks_per_q.np')

    SignTwinSigAmp_p_per_q_nerr = pickle_load(
        file=get_run_path(type_root, metric='P') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_ap_per_q_nerr = pickle_load(
        file=get_run_path(type_root, metric='AP') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_nerr_per_q_nerr = pickle_load(
        file=get_run_path(type_root, metric='nERR') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')
    SignTwinSigAmp_ndcg_per_q_nerr = pickle_load(
        file=get_run_path(type_root, metric='nDCG') + 'TwinRank_all_fold_nerr_at_ks_per_q.np')

    threshold = 0.01

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='{}-P'.format(type), t2_ndcg=SignTwinSigAmp_p_per_q_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='{}-P'.format(type), t2_ndcg=SignTwinSigAmp_p_per_q_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='{}-AP'.format(type), t2_ndcg=SignTwinSigAmp_ap_per_q_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='{}-AP'.format(type), t2_ndcg=SignTwinSigAmp_ap_per_q_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='{}-nERR'.format(type), t2_ndcg=SignTwinSigAmp_nerr_per_q_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='{}-nERR'.format(type), t2_ndcg=SignTwinSigAmp_nerr_per_q_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='{}-nDCG'.format(type), t2_ndcg=SignTwinSigAmp_ndcg_per_q_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='{}-nDCG'.format(type), t2_ndcg=SignTwinSigAmp_ndcg_per_q_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

def cmp_type_mart():
    root = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/'

    # TwinSigST
    TwinSigST_root = root + 'TwinSigST/gpu_grid_TwinRank/'
    check_t_test_mart_type(TwinSigST_root, 'TwinSigST')

    # SignTwinSig
    SignTwinSig_root = root + 'SignTwinSig/gpu_grid_TwinRank/'
    check_t_test_mart_type(SignTwinSig_root, 'SignTwinSig')

    # SignTwinSigAmp
    SignTwinSigAmp_root = root + 'SignTwinSigAmp/gpu_grid_TwinRank/'
    check_t_test_mart_type(SignTwinSigAmp_root, 'SignTwinSigAmp')

def cmp_listnet_mart():
    # lambdaMart
    mart_root = '/T2Root/dl-box/T2_WorkBench/Project_output/Out_L2R/Tree/LightGBMLambdaMART/MSLRWEB30K_MiD_10_MiR_1_TrBat_1_EarlyStop_200_QS_StandardScaler/BT_gbdt_Metric_ndcg_Leaves_400_Trees_1000_MiData_50_MSH_200_LR_0.05_EvalAt_5/'

    mart_ndcg = pickle_load(file=mart_root + 'MSLRWEB30K_LightGBMLambdaMART_all_fold_ndcg_at_ks_per_q.np')
    mart_nerr = pickle_load(file=mart_root + 'MSLRWEB30K_LightGBMLambdaMART_all_fold_err_at_ks_per_q.np')

    # listnet
    net_root = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/Results/gpu_grid_ListNet/ListNet_SF_R5R_BN_Affine_Adam_0.0001_MSLRWEB30K_MiD_10_MiR_1_TrBat_100_TrPresort_EP_300_V_nDCG@5_QS_StandardScaler/'
    net_ndcg = pickle_load(file=net_root + 'ListNet_all_fold_ndcg_at_ks_per_q.np')
    net_nerr = pickle_load(file=net_root + 'ListNet_all_fold_nerr_at_ks_per_q.np')

    threshold = 0.01

    pairwise_t(metric_str='nDCG', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_ndcg,
               t2_str='ListNet', t2_ndcg=net_ndcg,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])
    pairwise_t(metric_str='nERR', threshold=threshold,
               t1_str='MART', t1_ndcg=mart_nerr,
               t2_str='ListNet', t2_ndcg=net_nerr,
               cutoff_inds=[0, 2, 3, 4, 5], cutoff=[1, 3, 5, 10, 20, 50])

"""
/home/user/anaconda3/envs/pytorch18/bin/python "/home/user/Workbench/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/drl_ptranking/experiment/www_t_test.py"
nDCG-TwinSigST-vs-SignTwinSig-@-1
	!	 nDCG Ttest_relResult(statistic=-3.692530711424071, pvalue=0.00022242321980393598)
nDCG-TwinSigST-vs-SignTwinSig-@-5
	!	 P Ttest_relResult(statistic=-3.742364599451741, pvalue=0.00018263325313210534)
	!	 AP Ttest_relResult(statistic=-4.063887371343579, pvalue=4.8383752308663974e-05)
	!	 nERR Ttest_relResult(statistic=-3.913325543435709, pvalue=9.123253495200434e-05)
	!	 nDCG Ttest_relResult(statistic=-4.234414952422955, pvalue=2.298226909018343e-05)
nDCG-TwinSigST-vs-SignTwinSig-@-10
	!	 P Ttest_relResult(statistic=-6.341856711371892, pvalue=2.302114875978916e-10)
	!	 AP Ttest_relResult(statistic=-6.444399564305007, pvalue=1.178010232591995e-10)
	!	 nERR Ttest_relResult(statistic=-3.9455289875157193, pvalue=7.980340846300086e-05)
	!	 nDCG Ttest_relResult(statistic=-5.635741010670784, pvalue=1.7585428794337505e-08)
nDCG-TwinSigST-vs-SignTwinSig-@-20
	!	 P Ttest_relResult(statistic=-7.504008934185042, pvalue=6.359258277672309e-14)
	!	 AP Ttest_relResult(statistic=-5.858653843667289, pvalue=4.714534052843196e-09)
	!	 nERR Ttest_relResult(statistic=-3.0978997919827878, pvalue=0.0019507552526301065)
	!	 nDCG Ttest_relResult(statistic=-7.837516979970911, pvalue=4.745343930909855e-15)
nDCG-TwinSigST-vs-SignTwinSig-@-50
	!	 P Ttest_relResult(statistic=-11.220298906540133, pvalue=3.7010233240896524e-29)
	!	 AP Ttest_relResult(statistic=-5.583672064576608, pvalue=2.375084587016509e-08)
	!	 nERR Ttest_relResult(statistic=-3.467208293032594, pvalue=0.0005266235116073644)
	!	 nDCG Ttest_relResult(statistic=-9.30102660296534, pvalue=1.4815007667438007e-20)
nERR-TwinSigST-vs-SignTwinSig-@-1
	!	 nDCG Ttest_relResult(statistic=-3.692530711424071, pvalue=0.00022242321980393598)
nERR-TwinSigST-vs-SignTwinSig-@-5
	!	 P Ttest_relResult(statistic=-3.7315689111281216, pvalue=0.00019063839121368005)
	!	 AP Ttest_relResult(statistic=-3.7838527122894643, pvalue=0.00015471633739774055)
	!	 nDCG Ttest_relResult(statistic=-5.295109973844639, pvalue=1.1977312312082008e-07)
nERR-TwinSigST-vs-SignTwinSig-@-10
	!	 P Ttest_relResult(statistic=-4.271632620129777, pvalue=1.9463824415003417e-05)
	!	 AP Ttest_relResult(statistic=-4.216686549871589, pvalue=2.4863713763577453e-05)
	!	 nDCG Ttest_relResult(statistic=-5.537282797242153, pvalue=3.0974594155406876e-08)
nERR-TwinSigST-vs-SignTwinSig-@-20
	!	 P Ttest_relResult(statistic=-4.374510738584921, pvalue=1.2211117699410247e-05)
	!	 AP Ttest_relResult(statistic=-4.239134837802183, pvalue=2.250468404966705e-05)
	!	 nDCG Ttest_relResult(statistic=-5.8093438117401535, pvalue=6.334484599134265e-09)
nERR-TwinSigST-vs-SignTwinSig-@-50
	!	 P Ttest_relResult(statistic=-4.860883568364617, pvalue=1.1744741145565793e-06)
	!	 AP Ttest_relResult(statistic=-3.99339589155082, pvalue=6.528729904203816e-05)
	!	 nDCG Ttest_relResult(statistic=-5.97241588152519, pvalue=2.363686240013923e-09)
nDCG-TwinSigST-vs-SignTwinSigAmp-@-1
	!	 AP Ttest_relResult(statistic=-5.053970734206838, pvalue=4.3523335927214996e-07)
	!	 nERR Ttest_relResult(statistic=-3.9470860844535975, pvalue=7.928663335058751e-05)
	!	 nDCG Ttest_relResult(statistic=-7.373822040861881, pvalue=1.7004575563318256e-13)
nDCG-TwinSigST-vs-SignTwinSigAmp-@-5
	!	 P Ttest_relResult(statistic=-8.695190870044664, pvalue=3.6339387509226926e-18)
	!	 AP Ttest_relResult(statistic=-10.10031818346084, pvalue=6.009036726137785e-24)
	!	 nERR Ttest_relResult(statistic=-6.039108494494688, pvalue=1.5677028938035205e-09)
	!	 nDCG Ttest_relResult(statistic=-18.748318758415408, pvalue=5.527504712441385e-78)
nDCG-TwinSigST-vs-SignTwinSigAmp-@-10
	!	 P Ttest_relResult(statistic=-13.761392558743706, pvalue=5.860094636861258e-43)
	!	 AP Ttest_relResult(statistic=-13.903264120538982, pvalue=8.252911090291104e-44)
	!	 nERR Ttest_relResult(statistic=-9.350602971887458, pvalue=9.294651497963863e-21)
	!	 nDCG Ttest_relResult(statistic=-26.511891052500122, pvalue=3.9632280336054306e-153)
nDCG-TwinSigST-vs-SignTwinSigAmp-@-20
	!	 P Ttest_relResult(statistic=-16.517214723921377, pvalue=5.102414014916645e-61)
	!	 AP Ttest_relResult(statistic=-16.04979556985891, pvalue=9.92813294975853e-58)
	!	 nERR Ttest_relResult(statistic=-11.497245079456485, pvalue=1.5759426456115782e-30)
	!	 nDCG Ttest_relResult(statistic=-32.62968869437767, pvalue=1.4773538224700503e-229)
nDCG-TwinSigST-vs-SignTwinSigAmp-@-50
	!	 P Ttest_relResult(statistic=-19.595171508877804, pvalue=5.71727400202063e-85)
	!	 AP Ttest_relResult(statistic=-19.547484222243057, pvalue=1.4404431479165622e-84)
	!	 nERR Ttest_relResult(statistic=-15.0492805308171, pvalue=5.337965510770685e-51)
	!	 nDCG Ttest_relResult(statistic=-37.27453285483185, pvalue=2.2471043948677514e-297)
nERR-TwinSigST-vs-SignTwinSigAmp-@-1
	!	 AP Ttest_relResult(statistic=-5.053970734206838, pvalue=4.3523335927214996e-07)
	!	 nERR Ttest_relResult(statistic=-3.9470860844535975, pvalue=7.928663335058751e-05)
	!	 nDCG Ttest_relResult(statistic=-7.373822040861881, pvalue=1.7004575563318256e-13)
nERR-TwinSigST-vs-SignTwinSigAmp-@-5
	!	 P Ttest_relResult(statistic=-5.372296617905099, pvalue=7.831276614619727e-08)
	!	 AP Ttest_relResult(statistic=-7.852302321732497, pvalue=4.218945872473876e-15)
	!	 nERR Ttest_relResult(statistic=-5.787549303709431, pvalue=7.212395015834745e-09)
	!	 nDCG Ttest_relResult(statistic=-12.986047829892577, pvalue=1.8600284119477608e-38)
nERR-TwinSigST-vs-SignTwinSigAmp-@-10
	!	 P Ttest_relResult(statistic=-5.778008164266144, pvalue=7.632963381074627e-09)
	!	 AP Ttest_relResult(statistic=-8.428029660515257, pvalue=3.6688290793700706e-17)
	!	 nERR Ttest_relResult(statistic=-6.593926551515732, pvalue=4.353813764306141e-11)
	!	 nDCG Ttest_relResult(statistic=-13.906704010678002, pvalue=7.867925275369819e-44)
nERR-TwinSigST-vs-SignTwinSigAmp-@-20
	!	 P Ttest_relResult(statistic=-5.856729530121692, pvalue=4.769400797598812e-09)
	!	 AP Ttest_relResult(statistic=-8.66787905178451, pvalue=4.6177007794398835e-18)
	!	 nERR Ttest_relResult(statistic=-6.662506361771362, pvalue=2.7380384285033132e-11)
	!	 nDCG Ttest_relResult(statistic=-14.112840166775937, pvalue=4.3998795755697287e-45)
nERR-TwinSigST-vs-SignTwinSigAmp-@-50
	!	 P Ttest_relResult(statistic=-6.285155146673129, pvalue=3.3198184135693277e-10)
	!	 AP Ttest_relResult(statistic=-9.03939099276249, pvalue=1.6667791438084178e-19)
	!	 nERR Ttest_relResult(statistic=-6.508254237575557, pvalue=7.721556540002507e-11)
	!	 nDCG Ttest_relResult(statistic=-14.1631915204021, pvalue=2.1616500748792022e-45)
nDCG-SignTwinSig-vs-SignTwinSigAmp-@-1
	!	 AP Ttest_relResult(statistic=-3.303596858695946, pvalue=0.0009556360512218301)
	!	 nERR Ttest_relResult(statistic=-4.563121750380826, pvalue=5.059621040582302e-06)
	!	 nDCG Ttest_relResult(statistic=-5.262435099285113, pvalue=1.4312633621534972e-07)
nDCG-SignTwinSig-vs-SignTwinSigAmp-@-5
	!	 P Ttest_relResult(statistic=-6.050000267595912, pvalue=1.4654207320108184e-09)
	!	 AP Ttest_relResult(statistic=-6.144000251469379, pvalue=8.146993414709828e-10)
	!	 nERR Ttest_relResult(statistic=-3.0156500629863787, pvalue=0.0025664148706920696)
	!	 nDCG Ttest_relResult(statistic=-16.988503126906664, pvalue=1.9874495428954542e-64)
nDCG-SignTwinSig-vs-SignTwinSigAmp-@-10
	!	 P Ttest_relResult(statistic=-9.438501580557748, pvalue=4.042816390176004e-21)
	!	 AP Ttest_relResult(statistic=-7.802661857580274, pvalue=6.2557310971220045e-15)
	!	 nERR Ttest_relResult(statistic=-6.170190959773985, pvalue=6.907051398788167e-10)
	!	 nDCG Ttest_relResult(statistic=-24.400897614884077, pvalue=3.0330447456389485e-130)
nDCG-SignTwinSig-vs-SignTwinSigAmp-@-20
	!	 P Ttest_relResult(statistic=-11.581637396052466, pvalue=5.9339192890973115e-31)
	!	 AP Ttest_relResult(statistic=-10.0399508474616, pvalue=1.1078490187378288e-23)
	!	 nERR Ttest_relResult(statistic=-8.848142958072295, pvalue=9.371654612173003e-19)
	!	 nDCG Ttest_relResult(statistic=-29.826058588676787, pvalue=1.1022196871774163e-192)
nDCG-SignTwinSig-vs-SignTwinSigAmp-@-50
	!	 P Ttest_relResult(statistic=-12.229688683855743, pvalue=2.5998917058912415e-34)
	!	 AP Ttest_relResult(statistic=-12.944999737386576, pvalue=3.167607789230477e-38)
	!	 nERR Ttest_relResult(statistic=-12.037540773515577, pvalue=2.687492391244296e-33)
	!	 nDCG Ttest_relResult(statistic=-34.26505130835653, pvalue=1.738721221971285e-252)
nERR-SignTwinSig-vs-SignTwinSigAmp-@-1
	!	 AP Ttest_relResult(statistic=-3.303596858695946, pvalue=0.0009556360512218301)
	!	 nERR Ttest_relResult(statistic=-4.563121750380826, pvalue=5.059621040582302e-06)
	!	 nDCG Ttest_relResult(statistic=-5.262435099285113, pvalue=1.4312633621534972e-07)
nERR-SignTwinSig-vs-SignTwinSigAmp-@-5
	!	 P Ttest_relResult(statistic=-2.7099411028645575, pvalue=0.00673330201582995)
	!	 AP Ttest_relResult(statistic=-4.365316549034198, pvalue=1.2735891189898491e-05)
	!	 nERR Ttest_relResult(statistic=-5.008171105000437, pvalue=5.525805163613879e-07)
	!	 nDCG Ttest_relResult(statistic=-10.465405121215513, pvalue=1.3771696981489066e-25)
nERR-SignTwinSig-vs-SignTwinSigAmp-@-10
	!	 P Ttest_relResult(statistic=-2.7292612775435625, pvalue=0.00635130263913771)
	!	 AP Ttest_relResult(statistic=-4.596428485323093, pvalue=4.315275707962023e-06)
	!	 nERR Ttest_relResult(statistic=-5.952037797329288, pvalue=2.677337593412175e-09)
	!	 nDCG Ttest_relResult(statistic=-11.305674135739704, pvalue=1.410022422909897e-29)
nERR-SignTwinSig-vs-SignTwinSigAmp-@-20
	!	 P Ttest_relResult(statistic=-2.7329310461228866, pvalue=0.006280987949474046)
	!	 AP Ttest_relResult(statistic=-4.798185219998995, pvalue=1.6087131925518946e-06)
	!	 nERR Ttest_relResult(statistic=-6.188021883961022, pvalue=6.170344952694756e-10)
	!	 nDCG Ttest_relResult(statistic=-11.36863876104699, pvalue=6.888956681830037e-30)
nERR-SignTwinSig-vs-SignTwinSigAmp-@-50
	!	 P Ttest_relResult(statistic=-2.800001370974559, pvalue=0.005113473136133231)
	!	 AP Ttest_relResult(statistic=-5.311872908158556, pvalue=1.0926920743337901e-07)
	!	 nERR Ttest_relResult(statistic=-6.009998959043991, pvalue=1.8764168512322078e-09)
	!	 nDCG Ttest_relResult(statistic=-11.37388552162763, pvalue=6.48867511265616e-30)

Process finished with exit code 0

"""

if __name__ == '__main__':
    #1
    check_t_test_type()

    #2å
    #check_t_test_mart()

    #3
    #cmp_type_mart()

    #4
    #cmp_listnet_mart()




