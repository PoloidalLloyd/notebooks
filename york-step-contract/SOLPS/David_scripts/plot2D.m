% balfile = '/home/david/Dropbox/CCFE/WP-ADC/Meeting2020/crazycore.nc';
% balfile = '/home/david/Dropbox/CCFE/solps-iter/STEP/Sarah/P100n20e19gp1e21gpAr3e14_nE_c49/P100n20e19gp1e21gpAr3e14_nE_c49_balance.nc';
% balfile = '/media/david/My Passport/SOLPS-ITER-EPSRUNS/mastu_eq5_D+C_vcompare/ref_Hmode_12e21_cont_cont_cont_cont_cont_cont_cont_cont_balanceav1000/balance.nc';
% balfile = '/home/david/remote/freia/projects/physics/SOLPS/dmoulton/mastu_eq5_D+C_vcompare/ref_Hmode_12e21_cont_cont_cont_cont_cont_cont_cont_cont_balanceav1000/balance.nc';
% balfile = '/media/david/My Passport/SOLPS-ITER-OMYATRA/HFS_m_1e21_cont_cont_bal_av/balance.nc';
% balfile = '/media/david/My Passport/SOLPS-ITER-OMYATRA/HFS_m_3p2e21_bal_av/balance.nc';
% balfile = '/media/david/My Passport/SOLPS-ITER-WPADCRUNS/ref_puff23e21_Pin5_cryo_nn/balance.nc';
% balfile = '/home/david/remote/marconi/marconi_work/FUA34_ADC/dmoulton/solps-iter/runs/mastu_eq5/Dscan_D+C/ref_puff2e21/balance.nc';

% balfile = '/home/david/remote/cumulus/lustre/home/dmoulton/solps-iter/runs/mastu_eq5/D+C/puff12e21_Pin5_cryo_nn_OMP_drift/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_cd_45470_420ms/puff=12.0e21_pump=0.001_drifts_divchiconstfix_bcmom2/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/mastu_sxd_46860_450ms_nick/puff=0.5e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_noionmolel/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-osaw1/solps-runs/runs/step/step2021_v10/drsep0mm/stepv10releasedlowArWwall100source_CMLS_cont/P100nfl1p75gp3e23Ar8e21_nohi_rsnewpumpP100nfl1p75gp3e23gpAr8e21c42_c24/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-osaw1/solps-runs/runs/step/step2021_v10/drsep2mm/ArDpuff_corrpump_newparameters/P100nu1p75e22Ar1e22Dpuff3e23_balanceadded_rerun_cflim/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_307/runs/step/stepv10/stepv10releasedlowArWwall/droppfrwall/balance.nc';
% balfile = '/home/david/remote/cumulus/lustre/home/snewton/SOLPS/solps-iter_STEP_v10March2021/solps-iter/runs/stepv10/stepv10releasedlowArWwall100source/nopP100nfl1p75gp2p75e23gpAr3e22_rsoldpumpnopP100nfl1p75gp2p75e23gpAr3e22c42_c14/balance.nc';

% balfile = '/home/david/remote/cumulus/lustre/home/dmoulton/solps-iter/runs/mastu_eq5/D_driftoptimise_narrowX_widenosolmore/puff10e21IMP_cc0p03_drift1p0_visper1p0_corrcore0p005_vsa0p2_nringsfornoav7/balance.nc';
% balfile = '/tmp/balance.nc';
% balfile = '/media/david/My Passport/SOLPS-ITER-EPSRUNS/mastu_eq1_D+C_vcompare/ref_Hmode_12_pumpfix_cont_cont_cont_cont_balanceav1000/balance.nc';
% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_cd_45470_420ms/puff=1.0e21_pump=0.001_drifts_divchiconst_pufflfs/balance.nc';
% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_45456_445ms/puff=9.0e21_pump=0.001/balance.nc';
% balfile = '/home/david/remote/cumulus/lustre/home/dmoulton/solps-iter_master/runs/mastu_sxd_45456_445ms/puff=3.0e21_pump=0.001_mr/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms_Donly/puff=13.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc';
% balfile = 'Y:\physics\SOLPS\dmoulton\mastu_sxd_46860_450ms\puff=13.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans_pufflfs_pin1.0\balance.nc';
% simdbdir = 'Z:\simdb\simulations\aliases\dmoulton\solps-iter\mastu\45456\Dec1523\seq-4\'; a=dir([simdbdir,'*\balance.nc']); balfile=[a.folder,'\',a.name];
% balfile = '/home/david/Desktop/hpciter/work/projects/solps-iter/simpsoj/iter/narrow_SOL/123063_ref_recover/balance.nc';

% balfile = '\\wsl.localhost\Ubuntu\home\dmoulton\remote\gateway\pfs\work\g2dmoult\solps-iter_develop\runs\mastu_sxd_46860_450ms\puff=13.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_molcxshift\balance.nc';
% balfile = fullfile('\\wsl.localhost','Ubuntu','home','dmoulton','remote','gateway','pfs','work','g2dmoult','solps-iter_develop','runs','mastu_sxd_46860_450ms','puff=13.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0','balance.nc');
balfile = fullfile('\\wsl.localhost','Ubuntu','home','dmoulton','remote','csd3_ap001','ir-moul2','solps-runs','mastu_sxd_46860_450ms_fluxtube','volheat0p5MW_n02e19','balance.nc');
% balfile = 'C:\Temp\balance.nc.source';
% balfile = fullfile('\\wsl.localhost','Ubuntu','home','dmoulton','remote','gateway','pfs','work','g2dmoult','solps-iter_3.0.9_master','runs','tube1d_2targets_Bconst_compress_dy0p01','twotar_s0at0p8_Pin10MW_n02E19\balance.nc');

% balfile = '\\wsl.localhost\Ubuntu\home\dmoulton\remote\hpciter\home\ITER\moultod\solps-runs\123063_ref_beryllium_matchcne\balance.nc';

% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2xiang8/solps-iter-mastu/runs/mastu/fix_pfr_trans_coeff/mastu_cd_drifts/puff=2.0e21_pump=0.001/balance.nc';
% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/cumulus/lustre/home/dmoulton/solps-iter_master/runs/mastu_sxd_45456_445ms/puff=9.0e21_pump=0.001_mr/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=6.3e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_molcxstijn/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_45456_445ms/puff=9.0e21_pump=0.001_nodrifts/balance.nc';

% balfile = '/home/david/remote/csd3_rdsukaea/ir-osaw1/solps-iter_b2stel/runs/step/step2022_SPR45/step2022_SPR45_newpumpdesign_X_smallergap_onlyAr/P180nu1p6e22Aro1e21i1e16Dpuff1e22_Clq0p5_fromscratch/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/step2022_SPR45/step2022_SPR45_newpumpdesign_X_smallergap_onlyAr/P180nu1p6e22Aro1e21i1e16Dpuff1e22_Clq0p5_fromscratch/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/step2022_SPR45/step2022_SPR45_newpumpdesign_X_smallergap_onlyAr/P180nu1p6e22Aro1e21i1e16Dpuff1e22_Clq0p5_fromscratch_copyfort/balance.nc';
% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_cd_45470_420ms/puff=1.0e21_pump=0.001_drifts_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2/balance.nc';

% balfile = '/home/david/remote/csd3_rdsukaea/ir-osaw1/solps-iter_b2stel/runs/step/step2022_SPR45/step2022_SPR45_widersdw_modifyouteralso+innerclosed+wideCoreHor+corrDuct_pureD_01/P150nu1p6e22Dpuff5e22_Clq0p5_cz0p005_num_startfromSarahs_cflim_1e6r/balance.nc';

% balfile = '/home/david/remote/gateway/pfs/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms_redpfr/puff=5.3e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans_molcx_nhist100_nring12/balance.nc';
% balfile = '/home/david/remote/gateway/pfs/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=4.3e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc.i';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_45456_445ms/puff=5.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc';

% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_cd_45470_420ms/puff=18.0e21_pump=0.001_drifts_divchiconst_pufflfs_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_45456_445ms/puff=7.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc';

% balfile = '/home/david/remote/gateway/pfs/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=15.3e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans/balance.nc';
% balfile = '/home/david/remote/gateway/pfs/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_molcxshift/balance.nc.i';

% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_45456_445ms/puff=4.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc';

% balfile = '/tmp/balance.nc';
% f44file = '/tmp/fort.44';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc.i';
% balfile = '/home/david/Desktop/hpciter/work/projects/solps-iter/moultod/iter/narrow_SOL/123063_ref_recover/balance.nc';
% balfile = '/home/david/remote/gateway/pfs/work/g2dmoult/solps-iter_develop/runs/LinearExpansion/ref/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/step2022_SPR45/step2022_SPR45_newpumpdesign_X_fromSarah_addHe_corrpuff/D6e22Ar1p2e20/balance.nc.i';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/step2022_SPR45/step2022_SPR45_newpumpdesign_X_fromSarah_addHe_corrpuff_modgridbyhand2/D6e22Ar1p2e20_bcpot2edge/balance.nc';

% balfile = '/home/david/remote/gateway/pfs/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_45456_445ms/puff=7.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2xiang8/solps-iter-master-3.0.8/runs/step/correct_pump/kaveeva_speedup/core=4cm/hfs_gap=0.8mm/warmstart/D=1.8e22_AR=1.0e20_RECYCTAR=0.97953_from_D=1.3_AR=1.0/balance.nc';

% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/step2022_SPR45/step2022_SPR45_newpumpdesign_X_fromSarah_addHe_corrpuff/G0p725e22D2e22D2e22Ar8e19Ar8e19P200_p12_cflim_b2neut_corrRECYCT_fromwhereIstartedsp50_morehistHeallr_toDavid/balance.nc';
% balfile = '/tmp/balance.nc';
% MOL TEMP:
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=25.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=29.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_nomolel1step/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=29.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_noD2onD/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=29.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_nomolel1step_noD2onD/balance.nc';

% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_molcxstijn/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc';

% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_ed_47079_540ms/puff=4.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=4.3e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans/balance.nc';

% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc';
% balfile = '/home/david/remote/gateway/pfs/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=13.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans_pufflfs_pin1.0_fastsidesandtarget/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2xiang8/solps-iter-mastu/runs/mast_cumulus/fixed_core_density_mastu_bc/core_ne=1.0e19_pin=1.0mw/drsep=-3.12cm/drsep=-3.12/balance.nc';
% balfile = '/home/david/remote/hpciter/home/ITER/moultod/balance_123063';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2xiang8/solps-iter/runs/DEMO_SXD_meshes/grid_2pi_corr/balance.nc';
% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=1.3e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/stepv10releasedlowArWwall100source_CMLS/P100nfl1p75gp3e23Ar8e21_rsnewpumpP100nfl1p75gp2p5e23gpAr8e21c28_c42_energydist/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/balance_123063_iter.nc';
% balfile = '/media/david/MyPassport/SOLPS-ITER_MIRROR/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_cd_45470_420ms/puff=10.0e21_pump=0.001_drifts_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_pin2.0_redoutpfrtrans/balance.nc';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-karh1/solps-runs/SPR45_evolving_Ar/D1e22_Ar3e20-1e15_drift_rampup_ahn=1e-5_dt=1e-7_eirene_mod=10_corr_to_pot_off_lluciani=3_feedback=1e-3_nstg2=1_user_transport=6_model_sig=1_cfsigalf=8e-5core_2e-5SOL_bcpot=2NS/balance.nc';
% balfile = '/home/david/remote/cumulus/lustre/home/rtatsumi/SOLPS-ITER_develop/solps-iter/runs/step/drsep2mm/ArDpuff_corrpump_newparameters_drifts_He_Bcorrected/P100nu1p75e22Ar3e21Dpuff2e23_drifts_He_pot0_0.3_nospeedupr_startedfromPET_5em6r_708test/balance.nc';
% balfile = '/home/david/remote/cumulus/lustre/home/dmoulton/solps-iter/runs/mastu_eq5/D_driftoptimise_narrowX_widenosolmore/drift_bcflux_visper1p0_hcie2_pin2p5/balance.nc';
% balfile = '/home/david/remote/cumulus/lustre/home/dmoulton/solps-iter/runs/step_he_200721/drift_Dpuff3E23_Nepuff4E21_test_0p2/balance.nc';
% balfile = '/home/david/Desktop/hpciter/work/projects/solps-iter/moultod/iter/narrow_SOL/123063_ref_beryllium_matchcne/balance.nc';

% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/step2022_SPR45/step2022_SPR45_newpumpdesign_X_fromSarah_addHe_corrpuff/G0p725e22D2e22D2e22Ar1p2e20Ar1p2e20P200_p12_cflim_b2neut_corrRECYCT_fromwhereIstartedsp50_morehistHeall_SAr48r_toDavid/balance.nc.i';
% balfile = '/home/david/remote/csd3_rdsukaea/ir-moul2/solps-runs/step2022_SPR45/step2022_SPR45_newpumpdesign_X_fromSarah_addHe_corrpuff/G0p725e22D0p5e22D0p5e22Ar3e20Ar1e15_p12_cflim_b2neut_corrRECYCT_fromwhereIstartedsp50_morehistDHeallr/balance.nc.i';

% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2jbry/solps-iter/runs/BOX/slab_45deg/ref_slab_old_45deg_q20MWm2_backup/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2jbry/solps-iter/runs/BOX/slab_45deg/ref_slab_old_45deg_q20MWm2/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2jbry/solps-iter/runs/BOX/slab_45deg/ref_slab_old_45deg_q20MWm2_yacfull_eff_v2_backup/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2jbry/solps-iter/runs/BOX/slab_45deg/ref_slab_old_45deg_q20MWm2_yacfull_eff_v2/balance.nc';
% balfile = '/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_46860_450ms/puff=25.0e21_pump=0.001_nodrifts_bcmom2_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc';
balfile = '/users/jlb647/scratch/simulation_program/hermes-3_sim/simulation_dir/2025-10_SOLPS_analysis/puff=13.0e21_pump=0.001_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_redoutpfrtrans_pufflfs_pin1.0/balance.nc'

%destfile = '/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/SOLPS/David_scripts/balance.nc';
%copyfile(balfile,destfile)
%balfile = destfile;
%% parallel current:
% % plotvar =
% % flipud(dlmread('/home/david/remote/hpciter/work/projects/solps-iter/moultod/iter/narrow_SOL/123063_ref_eirenestepcpu60/output/b2tfhe__fch_px.dat','',1,1))';
% fch_p = ncread(balfile,'fch_p');
% gs = ncread(balfile,'gs');
% plotvar = fch_p(:,:,1)./repmat(gs(end,:,1),[size(gs,1),1]);
% % leftix = ncread(balfile,'leftix')+1;
% % xcut = find(diff(leftix(:,1))<1);
% % sep = ncread(balfile,'jsep')+1;
% % plotvar([xcut(1):xcut(2)-1,xcut(4)+1:xcut(5)],1:sep+1)=nan;
%% Thermoelectric coefficient:
% fhe_thermj = ncread(balfile,'fhe_thermj');
% fch_p = ncread(balfile,'fch_p');
% te = ncread(balfile,'te')/1.602e-19;
% te_left = zeros(size(te));
% leftix = ncread(balfile,'leftix')+2;
% leftiy = ncread(balfile,'leftiy')+2;
% for ix=1:size(te,1)
%     for iy=1:size(te,2)
%         if leftix(ix,iy)~=0
%             te_left(ix,iy) = 0.5*(te(ix,iy)+te(leftix(ix,iy),leftiy(ix,iy)));
%         end
%     end
% end
% plotvar = fhe_thermj(:,:,1)./fch_p(:,:,1)./te_left;
%% dib2:
% te = ncread(balfile,'te')/1.602e-19;
% vol = ncread(balfile,'vol');
% nx = size(te,1);
% ny = size(te,2);
% tmp = ncread(balfile,'dmb2');
% nH2 = tmp(1:end-5,:); % SN: tmp(1:end-2,:); CDN: tmp(1:end-5,:);
% tmp = ncread(balfile,'dib2');
% nH2p = tmp(1:end-5,:);
% % dib2 = reshape(readf44f46(f44file,'dib2'),nx-2,ny-2);
% % dmb2 = reshape(readf44f46(f44file,'dmb2'),nx-2,ny-2);
% figure; hold on;
% leftix = ncread(balfile,'leftix')+1;
% xcut = find(diff(leftix(:,1))<1);
% cmap = jet(size(te,2)-2);
% ix = [xcut(3)+2:xcut(4),xcut(5)+1:size(te,1)-1]; %size(te,1)-16:size(te,1)-1;
% for iy = 2:size(te,2)-1 %2:5
%     plot(reshape(te(ix,iy),[],1),...
%                  reshape(nH2p(ix,iy),[],1)./reshape(nH2(ix,iy),[],1),...
%                  'marker','.','linestyle','none','color',cmap(iy-1,:),'displayname',['ring ',num2str(iy)]);
% end
% % scatter(reshape(te(2:end-1,2:end-1),[],1),reshape(nH2p(2:end-1,2:end-1),[],1)./reshape(nH2(2:end-1,2:end-1),[],1),[],reshape(vol(2:end-1,2:end-1),[],1));
% % tmp = ncread(balfile,'dmb2');
% % nH2 = tmp(1:end-5,:); % SN: tmp(1:end-2,:); CDN: tmp(1:end-5,:);
% tmp = ncread(balfile,'na');
% nHp = tmp(:,:,2);
% ne = ncread(balfile,'ne');
% nH2p_post = calc_nH2p_post(ne,nHp,nH2,te,te);
% plot(reshape(te(ix,2:end-1),[],1),reshape(nH2p_post(ix,2:end-1),[],1)./reshape(nH2(ix,2:end-1),[],1),'or');
% set(gca,'yscale','log','xscale','log');
% disp('yo');
% % tmp = ncread(balfile,'dib2overdmb2');
% % nH2povernH2 = tmp(1:end-5,:);
% % plotvar = nH2p./nH2p_post-1; % nH2povernH2-nH2p_post./nH2;  %nH2povernH2;%./nH2; % (nH2p-nH2p_post)./nH2;
% eirene_mc_pipl_sna_bal = ncread(balfile,'eirene_mc_pipl_sna_bal');
% eirene_mc_pmpl_sna_bal = ncread(balfile,'eirene_mc_pmpl_sna_bal');
% % plotvar = -(sum(eirene_mc_pipl_sna_bal(:,:,2,:),4)+sum(eirene_mc_pmpl_sna_bal(:,:,2,:),4))./vol;
% load('get_amjuel');
% sigmav_H4_2_2_14 = h4_rate(AMJ.H4_2_2_14.table,ne,te);
% sigmav_H4_2_2_11 = h4_rate(AMJ.H4_2_2_11.table,ne,te);
% MAR_post = ne.*nH2p_post.*sigmav_H4_2_2_14;
% MAI_post = ne.*nH2p_post.*sigmav_H4_2_2_11;
% MAR_solps = ne.*nH2p.*sigmav_H4_2_2_14;
% MAI_solps = ne.*nH2p.*sigmav_H4_2_2_14;
% plotvar = MAR_solps-MAI_solps
%% Dalpha emissivity:
% ne = ncread(balfile,'ne')/1.602e-19;
% nx = size(ne,1);
% ny = size(ne,2);
% plotvar = [zeros(nx,1),[zeros(1,ny-2);reshape(readf44f46(fort44_loc,'emiss'),nx-2,ny-2);zeros(1,ny-2)],zeros(nx,1)];
%% density residuals:
% resco = ncread(balfile,'resco');
% plotvar = resco(:,:,2);% sum(resco(:,:,4:end),3); %abs(log10(resco(:,:,2)));
%% momentum residuals:
% resmo = ncread(balfile,'resmo');
% plotvar = resmo(:,:,2);
%% heat residuals:
% plotvar = ncread(balfile,'reshe');
%% potential residuals:
% plotvar = ncread(balfile,'respo');
%% main ion density:
% na = ncread(balfile,'na');
% plotvar = na(:,:,5);
%% ne:
% plotvar = ncread(balfile,'ne');
%% Te:
plotvar = ncread(balfile,'te')/1.602e-19;
%% Ti:
% plotvar = ncread(balfile,'ti')/1.602e-19;
%% Plasma beta:
% ne = ncread(balfile,'ne');
% te = ncread(balfile,'te');
% bb = ncread(balfile,'bb');
% mu0 = 1.25663706212E-6;
% plotvar = (ne.*te)./(bb(:,:,4).^2/mu0);
%% Ion gyroradius:
% bb = ncread(balfile,'bb');
% ti = ncread(balfile,'ti');
% mi = 2*1.67E-27;
% plotvar = sqrt(2*ti/mi)./(1.602e-19*bb(:,:,4)/mi);
%% plasma static pressure:
% plotvar = ncread(balfile,'ne').*ncread(balfile,'te')+sum(ncread(balfile,'na'),3).*ncread(balfile,'ti');
%% radially averaged plasma static pressure:
% pstat = ncread(balfile,'ne').*ncread(balfile,'te')+sum(ncread(balfile,'na'),3).*ncread(balfile,'ti');
% gs = ncread(balfile,'gs');
% plotvar = repmat(sum(pstat.*gs(:,:,1),2)./sum(gs(:,:,1),2),1,size(pstat,2));
%% neutral static pressure:
% dab2 = ncread(balfile,'dab2');
% tab2 = ncread(balfile,'tab2');
% dib2 = ncread(balfile,'dib2');
% tib2 = ncread(balfile,'tib2');
% dmb2 = ncread(balfile,'dmb2');
% tmb2 = ncread(balfile,'tmb2');
% plotvar = dab2(1:end-5,:,1).*tab2(1:end-5,:,1)+dib2(1:end-5,:).*tib2(1:end-5,:)+dmb2(1:end-5,:).*tmb2(1:end-5,:);
%% radially averaged neutral static pressure:
% dab2 = ncread(balfile,'dab2');
% tab2 = ncread(balfile,'tab2');
% dib2 = ncread(balfile,'dib2');
% tib2 = ncread(balfile,'tib2');
% dmb2 = ncread(balfile,'dmb2');
% tmb2 = ncread(balfile,'tmb2');
% pstat = dab2(1:end-5,:,1).*tab2(1:end-5,:,1)+dib2(1:end-5,:).*tib2(1:end-5,:)+dmb2(1:end-5,:).*tmb2(1:end-5,:);
% gs = ncread(balfile,'gs');
% plotvar = repmat(sum(pstat.*gs(:,:,1),2)./sum(gs(:,:,1),2),1,size(pstat,2));
%% Electron pressure poloidal gradient
% comuse = get_comuse(balfile);
% ne = ncread(balfile,'ne');
% te = ncread(balfile,'te')/1.602e-19;
% pe = ne.*te;
% plotvar = zeros(size(pe));
% for iy=1:size(pe,2)
%     plotvar(:,iy) = gradient(pe(:,iy),comuse.dspol(:,iy));
% end
% plotvar = plotvar./ne;
%% Potential poloidal gradient
% comuse = get_comuse(balfile);
% po = ncread(balfile,'po');
% plotvar = zeros(size(po));
% for iy=1:size(po,2)
%     plotvar(:,iy) = gradient(po(:,iy),comuse.dspol(:,iy));
% end
%% Ion parallel velocity:
% ua = ncread(balfile,'ua');
% plotvar = ua(:,:,2);
%% Ion toroidal velocity:
% ua = ncread(balfile,'ua');
% bb = ncread(balfile,'bb');
% plotvar = ua(:,:,2).*bb(:,:,3)./bb(:,:,4);
%% Drift flux:
% te = ncread(balfile,'te')/1.602e-19;
% ti = ncread(balfile,'ti')/1.602e-19;
% te_left = zeros(size(te));
% te_bottom = zeros(size(te));
% ti_left = zeros(size(ti));
% ti_bottom = zeros(size(ti));
% leftix = ncread(balfile,'leftix')+2;
% leftiy = ncread(balfile,'leftiy')+2;
% bottomix = ncread(balfile,'bottomix')+2;
% bottomiy = ncread(balfile,'bottomiy')+2;
% for ix=1:size(te,1)
%     for iy=1:size(te,2)
%         if leftix(ix,iy)~=0
%             te_left(ix,iy) = 0.5*(te(ix,iy)+te(leftix(ix,iy),leftiy(ix,iy)));
%             ti_left(ix,iy) = 0.5*(ti(ix,iy)+ti(leftix(ix,iy),leftiy(ix,iy)));
%         end
%         if bottomiy(ix,iy)~=0
%             te_bottom(ix,iy) = 0.5*(te(ix,iy)+te(bottomix(ix,iy),bottomiy(ix,iy)));
%             ti_bottom(ix,iy) = 0.5*(ti(ix,iy)+ti(bottomix(ix,iy),bottomiy(ix,iy)));
%         end
%     end
% end
% fna_drift = ncread(balfile,'fna_drift');
% gs = ncread(balfile,'gs');
% plotvar = 5/2*1.602e-19*sum(fna_drift(:,:,1,[2,4:end]),4).*(te_left+ti_left)./gs(:,:,1); % Poloidal
% % plotvar = 5/2*1.602e-19*sum(fna_drift(:,:,2,[2,4:end]),4).*(te_bottom+ti_bottom)./gs(:,:,2); % Radial
% % leftix = ncread(balfile,'leftix')+1;
% % xcut = find(diff(leftix(:,1))<1);
% % sep = ncread(balfile,'jsep')+1;
% % plotvar([xcut(1):xcut(2)-1,xcut(4)+1:xcut(5)],1:sep+1)=nan;
%% Divergence of ExB flux:
% plotvar = zeros(size(ncread(balfile,'ne')));
% rightix = ncread(balfile,'rightix')+2;
% rightiy = ncread(balfile,'rightiy')+2;
% topix = ncread(balfile,'topix')+2;
% topiy = ncread(balfile,'topiy')+2;
% fna_drift = ncread(balfile,'fna_drift');
% dv = ncread(balfile,'vol');
% for ix=1:size(plotvar,1)
%     for iy=1:size(plotvar,2)
%         if rightix(ix,iy)<size(plotvar,1) && topiy(ix,iy)<size(plotvar,2)
%             plotvar(ix,iy) = (-fna_drift(ix,iy,1,2)+fna_drift(rightix(ix,iy),rightiy(ix,iy))-fna_drift(ix,iy,2,2)+fna_drift(topix(ix,iy),topiy(ix,iy)))/dv(ix,iy);
%         end
%     end
% end
%% Poloidal ExB drift flux density:
% fna_drift = ncread(balfile,'fna_drift');
% gs = ncread(balfile,'gs');
% plotvar = fna_drift(:,:,1,2)./gs(:,:,1);
%% Radial ExB drift flux density:
% fna_drift = ncread(balfile,'fna_drift');
% gs = ncread(balfile,'gs');
% plotvar = fna_drift(:,:,2,2)./gs(:,:,2);
%% potential:
% plotvar = ncread(balfile,'po');
% po = ncread('/home/david/remote/gateway/gss_efgw_work/work/g2dmoult/solps-iter_develop/runs/mastu_sxd_45456_445ms/ref_widercore_drift/b2time.nc','po2d');
% plotvar = po(:,:,end-2);
%% Sine of pitch angle:
% bb = ncread(balfile,'bb');
% plotvar = bb(:,:,1)./bb(:,:,4);
%% Toroidal magnetic field:
% bb = ncread(balfile,'bb');
% plotvar = bb(:,:,3);
%% Poloidal magetic field:
% bb = ncread(balfile,'bb');
% plotvar = bb(:,:,1);
%% Total field:
% bb = ncread(balfile,'bb');
% plotvar = bb(:,:,4);
%% Radial plasma energy flux density:
% fht = ncread(balfile,'fht');
% gs = ncread(balfile,'gs');
% plotvar = fht(:,:,2)./gs(:,:,2);
%% Poloidal current:
% comuse = get_comuse(balfile);
% fna = ncread(balfile,'fna_tot');
% fne = ncread(balfile,'fne');
% ion_current = zeros(size(fne));
% for is=1:comuse.ns
%     ion_current = ion_current+double(comuse.za(is))*fna(:,:,:,is);
% end
% jcur = 1.602E-19*(ion_current-fne);
% plotvar = jcur(:,:,1);
%% fch:
% fch = ncread(balfile,'fch');
% plotvar = fch(:,:,1);
%% fht derived:
% fht = calc_fht_nodrift(balfile);
% gs = ncread(balfile,'gs');
% plotvar = fht(:,:,2)./gs(:,:,2);
%% Anomalous / total radial current:
% fch = ncread(balfile,'fch_p')+ncread(balfile,'fchdia')+ncread(balfile,'fchin')+ncread(balfile,'fchvispar')+ncread(balfile,'fchvisper')+ncread(balfile,'fchvisq')+ncread(balfile,'fchinert')+ncread(balfile,'fchanml');
% fchanml = ncread(balfile,'fchanml');
% plotvar = abs(fchanml(:,:,2)./fch(:,:,2));
%% Radial current (different components):
% fch = ncread(balfile,'fch');
% plotvar = fch(:,:,2);
%% Electron poloidal flux:
% fne = ncread(balfile,'fne');
% plotvar = fne(:,:,1);
%% Parallel area:
% bb = ncread(balfile,'bb');
% dv = ncread(balfile,'vol');
% hx = ncread(balfile,'hx');
% plotvar = dv./hx.*abs(bb(:,:,1)./bb(:,:,4));
%% Total static plasma pressure:
% plotvar = (ncread(balfile,'te')+ncread(balfile,'ti')).*ncread(balfile,'ne');
%% radial particle flux:
% fna = ncread(balfile,'fna_tot');
% gs = ncread(balfile,'gs');
% plotvar = fna(:,:,2,2)./gs(:,:,2);
%% radial particle flux due to current
%% radial heat flux:
% fhe_32 = ncread(balfile,'fhe_32');
% fhe_52 = ncread(balfile,'fhe_52');
% fhe_thermj = ncread(balfile,'fhe_thermj');
% fhe_cond = ncread(balfile,'fhe_cond');
% fhe_dia = ncread(balfile,'fhe_dia');
% fhe_ecrb = ncread(balfile,'fhe_ecrb');
% fhe_strange = ncread(balfile,'fhe_strange');
% fhe_pschused = ncread(balfile,'fhe_pschused');
% fhi_32 = ncread(balfile,'fhi_32');
% fhi_52 = ncread(balfile,'fhi_52');
% fhi_cond = ncread(balfile,'fhi_cond');
% fhi_dia = ncread(balfile,'fhi_dia');
% fhi_ecrb = ncread(balfile,'fhi_ecrb');
% fhi_strange = ncread(balfile,'fhi_strange');
% fhi_pschused = ncread(balfile,'fhi_pschused');
% fhi_inert = ncread(balfile,'fhi_inert');
% fhi_vispar = ncread(balfile,'fhi_vispar');
% fhi_anml = ncread(balfile,'fhi_anml');
% fhi_kevis = ncread(balfile,'fhi_kevis');
% gs = ncread(balfile,'gs');
% plotvar = fhi_kevis(:,:,2)./gs(:,:,2);
%% impurity ion density:
% na = ncread(balfile,'na');
% plotvar = sum(na(:,:,4:end),3); %sum(na(:,:,4:end),3);
% % dab2 = ncread(balfile,'dab2');
% % plotvar = dab2(1:end-5,:,2);
%% radial neutral atom energy flux:
% refluxa = ncread(balfile,'refluxa');
% plotvar = refluxa(1:end-5,:,1);
%% radial neutral molecule energy flux:
% refluxm = ncread(balfile,'refluxm');
% plotvar = refluxm(1:end-5,:);
%% Ion static pressure:
% na = ncread(balfile,'na');
% ti = ncread(balfile,'ti');
% plotvar = na(:,:,2).*ti;
%% dtmax for drift cases:
% bb = ncread(balfile,'bb');
% R = mean(ncread(balfile,'crx'),3);
% ne = ncread(balfile,'ne');
% te = ncread(balfile,'te');
% ti = ncread(balfile,'ti');
% plotvar = 1e-5*(bb(:,:,4).*R).^2./(te+te)*1.602E-19;
%% Total friction term:
% za=comuse.za;       
% tmp = ncread(balfile,'b2sifr_smofrea_bal');
% b2sifr_smofrea = sum(tmp(:,:,za>0),3);
% tmp = ncread(balfile,'b2sifr_smofria_bal');
% b2sifr_smofria = sum(tmp(:,:,za>0),3);
% tmp = ncread(balfile,'b2sifr_smotfea_bal');
% b2sifr_smotfea = sum(tmp(:,:,za>0),3);
% tmp = ncread(balfile,'b2sifr_smotfia_bal');
% b2sifr_smotfia = sum(tmp(:,:,za>0),3);
% plotvar = b2sifr_smofrea+b2sifr_smofria+b2sifr_smotfea+b2sifr_smotfia;
%%
%% Ionisation source due to atom-plasma collisions:
% eirene_mc_papl_sna_bal = ncread(balfile,'eirene_mc_papl_sna_bal');
% plotvar = sum(eirene_mc_papl_sna_bal(:,:,2,:),4)./ncread(balfile,'vol');
% % plotvar = eirene_mc_papl_sna_bal(:,:,2,4)./ncread(balfile,'vol');
%% Dominant impurity charge state:
% na = ncread(balfile,'na');
% [~,plotvar]=max(na(:,:,3:end),[],3);
%% Recombination source:
% eirene_mc_pppl_sna_bal = ncread(balfile,'eirene_mc_pppl_sna_bal');
% plotvar = -sum(eirene_mc_pppl_sna_bal(:,:,2,:),4)./ncread(balfile,'vol');
%% Test ion source:
% eirene_mc_pipl_sna_bal = ncread(balfile,'eirene_mc_pipl_sna_bal');
% plotvar = sum(eirene_mc_pipl_sna_bal(:,:,2,:),4); %squeeze(eirene_mc_pipl_sna_bal(:,:,2,7)); %
%% MAR+MAI:
% eirene_mc_pipl_sna_bal = ncread(balfile,'eirene_mc_pipl_sna_bal');
% eirene_mc_pmpl_sna_bal = ncread(balfile,'eirene_mc_pmpl_sna_bal');
% plotvar = -sum(eirene_mc_pipl_sna_bal(:,:,2,:)+eirene_mc_pmpl_sna_bal(:,:,2,:),4)./ncread(balfile,'vol');
%% Impurity ion fraction (%):
% iatm = 2;
% iion = 7:16; %4:21; %7:16;
% na = ncread(balfile,'na');
% dab2 = ncread(balfile,'dab2');
% ne = ncread(balfile,'ne');
% % plotvar = (100*(sum(na(:,:,iion),3)+zeros(size(dab2(1:end-5,:,iatm))))./ne);
% plotvar = (100*(sum(na(:,:,iion),3)+dab2(1:end-2,:,iatm))./ne); %(100*(sum(na(:,:,iion),3)+dab2(1:end-5,:,iatm))./ne);
% % % comuse = get_comuse(balfile);
% % % xcut = find(diff(comuse.leftix(:,1))<1);
% % % fprintf('%.5f\n',100*sum((sum(na(xcut(1):xcut(2),comuse.sep+2,iion),3)+dab2(xcut(1):xcut(2),comuse.sep+2,iatm)).*comuse.gs(xcut(1):xcut(2),comuse.sep+2,2))/...
% % %                      sum(ne(xcut(1):xcut(2),comuse.sep+2).*comuse.gs(xcut(1):xcut(2),comuse.sep+2,2)));
%% Mach number:
% ua = ncread(balfile,'ua');
% te = ncread(balfile,'te');
% ti = ncread(balfile,'ti');
% plotvar = ua(:,:,2)./sqrt((te+ti)/1.67e-27/2);
%% Zeff:
% na = ncread(balfile,'na');
% za = ncread(balfile,'za');
% num = zeros(size(na(:,:,1)));
% den = zeros(size(na(:,:,1)));
% for iz=find(za>0)'
%     num = num+na(:,:,iz)*double(za(iz))^2;
%     den = den+na(:,:,iz)*double(za(iz));
% end
% plotvar = num./den;
%% D atom density:
% dab2 = ncread(balfile,'dab2');
% plotvar = dab2(1:end-5,:,1);
%% D atom density from f44 file:
% nx = size(te,1);
% ny = size(te,2);
% tmp = ncread(balfile,'dmb2');
% nH2 = tmp(1:end-5,:); % SN: tmp(1:end-2,:); CDN: tmp(1:end-5,:);
% tmp = ncread(balfile,'dib2');
% nH2p = tmp(1:end-5,:);
% % dib2 = reshape(readf44f46(f44file,'dib2'),nx-2,ny-2);
%% ion / neutral density:
% dab2 = ncread(balfile,'dab2');
% dmb2 = ncread(balfile,'dmb2');
% na = ncread(balfile,'na');
% plotvar = (dab2(1:end-5,:,1)+dmb2(1:end-5,:,1))./na(:,:,2);
%% D2 molecular density:
% dmb2 = ncread(balfile,'dmb2');
% plotvar = dmb2(1:end-5,:,1); %dmb2(1:end-2,:,1); %
%% D2^+ molecular density:
% dib2 = ncread(balfile,'dib2');
% plotvar = dib2(1:end-5,:,1);
%% Neutral-neutral mean free path:
% dab2 = ncread(balfile,'dab2');
% dmb2 = ncread(balfile,'dmb2');
% tab2 = ncread(balfile,'tab2');
% tmb2 = ncread(balfile,'tmb2');
% plot(dspol{is}(xcut{is}(3)+2:end-1,iring(is)),tab2(xcut{is}(3)+2:end-6,iring(is),1)/1.602e-19);
% plot(dspol{is}(xcut{is}(3)+2:end-1,iring(is)),tmb2(xcut{is}(3)+2:end-6,iring(is))/1.602e-19);
% MFP = 20.1*(tab2(:,:,1)/1.602E-19).^0.25./(dab2(:,:,1)/1e26);
% plotvar = 1./MFP(1:end-5,:);
%% Neutral pressure:
% dab2 = ncread(balfile,'dab2');
% dmb2 = ncread(balfile,'dmb2');
% tab2 = ncread(balfile,'tab2');
% tmb2 = ncread(balfile,'tmb2');
% plotvar = dab2(:,:,1).*tab2(:,:,1)+dmb2(:,:,1).*tmb2(:,:,1);
% plotvar = plotvar(1:end-5,:);
%% Molecule temp:
% tmb2 = ncread(balfile,'tmb2')/1.602E-19*1.160451812E4; % (K)
% plotvar = tmb2(1:end-5,:,1);
%% Atomic temp:
% tab2 = ncread(balfile,'tab2')/1.602E-19;
% plotvar = tab2(1:end-5,:,1); %tab2(1:end-2,:,1);
%% Timescale for molecules to reach the side of the grid
% tmb2 = ncread(balfile,'tmb2');
% vtmol = sqrt(tmb2(1:end-5,:,1)/(4*1.67e-27));
% % Radial distance to side:
% distfrombot = cumsum(hy,2);
% distfromtop = repmat(distfrombot(:,end),1,size(hy,2))-distfrombot;
% plotvar = min(distfrombot,distfromtop)./vtmol;
%% Molecule - plasma momentum sink:
% eirene_mc_mmpl_smo_bal = ncread(balfile,'eirene_mc_mmpl_smo_bal');
% b2mndr_hz = ncread(balfile,'b2mndr_hz');
% dv = ncread(balfile,'vol');
% gs = ncread(balfile,'gs');
% hz = (1-b2mndr_hz)+b2mndr_hz*(dv./gs(:,:,3));
% plotvar = sum(eirene_mc_mmpl_smo_bal(:,:,2,:),4)./dv./hz;
%% Conductive electron heat flux:
% fhe_cond = ncread(balfile,'fhe_cond');
% plotvar = fhe_cond(:,:,1);
%% Poloidal convective ion flux:
% fna_pll = ncread(balfile,'fna_pll');
% plotvar = fna_pll(:,:,1,2);
%% Poloidal diffusive ion flux:
% fna_nanom = ncread(balfile,'fna_nanom');
% fna_panom = ncread(balfile,'fna_panom');
% plotvar = fna_nanom(:,:,1,2)+fna_panom(:,:,1,2);
%% Radial atom flux:
% tmp = ncread(balfile,'rfluxa');
% plotvar = tmp(1:end-5,:,1);
%% Radial molecule flux:
% plotvar = ncread(balfile,'rfluxm');
%% Radial heat flux:
% plotvar = ncread(balfile,'fhe_32')+ncread(balfile,'fhe_52')+ncread(balfile,'fhe_thermj')+ncread(balfile,'fhe_cond')+ncread(balfile,'fhe_dia')+ncread(balfile,'fhe_ecrb')+ncread(balfile,'fhe_strange')+ncread(balfile,'fhe_pschused')+...
%           ncread(balfile,'fhi_32')+ncread(balfile,'fhi_52')+ncread(balfile,'fhi_cond')+ncread(balfile,'fhi_dia')+ncread(balfile,'fhi_ecrb')+ncread(balfile,'fhi_strange')+ncread(balfile,'fhi_pschused')+ncread(balfile,'fhi_inert')+ncread(balfile,'fhi_vispar')+ncread(balfile,'fhi_anml')+ncread(balfile,'fhi_kevis');
% plotvar = plotvar(:,:,2);
%% Energy cost of ionisation:
% % % eirene_mc_papl_sna_bal = ncread(balfile,'eirene_mc_papl_sna_bal');
% eirene_mc_eael_she_bal = ncread(balfile,'eirene_mc_eael_she_bal');
% eirene_mc_papl_sna_bal = ncread(balfile,'eirene_mc_papl_sna_bal');
% % % plotvar = sum(eirene_mc_eael_she_bal,3)./sum(eirene_mc_papl_sna_bal(:,:,2,:),4)/1.602E-19;
% plotvar = -sum(eirene_mc_eael_she_bal,3)./ncread(balfile,'vol') -13.6*1.6E-19*sum(eirene_mc_papl_sna_bal(:,:,2,:),4)./ncread(balfile,'vol');
% % % % plotvar = -eirene_mc_eael_she_bal(:,:,11)/1.602E-19;
%% Electron energy cost of ion radiation:
% b2stel_she_bal = ncread(balfile,'b2stel_she_bal');
% plotvar = -sum(b2stel_she_bal(:,:,1),3)./ncread(balfile,'vol'); %-sum(b2stel_she_bal(:,:,7:end),3)./ncread(balfile,'vol'); % Last dimension here is the species index
% % te = ncread(balfile,'te');
% % inds0=plotvar==0 & te>1 & te<80;
% % plotvar(inds0)=10;
% % plotvar(~inds0)=-10;
%% Electron energy cost of fixed fraction impurity cooling:
% b2stel_she_bal = ncread(balfile,'b2stel_she_bal');
% plotvar = log10(-b2stel_she_bal(:,:,2)./ncread(balfile,'vol'));
%% Total radiation (fluid neutrals):
% b2stel_she_bal = ncread(balfile,'b2stel_she_bal');
% b2stel_sna_ion_bal = ncread(balfile,'b2stel_sna_ion_bal');
% rad_dens = (-sum(b2stel_she_bal,3)+13.6*1.602E-19*b2stel_sna_ion_bal(:,:,1))./ncread(balfile,'vol');
%% neon radiation (iter case):
% b2stel_she_bal = ncread(balfile,'b2stel_she_bal');
% plotvar = -sum(b2stel_she_bal(:,:,7:end),3)./ncread(balfile,'vol');
%% Eirene particle source:
% eirene_source = ncread(balfile,'eirene_mc_papl_sna_bal')+...
%                 ncread(balfile,'eirene_mc_pmpl_sna_bal')+...
%                 ncread(balfile,'eirene_mc_pipl_sna_bal')+...
%                 ncread(balfile,'eirene_mc_pppl_sna_bal');
% plotvar = sum(squeeze(eirene_source(:,:,2,:)),3)./ncread(balfile,'vol');
% plotvar = -eirene_source(:,:,2,34)./ncread(balfile,'vol');
% % plotvar(plotvar<=0) = 1e8;
% % plotvar = log10(plotvar);
%% Eirene energy source:
% eirene_source = ncread(balfile,'eirene_mc_eael_she_bal')+...
%                 ncread(balfile,'eirene_mc_emel_she_bal')+...
%                 ncread(balfile,'eirene_mc_eiel_she_bal')+...
%                 ncread(balfile,'eirene_mc_epel_she_bal')+...
%                 ncread(balfile,'eirene_mc_eapl_shi_bal')+...
%                 ncread(balfile,'eirene_mc_empl_shi_bal')+...
%                 ncread(balfile,'eirene_mc_eipl_shi_bal')+...
%                 ncread(balfile,'eirene_mc_eppl_shi_bal');
% plotvar = -sum(eirene_source,3)./ncread(balfile,'vol');
% % % plotvar = -eirene_source(:,:,2,34)./ncread(balfile,'vol');
% % % plotvar(plotvar<=0) = 1e8;
% % % plotvar = log10(plotvar);
%% Ion heat source due to molecule - plasma collisions:
% plotvar = sum(ncread(balfile,'eirene_mc_eipl_shi_bal'),3)./ncread(balfile,'vol');
%% b2siav_smovh:
% tmp = ncread(balfile,'b2siav_smovh_bal');
% plotvar = tmp(:,:,2);
%% b2siav_smovv:
% tmp = ncread(balfile,'b2siav_smovv_bal');
% plotvar = tmp(:,:,2);
%% anomalous / total current:
% fchanml = ncread(balfile,'fchanml');
% fch = ncread(balfile,'fch_p')+ncread(balfile,'fchdia')+ncread(balfile,'fchin')+ncread(balfile,'fchvispar')+ncread(balfile,'fchvisper')+ncread(balfile,'fchvisq')+ncread(balfile,'fchinert')+ncread(balfile,'fchanml');
% plotvar = log10(abs(fchanml(:,:,2)./fch(:,:,2)));
%% eirene atom-plasma ion heat source:
% eirene_mc_eapl_shi_bal = ncread(balfile,'eirene_mc_eapl_shi_bal');
% plotvar = sum(eirene_mc_eapl_shi_bal,3)./ncread(balfile,'vol')/1e6;
%% % Neutral fluxes and sinks:
% f44 = fopen(fort44file,'r');
% tmp = textscan(f44,'%d %d %*[^\n]',1);
% nx = tmp{1}+2;
% ny = tmp{2}+2;
% tmp = textscan(f44,'%d %d %d',1);
% natm = tmp{1};
% nmol = tmp{2};
% nion = tmp{3};
% atom_names = textscan(f44,'%s',natm);
% molecule_names = textscan(f44,'%s',nmol);
% ion_names = textscan(f44,'%s',nion);
% tline = fgetl(f44);
% while (isempty(strfind(tline,' refluxa ')))
%     tline = fgetl(f44);
% end
% tmp = textscan(f44,'%f',(nx-2)*(ny-2)*natm); 
% refluxa = cat(2,zeros(nx,1,natm),(cat(1,zeros(2,ny-2,natm),reshape(tmp{1},(nx-2),(ny-2),natm))),zeros(nx,1,natm)); % Reshape and pad with zeros for guard cells
% plotvar = refluxa(:,:,1);
% while (isempty(strfind(tline,' refluxm ')))
%     tline = fgetl(f44);
% end
% tmp = textscan(f44,'%f',(nx-2)*(ny-2)*nmol);    
% refluxm = cat(2,zeros(nx,1,nmol),(cat(1,zeros(2,ny-2,nmol),reshape(tmp{1},(nx-2),(ny-2),nmol))),zeros(nx,1,nmol)); % Reshape and pad with zeros for guard cells
% while (isempty(strfind(tline,' pefluxa ')))
%     tline = fgetl(f44);
% end
% tmp = textscan(f44,'%f',(nx-2)*(ny-2)*natm); 
% pefluxa = cat(2,zeros(nx,1,natm),(cat(1,zeros(2,ny-2,natm),reshape(tmp{1},(nx-2),(ny-2),natm))),zeros(nx,1,natm)); % Reshape and pad with zeros for guard cells
% while (isempty(strfind(tline,' pefluxm ')))
%     tline = fgetl(f44);
% end
% tmp = textscan(f44,'%f',(nx-2)*(ny-2)*nmol);    
% pefluxm = cat(2,zeros(nx,1,nmol),(cat(1,zeros(2,ny-2,nmol),reshape(tmp{1},(nx-2),(ny-2),nmol))),zeros(nx,1,nmol)); % Reshape and pad with zeros for guard cells
% while (isempty(strfind(tline,' eneutrad ')))
%     tline = fgetl(f44);
% end
% tmp = textscan(f44,'%f',(nx-2)*(ny-2)*natm);    
% eneutrad = cat(2,zeros(nx,1,natm),(cat(1,zeros(1,ny-2,natm),reshape(tmp{1},(nx-2),(ny-2),natm),zeros(1,ny-2,natm))),zeros(nx,1,natm)); % Reshape and pad with zeros for guard cells
% while (isempty(strfind(tline,' emolrad ')))
%     tline = fgetl(f44);
% end
% tmp = textscan(f44,'%f',(nx-2)*(ny-2)*nmol);    
% emolrad = cat(2,zeros(nx,1,nmol),(cat(1,zeros(1,ny-2,nmol),reshape(tmp{1},(nx-2),(ny-2),nmol),zeros(1,ny-2,nmol))),zeros(nx,1,nmol)); % Reshape and pad with zeros for guard cells
% while (isempty(strfind(tline,' eionrad ')))
%     tline = fgetl(f44);
% end
% tmp = textscan(f44,'%f',(nx-2)*(ny-2)*nion);    
% eionrad = cat(2,zeros(nx,1,nion),(cat(1,zeros(1,ny-2,nion),reshape(tmp{1},(nx-2),(ny-2),nion),zeros(1,ny-2,nion))),zeros(nx,1,nion)); % Reshape and pad with zeros for guard cells
% fclose(f44);
%% Balmer emission:
% ne = ncread(balfile,'ne');
% te = ncread(balfile,'te');
% na = ncread(balfile,'na');
% tmp = ncread(balfile,'dab2'); dab2 = tmp(1:end-5,:,1);
% tmp = ncread(balfile,'dmb2'); dmb2 = tmp(1:end-5,:);
% tmp = ncread(balfile,'dib2'); dib2 = tmp(1:end-5,:);
% [BalmerHpRec,BalmerHExc] = SOLPS_Balmer_Model_ADAS(ne,na(:,:,1),dab2,te,1:10);
% [BalmerH2Dis,BalmerH2pDis] = SOLPS_Balmer_Model_Yacora(ne,dmb2,dib2,te,1:6);
% % Dalpha:
% plotvar = BalmerHpRec(:,:,1)+BalmerHExc(:,:,1)+BalmerH2pDis(:,:,1)+BalmerH2Dis(:,:,1);
%% External electron heat source:
% plotvar = ncread(balfile,'ext_she_bal')./ncread(balfile,'vol')*1e-6;
%% External ion heat source:
% plotvar = ncread(balfile,'ext_shi_bal')./ncread(balfile,'vol')*1e-6;
%% External ion particle source:
% ext_sna_bal = ncread(balfile,'ext_sna_bal');
% plotvar = ext_sna_bal(:,:,2)./ncread(balfile,'vol');
%% External heat source divided by ion particle source:
% ext_she_bal = ncread(balfile,'ext_she_bal');
% ext_shi_bal = ncread(balfile,'ext_shi_bal');
% ext_sna_bal = ncread(balfile,'ext_sna_bal');
% plotvar = (ext_she_bal+ext_shi_bal)./ext_sna_bal(:,:,2)/3/1.602e-19;
%% Fulcher emission:
% ne = ncread(balfile,'ne');
% te = ncread(balfile,'te')/1.602E-19;
% tmp = ncread(balfile,'dmb2');
% dmb2 = tmp(1:end-5,:);
% plotvar = SOLPS_Fulcher_Model(ne,dmb2,te);

figure('windowstyle','docked');
axes;
hold on;
axis image;
xlabel('R (cm)');
ylabel('Z (cm)');

r = ncread(balfile,'crx');
z = ncread(balfile,'cry');
sep = ncread(balfile,'jsep')+1; %16;

rbl = r(:,:,1);
rbr = r(:,:,2);
rtl = r(:,:,3);
rtr = r(:,:,4);
zbl = z(:,:,1);
zbr = z(:,:,2);
ztl = z(:,:,3);
ztr = z(:,:,4);

% Plot the grid (including ghost cells):
patch([reshape(rbl,1,[]);...
       reshape(rbr,1,[]);...
       reshape(rtr,1,[]);...
       reshape(rtl,1,[])],...
      [reshape(zbl,1,[]);...
       reshape(zbr,1,[]);...
       reshape(ztr,1,[]);...
       reshape(ztl,1,[])],(reshape(plotvar,1,[])),'linestyle','none');
[~,i]=max(plotvar(:));
rc=mean(r,3);
zc=mean(z,3);
plot(rc(i),zc(i),'*m');
c=contour(rc,zc,plotvar,[5,5],'-m');

% r_efit = ncread('~/epm044677.nc','/epm/output/profiles2D/r');
% z_efit = ncread('~/epm044677.nc','/epm/output/profiles2D/z');
% Bpol_efit = ncread('~/epm044677.nc','/epm/output/profiles2D/Bpol');
% Bphi_efit = ncread('~/epm044677.nc','/epm/output/profiles2D/Bphi');
% [x,y]=meshgrid(r_efit,z_efit);
% vq = interp2(x,y,-Bphi_efit(:,:,78),rc,zc);
% patch(100*[reshape(rbl,1,[]);...
%        reshape(rbr,1,[]);...
%        reshape(rtr,1,[]);...
%        reshape(rtl,1,[])],...
%       100*[reshape(zbl,1,[]);...
%        reshape(zbr,1,[]);...
%        reshape(ztr,1,[]);...
%        reshape(ztl,1,[])],(reshape(vq,1,[])),'linestyle','none');

   
% cr = mean(r,3);
% cz = mean(z,3);
% plot(cr(48,:),cz(48,:),'.r');

leftix = ncread(balfile,'leftix')+1;
xcut = find(diff(leftix(:,1))<1);
% plot(100*rc([xcut(1),xcut(2)],sep+1),100*zc([xcut(1),xcut(2)],sep+1),'*g')
if length(xcut)==5
    plot(rbl(2:xcut(3),sep+2),zbl(2:xcut(3),sep+2),'-m');
    plot(rbl(xcut(3)+2:end,sep+2),zbl(xcut(3)+2:end,sep+2),'-m');
    
%     plot(100*rbl(xcut(1):xcut(2)-1,sep-5),100*zbl(xcut(1):xcut(2)-1,sep-5),'-g');
%     plot(100*rbl(xcut(4)+1:xcut(5),sep-5),100*zbl(xcut(4)+1:xcut(5),sep-5),'-g');
%     plot(100*rc(xcut(1):xcut(2)-1,sep+1),100*zc(xcut(1):xcut(2)-1,sep+1),'*m');
%     plot(100*rc(xcut(4)+1:xcut(5),sep+1),100*zc(xcut(4)+1:xcut(5),sep+1),'*m');
else
    plot(rbl(2:end,sep+2),zbl(2:end,sep+2),'-m');
end

% Flux surfaces:
% for sep = ncread(balfile,'jsep')+1:size(rbl,2)-2
%     plot(rbl(2:xcut(3),sep+2),zbl(2:xcut(3),sep+2),'-k');
%     plot(rbl(xcut(3)+2:end,sep+2),zbl(xcut(3)+2:end,sep+2),'-k');
% end
% for iy = 1:ncread(balfile,'jsep')+2
%     plot([rbl(xcut(1):xcut(2)-1,iy);rbr(xcut(2)-1,iy)],[zbl(xcut(1):xcut(2)-1,iy);zbr(xcut(2)-1,iy)],'-g');
%     plot([rbl(xcut(4)+1:xcut(5),iy);rbr(xcut(5),iy)],[zbl(xcut(4)+1:xcut(5),iy);zbr(xcut(5),iy)],'-g');
% end
% plot([rbl(2:xcut(1)-1,2);rbr(xcut(1)-1,2)],[zbl(2:xcut(1)-1,2);zbr(xcut(1)-1,2)]);
% plot(rbl(xcut(5)+1:end,2),zbl(xcut(5)+1:end,2));
% plot(rbl(xcut(2):xcut(3),2),zbl(xcut(2):xcut(3),2));
% plot([rbl(xcut(3)+2:xcut(4),2);rbr(xcut(4),2)],[zbl(xcut(3)+2:xcut(4),2);zbr(xcut(4),2)]);
% plot(rbl(1,2:end),zbl(1,2:end));
% plot(rbl(xcut(3),2:end),zbl(xcut(3),2:end));
% plot(rbl(xcut(3)+2,2:end),zbl(xcut(3)+2,2:end));
% plot(rbl(end,2:end),zbl(end,2:end));
% plot(rbl(xcut(4),sep+2:end),zbl(xcut(4),sep+2:end));

% omp = ncread(balfile,'jxa')+1;
% plot(100*rc(omp,:),100*zc(omp,:),'.g');
% imp = ncread(balfile,'jxi')+1;
% plot(100*rc(imp,:),100*zc(imp,:),'.g');

% plot(100*rc(xcut(1):xcut(2)-1,1),100*zc(xcut(1):xcut(2)-1,1),'.g');
% plot(100*rc(xcut(4)+1:xcut(5),1),100*zc(xcut(4)+1:xcut(5),1),'.g');

% rc=mean(r,3);
% zc=mean(z,3);
% % plot(rc(14,:),zc(14,:),'.r');
% comuse = get_comuse(balfile);
% xcut = find(diff(comuse.leftix(:,1))<1);
% plot(rc(xcut(3)+2,2:end-1),zc(xcut(3)+2,2:end-1),'.r');
% plot(rc(xcut(1):xcut(2)-1,comuse.sep+1),zc(xcut(1):xcut(2)-1,comuse.sep+1),'.r');
% plot(rc(xcut(4)+1:xcut(5),comuse.sep+1),zc(xcut(4)+1:xcut(5),comuse.sep+1),'.r');
% disp(sum(plotvar(xcut(1):xcut(2)-1,comuse.sep+2)));
% disp(sum(plotvar(xcut(4)+1:xcut(5),comuse.sep+2)));
% plot(rbl(xcut(1):xcut(1)+21,sep+2),zbl(xcut(1):xcut(1)+21,sep+2),'-g');
% plot(rbl(xcut(1)+21:xcut(2)+1,sep+2),zbl(xcut(1)+21:xcut(2)+1,sep+2),'-r');
% plot(comuse.cr(xcut(1):xcut(1)+20,sep+2),comuse.cz(xcut(1):xcut(1)+20,sep+2),'.g');
% plot(comuse.cr(xcut(1)+21:xcut(2),sep+2),comuse.cz(xcut(1)+21:xcut(2),sep+2),'.r');
% plot(comuse.cr(33,:),comuse.cz(33,:),'-m');
% plot(comuse.cr(49,:),comuse.cz(49,:),'-m');