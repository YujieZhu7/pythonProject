import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPush'
x=np.arange(0, 4, 0.01)

# # Plot the scores
# NoBC_score_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S1_score.npy")
# NoBC_score_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S2_score.npy")
# NoBC_score_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S3_score.npy")
# NoBC_score_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S4_score.npy")
# NoBC_score_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_score2.npy")
# NoBC_score_mean = (NoBC_score_s1 + NoBC_score_s2 + NoBC_score_s3 + NoBC_score_s4 + NoBC_score_s5)/5
#
# BC_only_score_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S1_score.npy")
# BC_only_score_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S2_score.npy")
# BC_only_score_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S3_score.npy")
# BC_only_score_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S4_score.npy")
# BC_only_score_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S5_score.npy")
# BC_only_score_mean = (BC_only_score_s1+BC_only_score_s2+BC_only_score_s3+BC_only_score_s4+BC_only_score_s5)/5
#
# Qfilter_score_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S1_score.npy")
# Qfilter_score_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S2_score.npy")
# Qfilter_score_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S3_score.npy")
# Qfilter_score_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S4_score.npy")
# Qfilter_score_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S5_score.npy")
# Qfilter_score_mean = (Qfilter_score_s1+Qfilter_score_s2+Qfilter_score_s3+Qfilter_score_s4+Qfilter_score_s5)/5
#
# score_Ens10_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S1_score.npy")
# score_Ens10_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S2_score.npy")
# score_Ens10_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S3_score.npy")
# score_Ens10_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S4_score.npy")
# score_Ens10_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S5_score.npy")
# score_Ens10_mean = (score_Ens10_s1 + score_Ens10_s2 + score_Ens10_s3+score_Ens10_s4+score_Ens10_s5)/5
#
# Ens10new_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/MeanNone/EnsSize_10_S5_score.npy")
#
# plt.plot(x, NoBC_score_s1[1::20], color='black', alpha=0.2)
# plt.plot(x, NoBC_score_s2[1::20], color='black', alpha=0.2)
# plt.plot(x, NoBC_score_s3[1::20], color='black', alpha=0.2)  # much slower than the others
# plt.plot(x, NoBC_score_s4[1::20], color='black', alpha=0.2)
# plt.plot(x, NoBC_score_s5[1::20], color='black', alpha=0.2)
#
# plt.plot(x, BC_only_score_s1[1::20], color='green', alpha=0.2)
# plt.plot(x, BC_only_score_s2[1::20], color='green', alpha=0.2)
# plt.plot(x, BC_only_score_s3[1::20], color='green', alpha=0.2)
# plt.plot(x, BC_only_score_s4[1::20], color='green', alpha=0.2)
# plt.plot(x, BC_only_score_s5[1::20], color='green', alpha=0.2)
#
# plt.plot(x, Qfilter_score_s1[1::20], color='purple', alpha=0.2)
# plt.plot(x, Qfilter_score_s2[1::20], color='purple', alpha=0.2)
# plt.plot(x, Qfilter_score_s3[1::20], color='purple', alpha=0.2)
# plt.plot(x, Qfilter_score_s4[1::20], color='purple', alpha=0.2)
# plt.plot(x, Qfilter_score_s5[1::20], color='purple', alpha=0.2)
#
# plt.plot(x, score_Ens10_s1[1::20], color='blue', alpha=0.2)
# plt.plot(x, score_Ens10_s2[1::20], color='blue', alpha=0.2)
# plt.plot(x, score_Ens10_s3[1::20], color='blue', alpha=0.2)
# plt.plot(x, score_Ens10_s4[1::20], color='blue', alpha=0.2)
# plt.plot(x, score_Ens10_s5[1::20], color='blue', alpha=0.2)
#
# plt.plot(x, NoBC_score_mean[1::20], color='black', label='NoBC')
# plt.plot(x, BC_only_score_mean[1::20], color='green', label='BC_only')
# plt.plot(x, Qfilter_score_mean[1::20], color='purple', label='Qfilter')
# plt.plot(x, score_Ens10_mean[1::20], color='blue', label='Qfilter_EnsSize10')
# plt.plot(x, Ens10new_s5[1::20], color='red', label='EnsSize10NEW')
#
# # plt.plot(x, score_Ens2[1::20], color='orange', label='Qfilter_EnsSize2')
#
# plt.title('Scores')
# plt.xlabel('Environment interactions (4e6)')
# plt.ylabel('Score')
# plt.legend()
# # plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores_BC_multiseeds.png')
# plt.show()

demoAccept_first_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S1_demoaccept.npy")
demoAccept_first_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S2_demoaccept.npy")
demoAccept_first_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S3_demoaccept.npy")
demoAccept_first_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S4_demoaccept.npy")
demoAccept_first_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S5_demoaccept.npy")
demoAccept_first_mean = (demoAccept_first_s1 + demoAccept_first_s2+demoAccept_first_s3+demoAccept_first_s4+demoAccept_first_s5)/5

demoAccept_mean_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S1_demoaccept.npy")
demoAccept_mean_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S2_demoaccept.npy")
demoAccept_mean_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S3_demoaccept.npy")
demoAccept_mean_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S4_demoaccept.npy")
demoAccept_mean_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S5_demoaccept.npy")
demoAccept_mean_mean = (demoAccept_mean_s1 +demoAccept_mean_s2+demoAccept_mean_s3+demoAccept_mean_s4+demoAccept_mean_s5)/5

demoAccept_min_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S1_demoaccept.npy")
demoAccept_min_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S2_demoaccept.npy")
demoAccept_min_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S3_demoaccept.npy")
demoAccept_min_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S4_demoaccept.npy")
demoAccept_min_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S5_demoaccept.npy")
demoAccept_min_mean = (demoAccept_min_s1+demoAccept_min_s2+demoAccept_min_s3+demoAccept_min_s4+demoAccept_min_s5)/5

demoAccept_lcb_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S1_demoaccept.npy")
demoAccept_lcb_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S2_demoaccept.npy")
demoAccept_lcb_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S3_demoaccept.npy")
demoAccept_lcb_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S4_demoaccept.npy")
demoAccept_lcb_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S4_demoaccept.npy")
demoAccept_lcb_mean = (demoAccept_lcb_s1+demoAccept_lcb_s2+demoAccept_lcb_s3+demoAccept_lcb_s4+demoAccept_lcb_s5)/5

demoAccept_lcb = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB0.5/EnsSize_10_S5_demoaccept.npy")

Ens10new_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/MeanNone/EnsSize_10_S5_demoaccept.npy")

plt.plot(x, demoAccept_first_s1[1::20], color='blue', alpha=0.2)
plt.plot(x, demoAccept_first_s2[1::20], color='blue', alpha=0.2)
plt.plot(x, demoAccept_first_s3[1::20], color='blue', alpha=0.2)
plt.plot(x, demoAccept_first_s4[1::20], color='blue', alpha=0.2)
plt.plot(x, demoAccept_first_s5[1::20], color='blue', alpha=0.2)

plt.plot(x, demoAccept_mean_s1[1::20], color='red', alpha=0.2)
plt.plot(x, demoAccept_mean_s2[1::20], color='red', alpha=0.2)
plt.plot(x, demoAccept_mean_s3[1::20], color='red', alpha=0.2)
plt.plot(x, demoAccept_mean_s4[1::20], color='red', alpha=0.2)
plt.plot(x, demoAccept_mean_s5[1::20], color='red', alpha=0.2)

plt.plot(x, demoAccept_min_s1[1::20], color='orange', alpha=0.2)
plt.plot(x, demoAccept_min_s2[1::20], color='orange', alpha=0.2)
plt.plot(x, demoAccept_min_s3[1::20], color='orange', alpha=0.2)
plt.plot(x, demoAccept_min_s4[1::20], color='orange', alpha=0.2)
plt.plot(x, demoAccept_min_s5[1::20], color='orange', alpha=0.2)

plt.plot(x, demoAccept_lcb_s1[1::20], color='green', alpha=0.2)
plt.plot(x, demoAccept_lcb_s2[1::20], color='green', alpha=0.2)
plt.plot(x, demoAccept_lcb_s3[1::20], color='green', alpha=0.2)
plt.plot(x, demoAccept_lcb_s4[1::20], color='green', alpha=0.2)
plt.plot(x, demoAccept_lcb_s5[1::20], color='green', alpha=0.2)

plt.plot(x, demoAccept_first_mean[1::20], color='blue', label='Qfilter')
plt.plot(x, demoAccept_mean_mean[1::20], color='red', label='Mean')
plt.plot(x, demoAccept_min_mean[1::20], color='orange', label='Minimum')
plt.plot(x, demoAccept_lcb_mean[1::20], color='green', label='LCB')
plt.plot(x, demoAccept_lcb[1::20], color='pink', label='LCB0.5')
# plt.plot(x, Ens10new_s5[1::20], color='pink', label='NEW')

# plt.plot(x, Qfilter_demo[1::20], color='yellow', label='Qfilter_noensemble')
# plt.plot(x, demoAccept5[1::20], color='purple', label='ModifiedLCB')
plt.title('Acceptance rate of demonstrations')
plt.xlabel('Environment interactions (4e6)')
plt.ylabel('Acceptance rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/demoaccept_lcb0.5.png')
plt.show()
plt.close()


score_first_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S1_score.npy")
score_first_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S2_score.npy")
score_first_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S3_score.npy")
score_first_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S4_score.npy")
score_first_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S5_score.npy")
score_first_mean = (score_first_s1+score_first_s2+score_first_s3+score_first_s4+score_first_s5)/5

score_mean_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S1_score.npy")
score_mean_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S2_score.npy")
score_mean_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S3_score.npy")
score_mean_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S4_score.npy")
score_mean_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S5_score.npy")
score_mean_mean = (score_mean_s1+score_mean_s2+score_mean_s3+score_mean_s4+score_mean_s5)/5

score_min_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S1_score.npy")
score_min_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S2_score.npy")
score_min_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S3_score.npy")
score_min_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S4_score.npy")
score_min_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S5_score.npy")
score_min_mean = (score_min_s1+score_min_s2+score_min_s3+score_min_s4+score_min_s5)/5

score_lcb_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S1_score.npy")
score_lcb_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S2_score.npy")
score_lcb_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S3_score.npy")
score_lcb_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S4_score.npy")
score_lcb_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S5_score.npy")
score_lcb_mean = (score_lcb_s1+score_lcb_s2+score_lcb_s3+score_lcb_s4+score_lcb_s5)/5
score_lcb = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB0.5/EnsSize_10_S5_score.npy")

Ens10new_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/MeanNone/EnsSize_10_S5_score.npy")

plt.plot(x, score_first_s1[1::20], color='blue', alpha=0.2)
plt.plot(x, score_first_s2[1::20], color='blue', alpha=0.2)
plt.plot(x, score_first_s3[1::20], color='blue', alpha=0.2)
plt.plot(x, score_first_s4[1::20], color='blue', alpha=0.2)
plt.plot(x, score_first_s5[1::20], color='blue', alpha=0.2)

plt.plot(x, score_mean_s1[1::20], color='red', alpha=0.2)
plt.plot(x, score_mean_s2[1::20], color='red', alpha=0.2)
plt.plot(x, score_mean_s3[1::20], color='red', alpha=0.2)
plt.plot(x, score_mean_s4[1::20], color='red', alpha=0.2)
plt.plot(x, score_mean_s5[1::20], color='red', alpha=0.2)

plt.plot(x, score_min_s1[1::20], color='orange', alpha=0.2)
plt.plot(x, score_min_s2[1::20], color='orange', alpha=0.2)
plt.plot(x, score_min_s3[1::20], color='orange', alpha=0.2)
plt.plot(x, score_min_s4[1::20], color='orange', alpha=0.2)
plt.plot(x, score_min_s5[1::20], color='orange', alpha=0.2)

plt.plot(x, score_lcb_s1[1::20], color='green', alpha=0.2)
plt.plot(x, score_lcb_s2[1::20], color='green', alpha=0.2)
plt.plot(x, score_lcb_s3[1::20], color='green', alpha=0.2)
plt.plot(x, score_lcb_s4[1::20], color='green', alpha=0.2)
plt.plot(x, score_lcb_s5[1::20], color='green', alpha=0.2)

plt.plot(x, score_first_mean[1::20], color='blue', label='Qfilter')
plt.plot(x, score_mean_mean[1::20], color='red', label='Mean')
plt.plot(x, score_min_mean[1::20], color='orange', label='Minimum')
plt.plot(x, score_lcb_mean[1::20], color='green', label='LCB')
plt.plot(x, score_lcb[1::20], color='pink', label='LCB0.5')
# plt.plot(x, Ens10new_s5[1::20], color='pink', label='NEW')

plt.title('Scores')
plt.xlabel('Environment interactions (4e6)')
plt.ylabel('Score')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores_lcb0.5.png')
plt.show()
plt.close()

# success_first_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S1_success.npy")
# success_first_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S2_success.npy")
# success_first_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S3_success.npy")
# success_first_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S4_success.npy")
# success_first_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S5_success.npy")
# success_first_mean = (success_first_s1+success_first_s2+success_first_s3+success_first_s4+success_first_s5)/5
#
# success_mean_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S1_success.npy")
# success_mean_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S2_success.npy")
# success_mean_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S3_success.npy")
# success_mean_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S4_success.npy")
# success_mean_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S5_success.npy")
# success_mean_mean = (success_mean_s1+success_mean_s2+success_mean_s3+success_mean_s4+success_mean_s5)/5
#
# success_min_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S1_success.npy")
# success_min_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S2_success.npy")
# success_min_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S3_success.npy")
# success_min_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S4_success.npy")
# success_min_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Minimum/EnsSize_10_S5_success.npy")
# success_min_mean = (success_min_s1+success_min_s2+success_min_s3+success_min_s4+success_min_s5)/5
#
# success_lcb_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S1_success.npy")
# success_lcb_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S2_success.npy")
# success_lcb_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S3_success.npy")
# success_lcb_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S4_success.npy")
# success_lcb_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S5_success.npy")
# success_lcb_mean = (success_lcb_s1+success_lcb_s2+success_lcb_s3+success_lcb_s4+success_lcb_s5)/5
#
# plt.plot(x, success_first_s1[1::20], color='blue', alpha=0.2)
# plt.plot(x, success_first_s2[1::20], color='blue', alpha=0.2)
# plt.plot(x, success_first_s3[1::20], color='blue', alpha=0.2)
# plt.plot(x, success_first_s4[1::20], color='blue', alpha=0.2)
# plt.plot(x, success_first_s5[1::20], color='blue', alpha=0.2)
#
# plt.plot(x, success_mean_s1[1::20], color='red', alpha=0.2)
# plt.plot(x, success_mean_s2[1::20], color='red', alpha=0.2)
# plt.plot(x, success_mean_s3[1::20], color='red', alpha=0.2)
# plt.plot(x, success_mean_s4[1::20], color='red', alpha=0.2)
# plt.plot(x, success_mean_s5[1::20], color='red', alpha=0.2)
#
# plt.plot(x, success_min_s1[1::20], color='orange', alpha=0.2)
# plt.plot(x, success_min_s2[1::20], color='orange', alpha=0.2)
# plt.plot(x, success_min_s3[1::20], color='orange', alpha=0.2)
# plt.plot(x, success_min_s4[1::20], color='orange', alpha=0.2)
# plt.plot(x, success_min_s5[1::20], color='orange', alpha=0.2)
#
# plt.plot(x, success_lcb_s1[1::20], color='green', alpha=0.2)
# plt.plot(x, success_lcb_s2[1::20], color='green', alpha=0.2)
# plt.plot(x, success_lcb_s3[1::20], color='green', alpha=0.2)
# plt.plot(x, success_lcb_s4[1::20], color='green', alpha=0.2)
# plt.plot(x, success_lcb_s5[1::20], color='green', alpha=0.2)
#
# plt.plot(x, success_first_mean[1::20], color='blue', label='Qfilter')
# plt.plot(x, success_mean_mean[1::20], color='red', label='Mean')
# plt.plot(x, success_min_mean[1::20], color='orange', label='Minimum')
# plt.plot(x, success_lcb_mean[1::20], color='green', label='LCB')
#
# plt.title('Success rate of demonstrations')
# plt.xlabel('Environment interactions (2e6)')
# plt.ylabel('Success rate')
# plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/success_multiseeds.png')
# plt.show()


