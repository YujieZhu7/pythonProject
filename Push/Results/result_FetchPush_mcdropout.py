import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPush'
x=np.arange(0, 4, 0.02)

# # Plot the scores
NoBC_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_score2.npy")
BC_only_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S5_score.npy")
Qfilter_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S5_score.npy")
mcdropout_mean_score2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.01/Mean/0.5+1EnsSize_2_S5_score.npy")
mcdropout_mean_score2_3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.03/Mean/0.5+1EnsSize_2_S5_score.npy")
mcdropout_first_score2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.01/First/0.5+1EnsSize_2_S5_score.npy")
mcdropout_first_score2_3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.03/First/0.5+1EnsSize_2_S5_score.npy")
mcdropout_mean_score4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden4/DropoutRate0.01/Mean/0.5+1EnsSize_2_S5_score.npy")
mcdropout_first_score4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden4/DropoutRate0.01/First/0.5+1EnsSize_2_S5_score.npy")


plt.plot(x, NoBC_score[:8000:40], color='black', label='NoBC')
plt.plot(x, BC_only_score[:8000:40], color='green', label='BC_only')
plt.plot(x, Qfilter_score[:8000:40], color='purple', label='Qfilter')
plt.plot(x, mcdropout_first_score2[:8000:40], color='blue', label='mcdropout_first_2hidden_0.01')
# plt.plot(x, mcdropout_first_score2_3[:8000:40], color='grey', label='mcdropout_first_2hidden_0.03')
plt.plot(x, mcdropout_first_score4[:8000:40], color='grey', label='mcdropout_first_4hidden_0.01')
plt.plot(x, mcdropout_mean_score2[:8000:40], color='red', label='mcdropout_mean_2hidden_0.01')
# plt.plot(x, mcdropout_mean_score2_3[:8000:40], color='orange', label='mcdropout_mean_2hidden_0.03')
plt.plot(x, mcdropout_mean_score4[:8000:40], color='orange', label='mcdropout_mean_4hidden_0.01')


plt.title('Scores')
plt.xlabel('Environment interactions (4e6)')
plt.ylabel('Score')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores_difflayers_dropout0.01.png')
plt.show()
plt.close()

Qfilter_demoAccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S5_demoaccept.npy")
mcdropout_mean_demoAccept2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.01/Mean/0.5+1EnsSize_2_S5_demoaccept.npy")
mcdropout_mean_demoAccept2_3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.03/Mean/0.5+1EnsSize_2_S5_demoaccept.npy")
mcdropout_first_demoAccept2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.01/First/0.5+1EnsSize_2_S5_demoaccept.npy")
mcdropout_first_demoAccept2_3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden2/DropoutRate0.03/First/0.5+1EnsSize_2_S5_demoaccept.npy")
mcdropout_mean_demoAccept4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden4/DropoutRate0.01/Mean/0.5+1EnsSize_2_S5_demoaccept.npy")
mcdropout_first_demoAccept4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/DropoutQfilter/hidden4/DropoutRate0.01/First/0.5+1EnsSize_2_S5_demoaccept.npy")

plt.plot(x, Qfilter_demoAccept[:8000:40], color='purple', label='Qfilter')
plt.plot(x, mcdropout_first_demoAccept2[:8000:40], color='blue', label='mcdropout_first_2hidden_0.01')
plt.plot(x, mcdropout_first_demoAccept2_3[:8000:40], color='grey', label='mcdropout_first_2hidden_0.03')
# plt.plot(x, mcdropout_first_demoAccept4[:8000:40], color='grey', label='mcdropout_first_4hidden_0.01')
plt.plot(x, mcdropout_mean_demoAccept2[:8000:40], color='red', label='mcdropout_mean_2hidden_0.01')
plt.plot(x, mcdropout_mean_demoAccept2_3[:8000:40], color='orange', label='mcdropout_mean_2hidden_0.03')
# plt.plot(x, mcdropout_mean_demoAccept4[:8000:40], color='orange', label='mcdropout_mean_4hidden_0.01')

plt.title('Acceptance rate of demonstrations')
plt.xlabel('Environment interactions (4e6)')
plt.ylabel('Acceptance rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/demoaccept_diffdropoutrate.png')
plt.show()
plt.close()
