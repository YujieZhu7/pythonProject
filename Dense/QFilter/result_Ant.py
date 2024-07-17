import numpy as np
import matplotlib.pyplot as plt
env_name = 'Ant'
x=np.arange(0, 1, 0.01)
# # Plot the scores
TD3_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_S5.npy")
Qfilter_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/QFilter_Rand0.1_S5_score.npy")

plt.plot(x, TD3_score[1::10], color='blue', label='TD3 score')
plt.plot(x, Qfilter_score[1::10], color='red', label='QFilter score')

plt.title('Score of Ant')
plt.xlabel('Environment interactions (1e6)')
plt.ylabel('Score')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Ant/scores_BCvsQfilter.png')
plt.show()
plt.close()

