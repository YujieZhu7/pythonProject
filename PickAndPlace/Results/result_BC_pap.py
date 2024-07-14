import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPickAndPlace'
x=np.arange(0, 2, 0.01)
# # Plot the scores
success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/RanNoise0.1/S5_success_1e6.npy")
plt.plot(x, success[1::20])

plt.title('Success rate of BC')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Success rate')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/PickAndPlace/BC/success_0.1.png')
plt.show()
plt.close()

score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/RanNoise0.1/S5_score_1e6.npy")
plt.plot(x, score[1::20])

plt.title('Score of BC')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Score')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/PickAndPlace/BC/scores_0.1.png')
plt.show()
plt.close()

loss = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/RanNoise0.1/S5_loss_1e6.npy")
plt.plot(x, loss[1::200])

plt.title('Loss of Actor by BC')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Loss')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/PickAndPlace/BC/loss_0.1.png')
plt.show()
plt.close()