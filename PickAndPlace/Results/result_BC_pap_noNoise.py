import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPickAndPlace'
x=np.arange(0, 0.5, 0.01)
# # Plot the scores
success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/noNoise/S5_success_5e5.npy")
plt.plot(x, success[1::20])

plt.title('Success rate of BC')
plt.xlabel('Environment interactions (5e5)')
plt.ylabel('Success rate')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/PickAndPlace/BC/success_no.png')
plt.show()
plt.close()

score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/noNoise/S5_score_5e5.npy")
plt.plot(x, score[1::20])

plt.title('Score of BC')
plt.xlabel('Environment interactions (5e5)')
plt.ylabel('Score')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/PickAndPlace/BC/scores_no.png')
plt.show()
plt.close()

loss = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/noNoise/S5_loss_5e5.npy")
plt.plot(x, loss[1::200])

plt.title('Loss of Actor by BC')
plt.xlabel('Environment interactions (5e5)')
plt.ylabel('Loss')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/PickAndPlace/BC/loss_no.png')
plt.show()
plt.close()