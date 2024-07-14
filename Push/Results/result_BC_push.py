import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPush'
x=np.arange(0, 2, 0.01)
# # Plot the scores
success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/RandGausNoise/S5_success.npy")
plt.plot(x, success[1::20])

plt.title('Success rate of BC')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Success rate')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/BC/success_0.5+1.png')
plt.show()
plt.close()

score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/RandGausNoise/S5_score.npy")
plt.plot(x, score[1::20])

plt.title('Score of BC')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Score')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/BC/scores_0.5+1.png')
plt.show()
plt.close()

loss = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/RandGausNoise/S5_loss.npy")
plt.plot(x, loss[1::200])

plt.title('Loss of Actor by BC')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Loss')
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/BC/loss_0.5+1.png')
plt.show()
plt.close()
