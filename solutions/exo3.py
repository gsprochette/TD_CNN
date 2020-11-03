plt.figure(figsize=(12, 6))

# first subplot: loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, c='r', label='Train')
plt.plot(test_loss, c='k', label='Test')
plt.yscale("log")
plt.title("Evolution of Loss during training, in log scale.")
plt.legend()

# second plot: accuracy
plt.subplot(1, 2, 2)
plt.plot(100 * train_accuracy, c='r', label='Train')
plt.plot(100 * test_accuracy, c='k', label='Test')
# plt.axhline(y=10, c='lightgray', ls='--', label='Random Guess: 10%')
plt.title("Evolution of Accuracy percentage during training.")
plt.ylim(top=100, bottom=90)
plt.legend()

plt.show()
