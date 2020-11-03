plt.figure(figsize=(12, 6))

# first subplot: loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, c='r', label='Train')
plt.plot(train_loss, c='k', label='Test')
plt.title("Evolution of Loss during training.")
plt.legend()

# second plot: accuracy
plt.subplot(1, 2, 2)
plt.plot(100 * train_accuracy, c='r', label='Train')
plt.plot(100 * train_accuracy, c='k', label='Test')
plt.axhline(y=10, c='lightgray', ls='--', label='Random Guess: 10%')
plt.title("Evolution of Accuracy percentage during training.")
plt.ylim(top=100, bottom=0)
plt.legend()

plt.show()
