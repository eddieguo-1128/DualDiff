import os 
import matplotlib.pyplot as plt

def plot_training_progress(history, log_dir):

    plt.figure(figsize=(12, 4))
    plt.style.use('tableau-colorblind10')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot([v * 100 for v in history["train_acc"]], label="Training Accuracy")
    plt.plot([v * 100 for v in history["val_acc"]], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.suptitle("Training Progress (Loss & Accuracy)")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(log_dir, 'training_progress.png'), dpi=300)
    plt.savefig(os.path.join(log_dir, 'training_progress.pdf'))
    
    # Show the figure if not running headless
    try:
        plt.show()
    except:
        pass