# Methodology for Rider Intention Prediction

![Workflow Diagram Planning Whiteboard in Purple Blue Modern Professional Style](https://github.com/user-attachments/assets/96a2b0dc-b106-4a6a-bbe8-00418437343e)


## 1. Introduction

The Rider Intention Prediction (RIP) challenge aims to enhance the safety of two-wheeler riders by predicting their maneuvers. The challenge is divided into two tasks: single-view and multi-view predictions. Our best-performing models focus on both tasks, leveraging VGG-16 features provided by the competition organizers.

## 2. Data Preprocessing

### Data Structure

The dataset consists of features extracted from videos using the VGG-16 network. Each video corresponds to a specific maneuver such as left turn, right turn, lane changes, slow-stop, and going straight.

The data is organized into three views: frontal, left side mirror, and right side mirror. The features are stored as `.npy` files, with sequence lengths varying across samples.

### Handling Sequence Lengths

Given the variability in sequence lengths, we standardized them to a fixed length of 400. This length was determined based on an analysis of the dataset, which revealed that most of the sequences were covered within this length. Sequences shorter than 400 were padded with zeros, while longer sequences were truncated.

```python
def resize_sequence(self, features):
    if features.shape[0] < self.seq_len:
        diff = self.seq_len - features.shape[0]
        pad = torch.zeros(diff, features.shape[1])
        features = torch.cat((features, pad), dim=0)
    else:
        features = features[:self.seq_len, :]
    return features
```

### Handling Corrupt Files

During testing, we encountered a set of corrupt files that caused errors during data loading. These corrupt files were identified and removed manually to ensure the integrity of the dataset and smooth training and evaluation processes.

## 3. Model Architecture

We employed a CNN-LSTM model for both single-view and multi-view tasks, designed to capture both spatial and temporal features. The model consists of a 1D convolutional layer followed by batch normalization, ReLU activation, dropout, and an LSTM layer.

### CNN-LSTM Architecture

- **Convolutional Layer**: Extracts spatial features from the input sequence.
- **Batch Normalization**: Normalizes the output of the convolutional layer to accelerate training.
- **ReLU Activation**: Introduces non-linearity.
- **Dropout**: Prevents overfitting by randomly setting a fraction of the input units to zero during training.
- **LSTM Layer**: Captures temporal dependencies in the sequence data.
- **Fully Connected Layer**: Outputs the final class probabilities using a softmax function.

### Implementation

The architecture of the CNN-LSTM model is designed to handle both single-view and multi-view inputs. For the multi-view task, each view's features are processed independently by the convolutional layers and then concatenated before being passed to the LSTM layer.

## 4. Hyperparameters

Through extensive hyperparameter tuning, the following parameters were found to be the most effective:

- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Epochs**: 400
- **LSTM Layers**: 2
- **Hidden Size**: 128
- **Dropout Rate**: 0.25
- **Use Scheduler**: False

These hyperparameters were chosen based on their ability to balance model performance and generalization.

## 5. Results

### Single-View Task

For the single-view task, the model achieved a training accuracy of 91.48% and a validation accuracy of 66.96%. The use of VGG-16 features from the frontal view provided significant insights into the rider's maneuvers.

Final results for the single-view task:

- **Training Accuracy**: 91.48%
- **Training F1 Score**: 90.88%
- **Training Loss**: 1.12947
- **Validation Accuracy**: 66.96%
- **Validation F1 Score**: 64.67%

### Multi-View Task

For the multi-view task, the model achieved improved performance due to the additional spatial context provided by the side mirror views. The model's ability to capture features from multiple perspectives led to better maneuver predictions.

Final results for the multi-view task:

- **Training Accuracy**: 94.57%
- **Training F1 Score**: 94.22%
- **Training Loss**: 1.09822
- **Validation Accuracy**: 71.43%
- **Validation F1 Score**: 68.63%

## 6. Conclusion

The CNN-LSTM model effectively captures both spatial and temporal features from single-view and multi-view video sequences, making it well-suited for the rider intention prediction task. The multi-view approach, in particular, enhanced the model's performance by providing additional context. Future work can explore advanced architectures such as 3D CNNs and vision transformers to further enhance performance.
