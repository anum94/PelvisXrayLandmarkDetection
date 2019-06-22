import torch
import numpy as np

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def percentage_correct_keypoints(output, target):
    predictions = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    distance_threshold = np.linalg.norm(target.shape[2:4]) * 0.1  # 10% of image diagonal as distance_threshold
    target_landmarks_batch = [[np.unravel_index(np.argmax(i_target[idx], axis=None), i_target[idx].shape)
                               for idx in range(i_target.shape[0])] for i_target in target]
    true_positives = 0
    all_predictions = 0
    #prediction = predictions[-1]

    threshold = 0.01
    #pred_landmarks_batch = np.array(
    #    [gaussian_filter(prediction_channel, sigma=CONFIG['prediction_blur']) for prediction_channel in prediction]
    #)

    pred_landmarks_batch = [[np.unravel_index(np.argmax(i_output[idx], axis=None), i_output[idx].shape) for idx in
                             range(i_output.shape[0])] for i_output in predictions]

    for idx, (pred_landmarks, target_landmarks) in enumerate(
            zip(np.array(pred_landmarks_batch), np.array(target_landmarks_batch))):
        for channel_idx, (pred_landmark, target_landmark) in enumerate(zip(pred_landmarks, target_landmarks)):
            # check if either landmark is correctly predicted as not given OR predicted landmarks is within radius
            if (predictions[idx, channel_idx, pred_landmark[0], pred_landmark[1]] <= threshold
                    and np.sum(target_landmark) == 0
                    or np.linalg.norm(pred_landmark - target_landmark) <= distance_threshold):
                true_positives += 1

            all_predictions += 1

    return (true_positives / all_predictions) * 100

