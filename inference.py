import torch
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

def predict(model, input, ans, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[ans]

    return predicted, expected


if __name__ == "__main__":
    #load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    #load mnist test dataset
    _, validation_data = download_mnist_datasets()

    #getting a sample for inference
    input, ans = validation_data[0][0], validation_data[0][1]

    #make an inference
    predicted, expected = predict(feed_forward_net, input, ans,
                                  class_mapping)

    print(f"prediction - {predicted}, answer - {expected}")