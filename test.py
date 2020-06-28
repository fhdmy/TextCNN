import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from config import parse_config
from data_loader import DataBatchIterator

def test_model(model, data, config):
    model.eval()
    size = 5000
    total_loss = 0
    data_iter = iter(data)
    corr = 0
    for idx, batch in enumerate(data_iter):
        model.zero_grad()
        batch_label = batch.label
        outputs = model(batch.sent)
        result = torch.max(outputs, 1)[1]
        corr += (result.view(batch_label.size()).data == batch_label.data).sum()
    acc = 100.0 * corr / size
    return acc, corr, size


def main(config=None):
    # parse config
    if config==None:
        config = parse_config()
    # load test data
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
    )
    test_data.load()

    # import model
    model = torch.load('./results/model.pt')

    # test
    acc, corr, size = test_model(model, test_data, config)

    print(f'eval acc: {round(float(acc),4)}% ({corr}/{size})')
    return round(float(acc/100),2)


if __name__ == "__main__":
    main()