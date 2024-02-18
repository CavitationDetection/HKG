from main import train
from opts import parse_opts
from utils import create_dirs


# train and test our model
if __name__ == '__main__':
    opts = parse_opts()
    create_dirs("./output_results")
    create_dirs("./model_result")
    train(opts)

    
    